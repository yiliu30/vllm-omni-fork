# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Data-free online SVDQuant: BF16 checkpoint -> MXFP4 + rank-R branch at load.

The serialized path in ``svdquant_config.py`` needs an offline-quantized
checkpoint. This module derives everything from ordinary BF16 weights during
``process_weights_after_loading``, so a stock release can run W4A4 with no
calibration data and no producer tool::

    W ~= quant_mxfp4(W - L) + (proj_up @ proj_down.T)

``L`` is the top-R part of the quantization residual ``W - dequant(quant(W))``.
Each extra ``iterations`` pass quantizes ``W - L``, factors what that pass left
behind and *appends* it to ``L``, so the deployed branch is ``rank * iterations``
wide. Appending rather than replacing matters: replacing made extra passes lose
accuracy (0.1914 -> 0.1957 at rank 32).

Relative error against an FP32 matmul of the same BF16 weight, measured on XPU
on a 5120x5120 Wan linear with batch 256, by deployed branch rank:

    branch rank              0       16      32      64     128     256
    well conditioned       0.1627   0.1620   0.1613   0.1601   0.1581  0.1543
    outlier channels       0.2045   0.1972   0.1914   0.1821   0.1732  0.1660

The branch is real but modest: rank 32 removes 1% of the error on
well-conditioned weights and 6% on outlier-heavy ones, and rank 128 is needed
for 15%. Two things bound it. Weight and activation quantization contribute
about equally (weight-only error alone is 0.115 well conditioned, 0.171 with
outliers), and no weight-side branch can touch the activation half. And the
branch exists to carry per-channel outliers that smoothing pushes out of the
weight, while MX group-32 e8m0 scales already absorb those outliers, so
re-quantizing ``W - L`` stays about as lossy as quantizing ``W``. Treat this
path as a way to measure W4A4 headroom on a stock checkpoint, not as a shipping
quantization: an all-linear W4A4 Wan2.2-A14B run is visibly destroyed at rank 32.

Smoothing is deliberately disabled. ``smooth_factor`` stays all-ones and is
elided: without activation statistics a weight-derived scale shrinks outlier
weight channels while inflating activations inside the group-32 blocks, which
measured worse than doing nothing at all (0.46 versus 0.20).
Calibration-derived smoothing needs activation statistics, which this
data-free path does not have.

Chained FP4 terms are the lever that actually moves accuracy. The XPU quantizer
picks one power-of-two scale per group of 32 with ``ceil``, so every group pays
up to 2x the resolution it needs and no weight-side correction can reach that;
a measured continuous-scale oracle is 27% below ``ceil`` where the low-rank
branch buys 4% at best. ``weight_terms=2`` and ``act_terms=2`` instead ship a
second FP4 tensor per operand holding what the first left behind, so the product
of two operands carries up to four W4A4 GEMMs. Measured per-linear output error
on real Wan2.2 linears with activations drawn from the dumped per-channel
profiles (geometric mean over attention and FFN projections of block 0):

    terms (weight, activation)   1,1    2,1    1,2    2,2
    extra GEMMs                     0      1      1      3
    relative output error        0.172  0.131  0.113  0.020

Two terms per operand lands at 0.020, below the 0.033 of MXFP8 W8A8, at the cost
of three extra GEMMs; the branch is left in place because it is free.
"""

from __future__ import annotations

import functools
import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization._copy_missing_attrs import copy_missing_attrs
from vllm_omni.quantization.mxfp8_config import _LazyWeightMixin

if TYPE_CHECKING:
    from vllm_omni.quantization.svdquant_config import DiffusionSVDQuantConfig

logger = init_logger(__name__)

MXFP4_GROUP_SIZE = 32

# E2M1 decode table indexed by nibble value; bit 3 is the sign.
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# torch.svd_lowrank is randomized and this torch build rejects a caller-supplied
# projection, so determinism comes from seeding instead.
_SVD_SEED = 1234

# Set to 1 to check every quantized linear for non-finite activations and
# outputs. Each check synchronizes the device and the data-dependent branch
# breaks torch.compile graphs, so run the probe with --enforce-eager.
_FINITE_PROBE_ENV = "VLLM_OMNI_SVDQUANT_CHECK_NAN"


def _finite_probe_enabled() -> bool:
    return os.environ.get(_FINITE_PROBE_ENV, "") not in ("", "0")


def _rank_correction(x_2d: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    """``x @ proj_down @ proj_up.T``, the BF16 half of the W4A4+SVDQuant GEMM."""
    return torch.mm(torch.mm(x_2d, layer.proj_down), layer.proj_up.transpose(0, 1))


def _chained_weight_term(
    remaining: torch.Tensor,
    q_weight: torch.Tensor,
    w_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize what the shipped weight term left behind, for a second GEMM."""
    residual = remaining.float()
    residual.sub_(_mxfp4_dequant(q_weight, w_scales))
    term = _quantize_mxfp4(residual.to(torch.bfloat16))
    del residual
    return term


def _activation_residual(x_2d: torch.Tensor) -> torch.Tensor:
    """``x - dequant(quant(x))``, the part of the activation the first term missed.

    Handing this to the same GEMM quantizes it under its own block scales, which is
    the activation-side half of ``act_terms=2``.
    """
    q_act, a_scales = _quantize_mxfp4(x_2d)
    residual = x_2d.float()
    residual.sub_(_mxfp4_dequant(q_act, a_scales))
    del q_act, a_scales
    return residual.to(x_2d.dtype).contiguous()


def _kernel_operand(weight: torch.Tensor, weight_scale: torch.Tensor) -> Any:
    """View one packed term as the two attributes ``apply_weights`` reads."""
    return SimpleNamespace(weight=weight, weight_scale=weight_scale)


class DerivedWeight(NamedTuple):
    """Everything one BF16 weight is replaced by."""

    q_weight: torch.Tensor
    w_scales: torch.Tensor
    proj_up: torch.Tensor
    proj_down: torch.Tensor
    residual_term: tuple[torch.Tensor, torch.Tensor] | None


def _probe_nonfinite(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Log the first non-finite value per layer and tensor kind, then stay quiet."""
    if torch.isfinite(tensor).all():
        return
    reported = f"_svdquant_reported_{name}"
    if getattr(layer, reported, False):
        return
    setattr(layer, reported, True)
    logger.error(
        "SVDQuant MXFP4: non-finite %s in '%s' shape %s (%d of %d values, absmax %.3e)",
        name,
        getattr(layer, "prefix", "?"),
        tuple(tensor.shape),
        int((~torch.isfinite(tensor)).sum()),
        tensor.numel(),
        float(tensor.abs().max()) if tensor.numel() else float("nan"),
    )


def _assert_mxfp4_supported() -> None:
    if not current_omni_platform.is_xpu():
        raise RuntimeError(
            "SVDQuant MXFP4 online quantization is implemented for XPU only; use "
            "precision='nvfp4' with a serialized checkpoint on other platforms."
        )


def _quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight along K into packed FP4 plus group-32 e8m0 scales.

    Delegates to the same op ``XPUMxFp4LinearKernel`` uses for activations, so the
    derived weight and the GEMM cannot disagree about rounding. Importing
    ``vllm._xpu_ops`` is what registers ``torch.ops.vllm.xpu_mxfp4_quantize``;
    ``mxfp4_utils`` does not import it itself.
    """
    import vllm._xpu_ops  # noqa: F401  (registers the custom op)
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import xpu_mxfp4_quantize

    return xpu_mxfp4_quantize(x)


@functools.cache
def _mxfp4_kernel() -> Any:
    """Select the MXFP4 GEMM backend, accepting only the true W4A4 XPU kernel."""
    import vllm._xpu_ops  # noqa: F401
    from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
    from vllm.model_executor.kernels.linear.mxfp4.xpu import XPUMxFp4LinearKernel

    kernel = init_mxfp4_linear_kernel(kMxfp4Dynamic)
    if not isinstance(kernel, XPUMxFp4LinearKernel):
        raise RuntimeError(
            "SVDQuant MXFP4 online quantization needs the XPU MXFP4 W4A4 kernel; "
            f"selected {type(kernel).__name__}. Other MXFP4 backends are weight-only "
            "and would not match the derived scales."
        )
    return kernel


def _mxfp4_dequant(qweight: torch.Tensor, wscales: torch.Tensor) -> torch.Tensor:
    """Dequantize packed MXFP4 back to float32 for residual computation.

    ``quant_dequant_mxfp4`` is unusable here because it requires ``amd-quark``, so
    the decode is plain torch. The low nibble holds the even-indexed element,
    matching vLLM's ``break_fp4_bytes``.
    """
    packed = qweight.view(torch.uint8).to(torch.int64)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    values = table[codes]

    if wscales.dtype is torch.uint8:
        scales = torch.exp2(wscales.to(torch.float32) - 127.0)
    else:
        scales = wscales.to(torch.float32)
    return values * scales.repeat_interleave(MXFP4_GROUP_SIZE, dim=1)


def _rank_r_factor(
    residual: torch.Tensor,
    rank: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``residual ~= proj_up @ proj_down.T`` keeping the top ``rank``."""
    device = residual.device
    n, k = residual.shape
    # torch LAPACK already runs on a CPU fallback on XPU, so going to host
    # explicitly is no slower and avoids a warning per layer.
    work = residual.detach().to(dtype=torch.float32, device="cpu").contiguous()
    q = min(rank + 8, min(n, k))
    torch.manual_seed(_SVD_SEED)
    u, s, v = torch.svd_lowrank(work, q=q, niter=niter)

    root = s[:rank].sqrt()
    proj_up = (u[:, :rank] * root).to(torch.bfloat16).to(device)
    proj_down = (v[:, :rank] * root).to(torch.bfloat16).to(device)
    return proj_up, proj_down


class SVDQuantOnlineLinearMethod(_LazyWeightMixin, LinearMethodBase):
    """Derive MXFP4 weights and the rank-R correction from BF16 at load time.

    MRO: ``SVDQuantOnlineLinearMethod -> _LazyWeightMixin -> LinearMethodBase``

      create_weights  : _LazyWeightMixin (meta device + patched loader)
      process_weights : this class (BF16 -> MXFP4 + rank branch -> kernel layout)
      apply           : this class (MXFP4 GEMM + BF16 rank correction)

    The serialized NVFP4 method is deliberately not reused: it expects
    checkpoint-supplied ``qweight``/``wscales``/``wtscale`` tensors and an
    alpha-carrying NVFP4 kernel, whereas MXFP4 needs neither a global scale nor a
    scale swizzle.
    """

    def __init__(self, quant_config: DiffusionSVDQuantConfig) -> None:
        _assert_mxfp4_supported()
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if input_size_per_partition % MXFP4_GROUP_SIZE != 0:
            prefix = getattr(layer, "prefix", "?")
            raise ValueError(
                f"SVDQuant MXFP4 requires each input partition to be divisible by the "
                f"group size {MXFP4_GROUP_SIZE}; got {input_size_per_partition} for "
                f"'{prefix}'"
            )
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def _derive(self, layer: torch.nn.Module):
        """Produce the FP4 terms and the low-rank branch for one BF16 weight."""
        quant_config = self.quant_config
        rank = quant_config.rank
        effective_rank = rank * quant_config.iterations
        original = layer.weight.detach().to(dtype=torch.bfloat16).contiguous()
        n, k = original.shape
        if effective_rank >= min(n, k):
            prefix = getattr(layer, "prefix", "?")
            raise ValueError(
                f"SVDQuant rank {rank} times {quant_config.iterations} iterations = "
                f"{effective_rank} must stay below min(N, K) = {min(n, k)} for '{prefix}'"
            )

        if effective_rank == 0:
            # Branchless control: identical derivation, packer and kernel, only the
            # correction is gone, so any rank can be compared against pure MXFP4.
            q_weight, w_scales = _quantize_mxfp4(original)
            layer.svdquant_residual_capture = 0.0
            term = _chained_weight_term(original, q_weight, w_scales) if quant_config.weight_terms > 1 else None
            return DerivedWeight(q_weight, w_scales, original.new_empty((n, 0)), original.new_empty((k, 0)), term)

        remaining = original
        branch_up: torch.Tensor | None = None
        branch_down: torch.Tensor | None = None
        first_residual_norm = 0.0
        branch_norm = 0.0

        for iteration in range(quant_config.iterations):
            q_weight, w_scales = _quantize_mxfp4(remaining)
            residual = remaining.float()
            residual.sub_(_mxfp4_dequant(q_weight, w_scales))
            if iteration == 0:
                first_residual_norm = float(residual.norm())
            proj_up, proj_down = _rank_r_factor(residual, rank, quant_config.svd_niter)
            del residual, q_weight, w_scales
            # Accumulate rather than replace. Each pass corrects what the branch so
            # far left behind, so a pass is worth roughly ``rank`` more correction
            # columns; replacing the branch instead made extra passes lose accuracy.
            branch_up = proj_up if branch_up is None else torch.cat((branch_up, proj_up), dim=1)
            branch_down = proj_down if branch_down is None else torch.cat((branch_down, proj_down), dim=1)
            branch = branch_up.float() @ branch_down.float().t()
            branch_norm = float(branch.norm())
            remaining = (original.float() - branch).to(torch.bfloat16)
            del branch, proj_up, proj_down

        # The base GEMM must not re-supply what the branch already covers, so the
        # shipped weight is quantized against the accumulated branch.
        q_weight, w_scales = _quantize_mxfp4(remaining)
        layer.svdquant_residual_capture = branch_norm / first_residual_norm if first_residual_norm > 0 else float("nan")
        term = _chained_weight_term(remaining, q_weight, w_scales) if quant_config.weight_terms > 1 else None
        del remaining
        assert branch_up is not None and branch_down is not None
        return DerivedWeight(q_weight, w_scales, branch_up, branch_down, term)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # A weight that never received a load chunk (profiling or dummy passes)
        # is still on the meta device; materialize it the way the online MXFP
        # methods do before deriving from it.
        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        started = time.perf_counter()
        derived = self._derive(layer)
        q_weight, w_scales, proj_up, proj_down, residual_term = derived

        replace_parameter(layer, "weight", q_weight.view(torch.uint8))
        layer.register_parameter("weight_scale", Parameter(w_scales.view(torch.uint8), requires_grad=False))
        layer.register_parameter("proj_up", Parameter(proj_up, requires_grad=False))
        layer.register_parameter("proj_down", Parameter(proj_down, requires_grad=False))
        if residual_term is None:
            layer.residual_weight = None
            layer.residual_weight_scale = None
        else:
            # The kernel owns the operand layout, so the extra term goes through the
            # same transform instead of reimplementing it here. The op hands back FP4
            # and e8m0 views; the kernel expects to reinterpret raw bytes.
            shim = _kernel_operand(
                residual_term[0].view(torch.uint8),
                residual_term[1].view(torch.uint8),
            )
            _mxfp4_kernel().process_weights_after_loading(shim)
            layer.register_parameter("residual_weight", shim.weight)
            layer.register_parameter("residual_weight_scale", shim.weight_scale)
        # Online quantization has no smoothing and no per-output outer scale, so
        # apply() skips both epilogues entirely.
        layer.smooth_factor = None
        layer.output_channel_scale = None

        # The kernel transposes into its own layout; MXFP4 swizzles nothing.
        _mxfp4_kernel().process_weights_after_loading(layer)
        # The online path is not idempotent: a second derive would quantize the
        # packed FP4 bytes as if they were BF16. The lazy loader sets this flag too,
        # but the method must hold the invariant itself.
        layer._already_called_process_weights_after_loading = True
        layer.online_derive_seconds = time.perf_counter() - started

        self.quant_config.online_derived_layers += 1
        self.quant_config.online_derive_seconds += layer.online_derive_seconds
        logger.debug(
            "SVDQuant MXFP4 online: weight %s %s, scale %s %s, branch rank %d, weight terms %d, "
            "activation terms %d, residual captured %.3f, %.3fs",
            tuple(layer.weight.shape),
            layer.weight.dtype,
            tuple(layer.weight_scale.shape),
            layer.weight_scale.dtype,
            self.quant_config.rank * self.quant_config.iterations,
            self.quant_config.weight_terms,
            self.quant_config.act_terms,
            layer.svdquant_residual_capture,
            layer.online_derive_seconds,
        )
        logger.info_once(
            "SVDQuant is deriving MXFP4 weights plus a rank-%d correction at load time; "
            "budget roughly 0.2-0.4s per linear.",
            self.quant_config.rank * self.quant_config.iterations,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute every FP4 term plus the BF16 rank correction."""
        if x.dtype is not torch.bfloat16:
            raise ValueError(f"SVDQuant MXFP4 requires BF16 activations; got {x.dtype}")

        original_shape = x.shape
        # The MXFP4 activation quantizer requires contiguous groups.
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()

        probe = _finite_probe_enabled()
        if probe:
            _probe_nonfinite(layer, "activations", x_2d)

        kernel = _mxfp4_kernel()
        out = kernel.apply_weights(layer=layer, x=x_2d, bias=None)
        if probe:
            _probe_nonfinite(layer, "base gemm", out)

        # Chained terms: the product of the operand expansions needs one GEMM per
        # pair, minus the base GEMM already computed above. Each operand re-quantizes
        # its own activations, which costs a pass over x and is noise next to the GEMM.
        residual_weight = getattr(layer, "residual_weight", None)
        activations = [x_2d]
        if self.quant_config.act_terms > 1:
            activations.append(_activation_residual(x_2d))
        weights = [(layer.weight, layer.weight_scale)]
        if residual_weight is not None:
            weights.append((residual_weight, layer.residual_weight_scale))
        for weight_index, (weight, weight_scale) in enumerate(weights):
            for activation_index, activation in enumerate(activations):
                if weight_index == 0 and activation_index == 0:
                    continue
                out = out + kernel.apply_weights(layer=_kernel_operand(weight, weight_scale), x=activation, bias=None)
                if probe:
                    _probe_nonfinite(layer, "residual gemm", out)

        if layer.proj_down.shape[1]:
            out = out + _rank_correction(x_2d, layer)
        if probe:
            _probe_nonfinite(layer, "output", out)
        if bias is not None:
            out.add_(bias)
        return out.reshape(*original_shape[:-1], layer.output_size_per_partition)


__all__ = ["SVDQuantOnlineLinearMethod"]
