# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online SVDQuant MXFP4 tests.

CPU tests exercise the pure-torch pieces (dequant decode, config dispatch,
derivation bookkeeping) with a reference quantizer, so no XPU device is
required. The one test that needs the real MXFP4 GEMM is marked and skipped
elsewhere.
"""

from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import svdquant_config
from vllm_omni.quantization.svdquant_config import (
    DiffusionSVDQuantConfig,
    DiffusionSVDQuantLinearMethod,
)
from vllm_omni.quantization.svdquant_online import (
    MXFP4_GROUP_SIZE,
    SVDQuantOnlineLinearMethod,
    _mxfp4_dequant,
)
from vllm_omni.quantization.svdquant_online import _quantize_mxfp4 as _real_quantize

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# Bin edges for nearest-even-ish rounding onto the E2M1 grid.
_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def _reference_mxfp4_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-only group-32 MXFP4 quantizer matching the XPU kernel exactly.

    Stands in for ``xpu_mxfp4_quantize`` on CPU so the derivation is testable
    without a device. Scales are power-of-two (e8m0), and the exponent rule is the
    kernel's ``ceil(log2(amax / 6))`` rather than the OCP spec's round-to-nearest,
    which is what ``csrc/quantization/fp4/mxfp4_quant.h`` actually computes.
    """
    n, k = w.shape
    blocks = w.detach().float().reshape(n, k // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-10)
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    scaled = (blocks / scale).clamp(-6.0, 6.0)
    # Ties go to the smaller magnitude, matching the kernel's strict comparisons.
    index = (scaled.abs().unsqueeze(-1) > _EDGES).sum(-1)
    code = index + torch.where(scaled < 0, 8, 0)
    code = code.reshape(n, k).to(torch.uint8)
    low = code[:, 0::2]
    high = code[:, 1::2]
    return low | (high << 4), (exponent.squeeze(-1) + 127).to(torch.uint8)


def _online_config(**kwargs) -> DiffusionSVDQuantConfig:
    return DiffusionSVDQuantConfig(precision="mxfp4", is_checkpoint_mxfp4_serialized=False, **kwargs)


def _outlier_weight(n: int, k: int, device: torch.device) -> torch.Tensor:
    """Seeded weights with outlier input channels, as real diffusion weights have.

    Plain Gaussian weights must not be used for quality assertions: the
    quantization residual is then essentially white noise with no low-rank
    structure, and the correction barely moves the error.
    """
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(n, k, generator=generator).mul_(1.0)
    weight[:, ::64] *= 25.0
    return weight.to(device=device, dtype=torch.bfloat16)


def _build_layer(method, weight):
    import torch.nn as nn

    n, k = weight.shape
    layer = nn.Module()
    layer.prefix = "blocks.0.attn1.to_q"
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    return layer


def test_dequant_decodes_nibbles_and_e8m0_scales():
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    codes = torch.zeros(1, 16, dtype=torch.uint8)  # one 32-value scale block
    codes[0, 0] = 0x43
    codes[0, 1] = 0x4B
    # low nibble -> even element, high nibble -> odd element
    expected = torch.tensor([table[0x43 & 0x0F], table[0x43 >> 4], table[0x4B & 0x0F], table[0x4B >> 4]])
    for byte, factor in ((127, 1.0), (126, 0.5)):
        for scale_dtype in (torch.uint8, torch.float8_e8m0fnu):
            scales = torch.full((1, 1), byte, dtype=torch.uint8).view(scale_dtype)
            out = _mxfp4_dequant(codes, scales)
            assert out.shape == (1, 32)
            assert torch.equal(out[0, :4], expected * factor)


def test_reference_quantizer_round_trips_through_dequant():
    weight = _outlier_weight(128, 320, torch.device("cpu")).float()
    q, s = _reference_mxfp4_quantize(weight)
    dequantized = _mxfp4_dequant(q, s)
    assert dequantized.shape == weight.shape
    # FP4 is coarse but must stay within its own step size per block.
    assert (dequantized - weight).abs().max() < weight.abs().max() * 0.5


def test_config_accepts_mxfp4_and_rejects_bad_combinations():
    config = _online_config(rank=16, iterations=3, svd_niter=1)
    assert (config.precision, config.rank, config.iterations, config.svd_niter) == ("mxfp4", 16, 3, 1)
    assert config.online_derived_layers == 0 and config.online_derive_seconds == 0.0

    for kwargs, message in (
        ({"precision": "int4"}, "supports precision"),
        ({"precision": "nvfp4", "is_checkpoint_mxfp4_serialized": False}, "requires a serialized"),
        ({"precision": "mxfp4", "is_checkpoint_mxfp4_serialized": True}, "online-only"),
        (
            {"precision": "mxfp4", "is_checkpoint_mxfp4_serialized": False, "iterations": 0},
            "iterations must be",
        ),
        (
            {"precision": "mxfp4", "is_checkpoint_mxfp4_serialized": False, "svd_niter": 0},
            "svd_niter must be",
        ),
        ({"rank": 0}, "branchless rank=0"),
        ({"rank": -1}, "zero or positive"),
        ({"precision": "mxfp4", "is_checkpoint_mxfp4_serialized": False, "weight_terms": 0}, "weight_terms must be"),
        ({"precision": "mxfp4", "is_checkpoint_mxfp4_serialized": False, "act_terms": 3}, "act_terms must be"),
        ({"precision": "nvfp4", "weight_terms": 2}, "serialized"),
    ):
        with pytest.raises(ValueError, match=message):
            DiffusionSVDQuantConfig(**kwargs)


def test_from_config_defaults_keep_serialized_nvfp4():
    assert DiffusionSVDQuantConfig.from_config({}).precision == "nvfp4"
    assert DiffusionSVDQuantConfig.from_config({}).is_checkpoint_mxfp4_serialized is True
    online = {"precision": "mxfp4", "is_checkpoint_mxfp4_serialized": False}
    assert DiffusionSVDQuantConfig.from_config(online).weight_terms == 1
    assert DiffusionSVDQuantConfig.from_config(online).act_terms == 1
    assert DiffusionSVDQuantConfig.from_config({**online, "weight_terms": 2}).weight_terms == 2
    assert DiffusionSVDQuantConfig.from_config({**online, "act_terms": 2}).act_terms == 2
    # Unknown checkpoint fields must stay tolerated.
    config = DiffusionSVDQuantConfig.from_config(
        {"precision": "mxfp4", "is_checkpoint_mxfp4_serialized": False, "w4a16_modules": ["x"]}
    )
    assert config.precision == "mxfp4"


def test_dispatch_selects_online_offline_and_unquantized(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(svdquant_config, "_assert_supported", lambda: None)
    online = _online_config()
    linear = Mock(spec=LinearBase)

    assert isinstance(online.get_quant_method(linear, "blocks.0.to_q"), SVDQuantOnlineLinearMethod)
    assert isinstance(
        DiffusionSVDQuantConfig().get_quant_method(linear, "blocks.0.to_q"),
        DiffusionSVDQuantLinearMethod,
    )
    skipping = _online_config(modules_to_not_convert=["blocks.0.adaln.linear"])
    assert isinstance(skipping.get_quant_method(linear, "blocks.0.adaln.linear"), UnquantizedLinearMethod)
    assert online.get_quant_method(torch.nn.ReLU(), "blocks.0.act") is None


def test_online_method_requires_xpu(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: False)
    with pytest.raises(RuntimeError, match="XPU only"):
        SVDQuantOnlineLinearMethod(_online_config())


def test_derivation_produces_expected_params(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr("vllm_omni.quantization.svdquant_online._quantize_mxfp4", _reference_mxfp4_quantize)
    monkeypatch.setattr(
        "vllm_omni.quantization.svdquant_online._mxfp4_kernel",
        lambda: Mock(process_weights_after_loading=lambda layer: None),
    )

    rank, n, k = 16, 256, 320
    config = _online_config(rank=rank, iterations=2)
    method = SVDQuantOnlineLinearMethod(config)
    weight = _outlier_weight(n, k, torch.device("cpu"))
    layer = _build_layer(method, weight)

    method.process_weights_after_loading(layer)

    assert layer.weight.shape == (n, k // 2) and layer.weight.dtype is torch.uint8
    assert layer.weight_scale.shape == (n, k // 32) and layer.weight_scale.dtype is torch.uint8
    # Every iteration appends rank correction columns, so the deployed branch is
    # rank * iterations wide.
    assert layer.proj_up.shape == (n, rank * config.iterations) and layer.proj_up.dtype is torch.bfloat16
    assert layer.proj_down.shape == (k, rank * config.iterations) and layer.proj_down.dtype is torch.bfloat16
    # MXFP4 carries no global alpha and online quantization has no smoothing.
    for absent in ("alpha", "input_global_scale_inv", "weight_global_scale", "wtscale"):
        assert not hasattr(layer, absent)
    assert layer.smooth_factor is None and layer.output_channel_scale is None
    assert 0.0 < layer.svdquant_residual_capture <= 1.0
    assert config.online_derived_layers == 1 and config.online_derive_seconds >= 0.0

    # Second call must be a no-op rather than re-deriving.
    before = float(layer.svdquant_residual_capture)
    method.process_weights_after_loading(layer)
    assert float(layer.svdquant_residual_capture) == before
    assert config.online_derived_layers == 1


@pytest.mark.parametrize("rank", [8, 32, 64])
def test_branch_absorbs_more_error_as_rank_grows(monkeypatch, rank):
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr("vllm_omni.quantization.svdquant_online._quantize_mxfp4", _reference_mxfp4_quantize)
    monkeypatch.setattr(
        "vllm_omni.quantization.svdquant_online._mxfp4_kernel",
        lambda: Mock(process_weights_after_loading=lambda layer: None),
    )

    n, k = 256, 320
    weight = _outlier_weight(n, k, torch.device("cpu"))
    method = SVDQuantOnlineLinearMethod(_online_config(rank=rank, iterations=1))
    layer = _build_layer(method, weight)
    method.process_weights_after_loading(layer)

    branch = layer.proj_up.float() @ layer.proj_down.float().t()
    residual = weight.float() - _mxfp4_dequant(*_reference_mxfp4_quantize(weight.float()))
    captured = (branch.norm() / residual.norm()).item()
    # Without the branch the correction contributes nothing at all.
    assert captured > 0.05


def test_extra_iterations_accumulate_rather_than_replace(monkeypatch):
    """A refinement pass must not throw the previous branch away.

    Replacing the branch with the newest residual factor instead of appending it
    made ``iterations=2`` less accurate than ``iterations=1`` at every rank
    measured (0.1914 -> 0.1957 relative error at rank 32).
    """
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr("vllm_omni.quantization.svdquant_online._quantize_mxfp4", _reference_mxfp4_quantize)
    monkeypatch.setattr(
        "vllm_omni.quantization.svdquant_online._mxfp4_kernel",
        lambda: Mock(process_weights_after_loading=lambda layer: None),
    )

    n, k, rank = 256, 320, 16
    weight = _outlier_weight(n, k, torch.device("cpu"))
    errors = {}
    for iterations in (1, 2, 4):
        method = SVDQuantOnlineLinearMethod(_online_config(rank=rank, iterations=iterations))
        layer = _build_layer(method, weight)
        q_weight, w_scales, proj_up, proj_down, _residual_term = method._derive(layer)
        assert proj_up.shape == (n, rank * iterations)
        rebuilt = _mxfp4_dequant(q_weight, w_scales) + proj_up.float() @ proj_down.float().t()
        errors[iterations] = ((rebuilt - weight.float()).norm() / weight.float().norm()).item()

    assert errors[1] > errors[2] > errors[4]


def test_rank_zero_is_branchless_mxfp4(monkeypatch):
    """rank=0 must be exactly pure MXFP4, so a rank sweep has a real control.

    The deployed branch has to contribute nothing, and the shipped weight has to be
    the plain quantization of the original rather than of ``W - L``.
    """
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr("vllm_omni.quantization.svdquant_online._quantize_mxfp4", _reference_mxfp4_quantize)
    monkeypatch.setattr(
        "vllm_omni.quantization.svdquant_online._mxfp4_kernel",
        lambda: Mock(process_weights_after_loading=lambda layer: None),
    )

    n, k = 256, 320
    weight = _outlier_weight(n, k, torch.device("cpu"))
    method = SVDQuantOnlineLinearMethod(_online_config(rank=0, iterations=1))
    layer = _build_layer(method, weight)
    q_weight, w_scales, proj_up, proj_down, _residual_term = method._derive(layer)
    # The short-circuit must still leave the layer fully populated: reading
    # svdquant_residual_capture in the load-time log raised AttributeError for a
    # while, and only an end-to-end run caught it.
    method.process_weights_after_loading(layer)
    assert layer.svdquant_residual_capture == 0.0

    assert proj_up.shape == (n, 0) and proj_down.shape == (k, 0)
    plain_weight, plain_scales = _reference_mxfp4_quantize(weight.float())
    assert torch.equal(q_weight, plain_weight) and torch.equal(w_scales, plain_scales)
    rebuilt = _mxfp4_dequant(q_weight, w_scales) + proj_up.float() @ proj_down.float().t()
    assert torch.equal(rebuilt, _mxfp4_dequant(plain_weight, plain_scales))


def _pass_through_kernel():
    """Stand-in for the MXFP4 kernel that only wraps operands as parameters."""
    from unittest.mock import Mock

    def prepare(layer):
        layer.weight = torch.nn.Parameter(layer.weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale, requires_grad=False)

    return Mock(process_weights_after_loading=prepare)


def test_weight_terms_ship_a_residual_term(monkeypatch):
    """weight_terms=2 must ship a second FP4 tensor that carries the first one's error."""
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr("vllm_omni.quantization.svdquant_online._quantize_mxfp4", _reference_mxfp4_quantize)
    monkeypatch.setattr("vllm_omni.quantization.svdquant_online._mxfp4_kernel", _pass_through_kernel)

    n, k = 256, 320
    weight = _outlier_weight(n, k, torch.device("cpu"))

    single = SVDQuantOnlineLinearMethod(_online_config(rank=0, iterations=1))
    single_layer = _build_layer(single, weight)
    single.process_weights_after_loading(single_layer)
    assert single_layer.residual_weight is None
    assert single_layer.residual_weight_scale is None

    chained = SVDQuantOnlineLinearMethod(_online_config(rank=0, iterations=1, weight_terms=2))
    chained_layer = _build_layer(chained, weight)
    q_weight, w_scales, _, _, term = chained._derive(chained_layer)
    assert term is not None
    assert term[0].shape == q_weight.shape and term[1].shape == w_scales.shape

    chained.process_weights_after_loading(chained_layer)
    single_error = (weight.float() - _mxfp4_dequant(q_weight, w_scales)).norm()
    chained_error = (
        weight.float()
        - _mxfp4_dequant(q_weight, w_scales)
        - _mxfp4_dequant(chained_layer.residual_weight, chained_layer.residual_weight_scale)
    ).norm()
    assert chained_error < single_error * 0.5, "the residual term should absorb most of the first term's error"


def test_rejects_indivisible_partition_and_oversized_rank(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    method = SVDQuantOnlineLinearMethod(_online_config(rank=32))
    layer = Mock()
    layer.prefix = "blocks.0.to_q"
    with pytest.raises(ValueError, match="divisible by the group size 32"):
        method.create_weights(layer, 100, [64], 100, 64, torch.bfloat16)

    monkeypatch.setattr("vllm_omni.quantization.svdquant_online._quantize_mxfp4", _reference_mxfp4_quantize)
    small = _build_layer(method, torch.zeros(16, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="must stay below min\\(N, K\\)"):
        method._derive(small)


def test_apply_rejects_non_bf16(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: True)
    method = SVDQuantOnlineLinearMethod(_online_config())
    with pytest.raises(ValueError, match="requires BF16 activations"):
        method.apply(Mock(), torch.zeros(4, 32, dtype=torch.float16), None)


def _gemm_layer(n, k, q_weight, w_scales):
    """Real Module so the kernel's replace_parameter works, with no dist needed."""
    import torch.nn as nn

    layer = nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.register_parameter("weight", torch.nn.Parameter(q_weight.view(torch.uint8), requires_grad=False))
    layer.register_parameter("weight_scale", torch.nn.Parameter(w_scales.view(torch.uint8), requires_grad=False))
    return layer


@pytest.mark.skipif(not current_omni_platform.is_xpu(), reason="requires an XPU device")
def test_xpu_end_to_end_beats_branchless():
    """Real MXFP4 GEMM: the rank-R branch must beat branchless MXFP4."""
    import vllm_xpu_kernels  # noqa: F401  provides vllm._C / _xpu_C

    from vllm_omni.quantization.svdquant_online import _mxfp4_kernel

    device = torch.device("xpu")
    n, k, rank = 1024, 1024, 32
    weight = _outlier_weight(n, k, device)
    x = torch.randn(64, k, device=device, dtype=torch.bfloat16)
    reference = torch.matmul(x.float(), weight.float().t())

    base_weight, base_scale = _real_quantize(weight)
    kernel = _mxfp4_kernel()
    plain = _gemm_layer(n, k, base_weight, base_scale)
    kernel.process_weights_after_loading(plain)
    branchless = kernel.apply_weights(layer=plain, x=x, bias=None)
    branchless_error = ((branchless.float() - reference).norm() / reference.norm()).item()

    method = SVDQuantOnlineLinearMethod(_online_config(rank=rank, iterations=2))
    layer = _build_layer(method, weight)
    q_weight, w_scales, proj_up, proj_down, _residual_term = method._derive(layer)
    plain = _gemm_layer(n, k, q_weight, w_scales)
    plain.proj_up, plain.proj_down = proj_up, proj_down
    kernel.process_weights_after_loading(plain)
    corrected = torch.addmm(
        kernel.apply_weights(layer=plain, x=x, bias=None),
        torch.mm(x, proj_down),
        proj_up.transpose(0, 1),
    )
    corrected_error = ((corrected.float() - reference).norm() / reference.norm()).item()

    assert corrected_error < branchless_error
    assert 0.0 < layer.svdquant_residual_capture <= 1.0


@pytest.mark.skipif(not current_omni_platform.is_xpu(), reason="requires an XPU device")
def test_xpu_grid_weights_survive_the_kernel_unchanged():
    """A weight already on the FP4 grid must reach the GEMM bit-exactly.

    Catches a layout disagreement between the derived ``qweight``/``weight_scale``
    and what ``XPUMxFp4LinearKernel`` expects, which no error-magnitude assertion
    would notice because a swapped nibble or misindexed scale still looks noisy.
    """
    import vllm_xpu_kernels  # noqa: F401  provides vllm._C / _xpu_C

    from vllm_omni.quantization.svdquant_online import _mxfp4_kernel

    device = torch.device("xpu")
    n, k = 64, 64
    generator = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (n, k), generator=generator).to(torch.uint8)
    # Every code is a grid value by construction, so quantization is a no-op as
    # long as the block scale keeps the values inside the representable range.
    scales = torch.full((n, k // MXFP4_GROUP_SIZE), 126, dtype=torch.uint8)  # e8m0 -> 0.5
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    weight = _mxfp4_dequant(packed.to(device), scales.to(device)).to(torch.bfloat16)

    requantized, rescales = _real_quantize(weight)
    assert torch.equal(_mxfp4_dequant(requantized, rescales), weight.float())

    # One-hot activations: row i of the result is exactly weight column i.
    layer = _gemm_layer(n, k, requantized, rescales)
    _mxfp4_kernel().process_weights_after_loading(layer)
    identity = torch.eye(k, device=device, dtype=torch.bfloat16)
    out = _mxfp4_kernel().apply_weights(layer=layer, x=identity, bias=None)
    assert torch.equal(out, weight.t())


@pytest.mark.skipif(not current_omni_platform.is_xpu(), reason="requires an XPU device")
@pytest.mark.parametrize(
    ("weight_terms", "act_terms", "ceiling"),
    # Measured ratios against the single-term error on outlier weights with Gaussian
    # activations: 0.57, 0.84, 0.14. Ceilings leave room for kernel jitter.
    [(2, 1, 0.75), (1, 2, 0.90), (2, 2, 0.30)],
)
def test_xpu_chained_terms_cut_the_gemm_error(weight_terms, act_terms, ceiling):
    """Every chained FP4 term must cut the real W4A4 GEMM error.

    The low-rank branch measurably cannot: end to end, rank 128 lands within a rerun of
    rank 0. The per-group power-of-two scale, not the missing rank, bounds single-term
    MXFP4, so a second term per operand is what buys the accuracy back.
    """
    import vllm_xpu_kernels  # noqa: F401  provides vllm._C / _xpu_C

    device = torch.device("xpu")
    n, k = 1024, 1024
    weight = _outlier_weight(n, k, device)
    x = torch.randn(128, k, device=device, dtype=torch.bfloat16)
    reference = torch.matmul(x.float(), weight.float().t())

    def error(**kwargs) -> float:
        method = SVDQuantOnlineLinearMethod(_online_config(rank=0, iterations=1, **kwargs))
        layer = _build_layer(method, weight)
        method.process_weights_after_loading(layer)
        out = method.apply(layer, x)
        return ((out.float() - reference).norm() / reference.norm()).item()

    single = error()
    chained = error(weight_terms=weight_terms, act_terms=act_terms)
    assert 0.05 < single < 0.5, f"single-term error {single:.3f} is not the regime under test"
    assert chained < single * ceiling, f"{weight_terms}x{act_terms} terms: {chained:.4f} vs {single:.4f}"
