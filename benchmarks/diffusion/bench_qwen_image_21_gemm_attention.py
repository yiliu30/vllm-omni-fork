# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark Qwen-Image-2.1 DiT GEMM and attention shapes on Intel XPU.

The shapes mirror the Qwen-Image-2.1 transformer at 1440x1440 and 2048x2048,
with a 31-token text prefix. Prefill includes the prefix; denoising attends the
target-image tokens against the cached prefix plus target K/V.

Run from the repository root with the SageAttention V3 hybrid build first on
``PYTHONPATH`` when benchmarking that kernel, for example:

    export ZE_AFFINITY_MASK=2
    export PYTHONPATH=$PWD:/home/yiliu7/deepklox-sav3-hybrid-hmt-20261026:/opt/vllm-v032
    python benchmarks/diffusion/bench_qwen_image_21_gemm_attention.py --hmt

``--hmt`` enables the Hadamard Q/K rotation in the DeepKLOX hybrid kernel.
Use ``--suite gemm`` or ``--suite attn`` to run only one half of the sweep.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch

HIDDEN_SIZE = 4096
MLP_HIDDEN_SIZE = 12288
NUM_HEADS = 32
HEAD_DIM = 128


@dataclass(frozen=True)
class ResolutionCase:
    name: str
    resolution: int
    target_tokens: int
    prefill_tokens: int


@dataclass(frozen=True)
class GemmSpec:
    name: str
    k: int
    n: int
    modules_per_block: int


@dataclass(frozen=True)
class Timing:
    median_ms: float
    mean_ms: float
    min_ms: float


_RESOLUTIONS = (
    ResolutionCase("1440x1440", 1440, 8100, 8131),
    ResolutionCase("2048x2048", 2048, 16384, 16415),
)

# QKV, MLP gate and MLP up share one shape. Attention output and MLP down use
# the other two shapes. modules_per_block is used for the aggregate table.
_GEMM_SPECS = (
    GemmSpec("QKV / MLP gate / MLP up", HIDDEN_SIZE, MLP_HIDDEN_SIZE, 3),
    GemmSpec("Attention output", HIDDEN_SIZE, HIDDEN_SIZE, 1),
    GemmSpec("MLP down", MLP_HIDDEN_SIZE, HIDDEN_SIZE, 1),
)


def _parse_resolutions(value: str) -> tuple[ResolutionCase, ...]:
    if value == "both":
        return _RESOLUTIONS
    wanted = {"1440": "1440x1440", "2048": "2048x2048"}[value]
    return tuple(case for case in _RESOLUTIONS if case.name == wanted)


def _measure(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> Timing:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return Timing(
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
    )


def _check_finite(fn: Callable[[], torch.Tensor], label: str) -> None:
    output = fn()
    torch.xpu.synchronize()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"{label} produced non-finite values")


def _tflops(flops: float, timing: Timing) -> float:
    return flops / (timing.median_ms * 1e-3) / 1e12


def _empty_xpu_cache() -> None:
    if torch.xpu.is_available():
        torch.xpu.empty_cache()


def _bf16_gemm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(hidden, weight.t())


def _mxfp8_gemm(
    hidden_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    hidden_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._xpu_C.fp8_gemm(
        hidden_fp8,
        weight_fp8.t(),
        torch.bfloat16,
        hidden_scale,
        weight_scale,
        None,
    )


def _mxfp8_linear(
    hidden: torch.Tensor,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    hidden_fp8, hidden_scale = quantize(hidden)
    return _mxfp8_gemm(hidden_fp8, weight_fp8, hidden_scale, weight_scale)


def _bf16_attention(
    impl: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return impl.forward_xpu(query, key, value, None)  # type: ignore[attr-defined]


def _sage_attention(
    attention: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kwargs: dict[str, object],
) -> torch.Tensor:
    return attention(query, key, value, **kwargs)


def _environment(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
    }
    try:
        import vllm

        info["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001 - environment reporting must not fail the benchmark
        pass
    try:
        import vllm_omni

        info["vllm_omni"] = vllm_omni.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        props = torch.xpu.get_device_properties(device)
        info["device_name"] = props.name
        info["device_memory_gib"] = round(props.total_memory / 1024**3, 2)
    except Exception:  # noqa: BLE001
        pass
    return info


def _benchmark_gemms(
    device: torch.device,
    cases: Sequence[ResolutionCase],
    warmup: int,
    iters: int,
    seed: int,
) -> list[dict[str, object]]:
    from vllm.model_executor.kernels.linear.mxfp8.xpu import quant_mxfp8
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        mxfp8_e4m3_quantize,
    )

    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        phases = (
            ("prefill", case.prefill_tokens),
            ("denoise", case.target_tokens),
        )
        for phase, m in phases:
            for spec_index, spec in enumerate(_GEMM_SPECS):
                torch.manual_seed(seed + case_index * 100 + spec_index)
                hidden = torch.randn((m, spec.k), device=device, dtype=torch.bfloat16)
                weight = torch.randn((spec.n, spec.k), device=device, dtype=torch.bfloat16)

                bf16_fn = partial(_bf16_gemm, hidden, weight)

                _check_finite(bf16_fn, f"BF16 {case.name} {phase} {spec.name}")
                bf16_timing = _measure(bf16_fn, warmup, iters)

                weight_fp8, weight_scale_raw = mxfp8_e4m3_quantize(weight)
                weight_scale = weight_scale_raw.view(torch.float8_e8m0fnu).t().contiguous()
                hidden_fp8, hidden_scale = quant_mxfp8(hidden)

                mxfp8_gemm = partial(_mxfp8_gemm, hidden_fp8, weight_fp8, hidden_scale, weight_scale)
                mxfp8_linear = partial(_mxfp8_linear, hidden, quant_mxfp8, weight_fp8, weight_scale)

                label = f"MXFP8 {case.name} {phase} {spec.name}"
                _check_finite(mxfp8_gemm, label + " GEMM")
                _check_finite(mxfp8_linear, label + " linear")
                mxfp8_gemm_timing = _measure(mxfp8_gemm, warmup, iters)
                mxfp8_linear_timing = _measure(mxfp8_linear, warmup, iters)

                flops = 2.0 * m * spec.n * spec.k
                rows.append(
                    {
                        "resolution": case.name,
                        "phase": phase,
                        "module": spec.name,
                        "modules_per_block": spec.modules_per_block,
                        "m": m,
                        "n": spec.n,
                        "k": spec.k,
                        "flops": flops,
                        "bf16_ms": bf16_timing.median_ms,
                        "mxfp8_gemm_ms": mxfp8_gemm_timing.median_ms,
                        "mxfp8_linear_ms": mxfp8_linear_timing.median_ms,
                        "bf16_tflops": _tflops(flops, bf16_timing),
                        "mxfp8_gemm_tflops": _tflops(flops, mxfp8_gemm_timing),
                        "mxfp8_linear_tflops": _tflops(flops, mxfp8_linear_timing),
                        "mxfp8_gemm_speedup": bf16_timing.median_ms / mxfp8_gemm_timing.median_ms,
                        "mxfp8_linear_speedup": bf16_timing.median_ms / mxfp8_linear_timing.median_ms,
                        "bf16_timing": asdict(bf16_timing),
                        "mxfp8_gemm_timing": asdict(mxfp8_gemm_timing),
                        "mxfp8_linear_timing": asdict(mxfp8_linear_timing),
                    }
                )
                del hidden, weight, weight_fp8, weight_scale_raw, weight_scale
                del hidden_fp8, hidden_scale
                _empty_xpu_cache()
    return rows


def _benchmark_attention(
    device: torch.device,
    cases: Sequence[ResolutionCase],
    warmup: int,
    iters: int,
    seed: int,
    hmt: bool,
) -> list[dict[str, object]]:
    from vllm_omni.diffusion.attention.backends.flash_attn import (
        FlashAttentionImpl,
    )

    try:
        from deepklox.sageattn_interface import sageattn_v3_hybrid
    except ImportError as exc:
        raise SystemExit(
            "DeepKLOX SageAttention V3 hybrid is unavailable. Put a DeepKLOX "
            "build with the hybrid kernel first on PYTHONPATH, for example "
            "/home/yiliu7/deepklox-sav3-hybrid-hmt-20261026."
        ) from exc

    sage_signature = inspect.signature(sageattn_v3_hybrid)
    if hmt and "hmt" not in sage_signature.parameters:
        raise SystemExit("This DeepKLOX build does not accept hmt=True")

    flash_impl = FlashAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        num_kv_heads=NUM_HEADS,
    )
    sage_label = "Sage V3 hybrid + HMT" if hmt else "Sage V3 hybrid"

    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        phases = (
            ("prefill", case.prefill_tokens, case.prefill_tokens),
            ("denoise", case.target_tokens, case.prefill_tokens),
        )
        for phase_index, (phase, seq_q, seq_kv) in enumerate(phases):
            torch.manual_seed(seed + case_index * 100 + phase_index)
            shape_q = (1, seq_q, NUM_HEADS, HEAD_DIM)
            shape_kv = (1, seq_kv, NUM_HEADS, HEAD_DIM)
            query = torch.randn(shape_q, device=device, dtype=torch.bfloat16)
            key = torch.randn(shape_kv, device=device, dtype=torch.bfloat16)
            value = torch.randn(shape_kv, device=device, dtype=torch.bfloat16)

            bf16_fn = partial(_bf16_attention, flash_impl, query, key, value)

            sage_kwargs = {"tensor_layout": "NHD"}
            if hmt:
                sage_kwargs["hmt"] = True

            sage_fn = partial(_sage_attention, sageattn_v3_hybrid, query, key, value, sage_kwargs)

            label = f"{case.name} {phase}"
            _check_finite(bf16_fn, f"BF16 attention {label}")
            _check_finite(sage_fn, f"{sage_label} {label}")
            bf16_timing = _measure(bf16_fn, warmup, iters)
            sage_timing = _measure(sage_fn, warmup, iters)

            flops = 4.0 * NUM_HEADS * seq_q * seq_kv * HEAD_DIM
            rows.append(
                {
                    "resolution": case.name,
                    "phase": phase,
                    "seq_q": seq_q,
                    "seq_kv": seq_kv,
                    "heads": NUM_HEADS,
                    "head_dim": HEAD_DIM,
                    "flops": flops,
                    "bf16_ms": bf16_timing.median_ms,
                    "sage_ms": sage_timing.median_ms,
                    "bf16_tflops": _tflops(flops, bf16_timing),
                    "sage_tflops": _tflops(flops, sage_timing),
                    "sage_speedup": bf16_timing.median_ms / sage_timing.median_ms,
                    "sage_kernel": sage_label,
                    "bf16_timing": asdict(bf16_timing),
                    "sage_timing": asdict(sage_timing),
                }
            )
            del query, key, value
            _empty_xpu_cache()
    return rows


def _print_gemm_table(rows: Sequence[dict[str, object]]) -> None:
    print("\n### GEMM shapes\n")
    print(
        "| Resolution | Phase | GEMM | M | N | K | BF16 ms | MXFP8 GEMM ms | "
        "MXFP8 + quant ms | BF16 TFLOPS | MXFP8 TFLOPS | "
        "MXFP8 + quant speedup |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['resolution']} | {row['phase']} | {row['module']} | "
            f"{row['m']} | {row['n']} | {row['k']} | "
            f"{row['bf16_ms']:.3f} | {row['mxfp8_gemm_ms']:.3f} | "
            f"{row['mxfp8_linear_ms']:.3f} | {row['bf16_tflops']:.1f} | "
            f"{row['mxfp8_gemm_tflops']:.1f} | "
            f"{row['mxfp8_linear_speedup']:.3f}x |"
        )


def _print_attention_table(rows: Sequence[dict[str, object]]) -> None:
    print("\n### Attention shapes\n")
    print("| Resolution | Phase | Sq | Skv | BF16 ms | Sage ms | BF16 TFLOPS | Sage TFLOPS | Sage speedup |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['resolution']} | {row['phase']} | {row['seq_q']} | "
            f"{row['seq_kv']} | {row['bf16_ms']:.3f} | {row['sage_ms']:.3f} | "
            f"{row['bf16_tflops']:.1f} | {row['sage_tflops']:.1f} | "
            f"{row['sage_speedup']:.3f}x |"
        )


def _print_aggregate(rows: Sequence[dict[str, object]]) -> None:
    print("\n### Per-block GEMM aggregate\n")
    print("| Resolution | Phase | BF16 ms | MXFP8 GEMM ms | MXFP8 + quant ms | MXFP8 speedup |")
    print("|---|---|---:|---:|---:|---:|")
    groups: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["resolution"]), str(row["phase"]))
        group = groups.setdefault(key, {"bf16": 0.0, "mxfp8": 0.0, "mxfp8_linear": 0.0})
        count = int(row["modules_per_block"])
        group["bf16"] += float(row["bf16_ms"]) * count
        group["mxfp8"] += float(row["mxfp8_gemm_ms"]) * count
        group["mxfp8_linear"] += float(row["mxfp8_linear_ms"]) * count
    for (resolution, phase), group in groups.items():
        print(
            f"| {resolution} | {phase} | {group['bf16']:.3f} | "
            f"{group['mxfp8']:.3f} | {group['mxfp8_linear']:.3f} | "
            f"{group['bf16'] / group['mxfp8_linear']:.3f}x |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--suite", choices=("all", "gemm", "attn"), default="all")
    parser.add_argument("--resolutions", choices=("both", "1440", "2048"), default="both")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hmt", action="store_true", help="Enable Sage V3 hybrid HMT")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be >= 0 and --iters must be >= 1")
    if not torch.xpu.is_available():
        raise SystemExit("XPU is not available")

    device = torch.device(args.device)
    cases = _parse_resolutions(args.resolutions)
    result: dict[str, object] = {"environment": _environment(device)}

    print("# Qwen-Image-2.1 shape benchmark")
    for key, value in result["environment"].items():
        print(f"- {key}: {value}")

    if args.suite in ("all", "gemm"):
        gemm_rows = _benchmark_gemms(device, cases, args.warmup, args.iters, args.seed)
        result["gemm"] = gemm_rows
        _print_gemm_table(gemm_rows)
        _print_aggregate(gemm_rows)

    if args.suite in ("all", "attn"):
        attention_rows = _benchmark_attention(device, cases, args.warmup, args.iters, args.seed, args.hmt)
        result["attention"] = attention_rows
        _print_attention_table(attention_rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
