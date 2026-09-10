# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated quantization debug dump for diffusion (XPU).

Enabled only when ``VLLM_OMNI_QUANT_DUMP_DIR`` is set; zero overhead otherwise.
This is a debug feature, not a production path.

Optional env:
  VLLM_OMNI_QUANT_DUMP_TAG    output tag (default: dump dir basename)
  VLLM_OMNI_QUANT_DUMP_TOKENS deterministic token subsample size for all
                              reference computes / stats (default 2048, 0 = full)

The pass index counts transformer forwards (each denoising step runs the
active expert once per CFG branch); it includes the engine warmup pass.
With cache-dit, cached steps compute only the first block, so non-block-0
entries only have rows on computing steps.

What is recorded (per forward pass through the DiT blocks, in-run on
identical inputs):

  linear   per (expert, block, op): diff between the quantized GEMM output
           and an in-run BF16 reference GEMM computed from the original
           weights (kept as ``layer.weight_bf16`` by the online MXFP
           methods in dump mode): {cos, rel_l2, max_abs, mean_abs}, plus
           input stats {rms, absmax} and per-input-channel absmean.
  attn     per (expert, block, kind="self"): diff between the Sage V3
           Hybrid output and an in-run BF16 SDPA on the same q/k/v, per
           attention head: {cos, rel_l2, max_abs}, plus per-head q/k rms.
           kind in {"cross", "forced", "fallback", "sdpa"}: stats only
           (per-head q/k rms, output rms).

On process exit the recorder writes ``<dir>/<tag>.npz`` and
``<dir>/<tag>_summary.json``.  The pass index counts block-0 self-attention
visits of the active expert; it includes the engine warmup/dummy pass.
"""

from __future__ import annotations

import atexit
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from torch.compiler import disable as _torch_compiler_disable

logger = init_logger(__name__)

_DUMP_DIR_ENV = "VLLM_OMNI_QUANT_DUMP_DIR"
_DUMP_TAG_ENV = "VLLM_OMNI_QUANT_DUMP_TAG"
_DUMP_TOKENS_ENV = "VLLM_OMNI_QUANT_DUMP_TOKENS"
_DEFAULT_TOKENS = 2048
_EPS = 1e-6

_BLOCK_RE = re.compile(r"^(transformer(?:_2)?\.blocks\.(\d+)\.)")
_TRIGGER_OP = "attn1.to_qkv"


def is_enabled() -> bool:
    """True when VLLM_OMNI_QUANT_DUMP_DIR is set at import time."""
    return bool(os.environ.get(_DUMP_DIR_ENV, "").strip())


def _tag() -> str:
    return (
        os.environ.get(_DUMP_TAG_ENV, "").strip()
        or os.path.basename(os.environ.get(_DUMP_DIR_ENV, "").rstrip("/"))
        or "dump"
    )


def _tokens() -> int:
    try:
        return int(os.environ.get(_DUMP_TOKENS_ENV, "").strip() or _DEFAULT_TOKENS)
    except ValueError:
        return _DEFAULT_TOKENS


@dataclass
class _Recorder:
    pass_count: int = 0
    has_seen_linear: bool = False
    # (expert, block, op) -> list of per-pass records
    lin: dict[tuple[str, int, str], list] = field(default_factory=dict)
    # (expert, block, kind) -> list of per-pass records
    attn: dict[tuple[str, int, str], list] = field(default_factory=dict)
    idx_cache: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)

    def note_pass(self, expert: str, block: int, is_self: bool) -> int:
        """Return the current pass index, advancing on block-0 self-attention.

        In quantized runs the pass is advanced by the block-0 trigger linear
        (see :meth:`note_linear_pass`), which fires before the attention; in
        plain BF16 runs no linear is recorded, so the block-0 self-attention
        is the per-forward trigger instead. Either way each transformer
        forward gets exactly one pass index, including cache-dit steps that
        compute only block 0.
        """
        if not self.has_seen_linear and block == 0 and is_self:
            self.pass_count += 1
        return max(self.pass_count - 1, 0)

    def note_linear_pass(self, block: int, is_trigger: bool) -> int:
        """Pass index from a recorded linear; advances on the block-0 trigger op."""
        self.has_seen_linear = True
        if is_trigger and block == 0:
            self.pass_count += 1
        return max(self.pass_count - 1, 0)

    def subsample(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, str(device))
        idx = self.idx_cache.get(key)
        if idx is None:
            tokens = _tokens()
            if tokens <= 0 or n <= tokens:
                idx = torch.arange(n, device=device)
            else:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(0x5EED + n)
                idx = torch.randperm(n, generator=gen)[:tokens].sort().values.to(device)
            self.idx_cache[key] = idx
        return idx

    def _append(self, table: dict, key: tuple, p: int, value) -> None:
        rows = table.setdefault(key, [])
        while len(rows) <= p:
            rows.append(None)
        rows[p] = value


_RECORDER = _Recorder()


def _parse_block(prefix: str) -> tuple[str, int, str] | None:
    """'...blocks.3.attn1.to_qkv' -> (expert, 3, 'attn1.to_qkv').

    The expert is everything before 'blocks.N'; prefixes without an expert
    root collapse to the empty string.
    """
    if not prefix:
        return None
    match = re.match(r"^(?P<expert>.*?blocks\.(\d+))\.(?P<op>.+)$", prefix)
    if not match:
        return None
    expert = match.group("expert")[: -len(f"blocks.{match.group(2)}")]
    return expert, int(match.group(2)), match.group("op")


def _diff_metrics_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """{cos, rel_l2, max_abs, mean_abs} for two flat tensors."""
    dot = (a * b).sum()
    na = a.norm()
    nb = b.norm()
    diff = a - b
    return torch.stack([dot / (na * nb + _EPS), diff.norm() / (nb + _EPS), diff.abs().max(), diff.abs().mean()])


def _per_head_diff(a_h: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    """Per-head {cos, rel_l2, max_abs} for [B, H, S, D] tensors on one batch."""
    a = a_h[0].float().reshape(a_h.shape[1], -1)
    b = b_h[0].float().reshape(b_h.shape[1], -1)
    dot = (a * b).sum(dim=1)
    na = a.norm(dim=1)
    nb = b.norm(dim=1)
    cos = dot / (na * nb + _EPS)
    rel_l2 = (a - b).norm(dim=1) / (nb + _EPS)
    max_abs = (a - b).abs().max(dim=1).values
    return torch.stack([cos, rel_l2, max_abs], dim=1)


def _head_rms(t_hnd: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
    """Per-head RMS of [B, H, S, D] (optionally subsampled along S)."""
    t = t_hnd[0]
    if idx is not None:
        t = t[:, idx]
    return t.float().square().mean(dim=(1, 2)).sqrt()


@_torch_compiler_disable
def record_linear(prefix: str, x2d: torch.Tensor, y2d: torch.Tensor, layer) -> None:
    """Record quant-vs-BF16 GEMM diff and input stats for one linear call."""
    if not is_enabled():
        return
    try:
        parsed = _parse_block(prefix)
        if parsed is None:
            return
        weight_bf16 = getattr(layer, "weight_bf16", None)
        if weight_bf16 is None:
            return
        expert, block, op = parsed
        p = _RECORDER.note_linear_pass(block, op == _TRIGGER_OP)
        idx = _RECORDER.subsample(x2d.shape[0], x2d.device)
        xs = x2d[idx]
        if weight_bf16.device != x2d.device:
            # Shadow lives in host RAM (device-resident copies would OOM the
            # 122GB XPU on long-sequence runs); stream it for this GEMM only.
            weight_bf16 = weight_bf16.to(x2d.device)
        ref = xs @ weight_bf16.t()
        yq = y2d[idx]
        metrics = _diff_metrics_1d(yq.reshape(-1).float(), ref.reshape(-1).float()).cpu().numpy()
        xs_f = xs.float()
        in_stats = torch.stack([xs_f.square().mean().sqrt(), xs_f.abs().max()]).cpu().numpy()
        in_ch = xs_f.abs().mean(dim=0).cpu().numpy()
        _RECORDER._append(_RECORDER.lin, (expert, block, op), p, (metrics, in_stats, in_ch))
    except Exception as error:  # debug feature must never break the run
        logger.warning_once("quant_dump record_linear failed: %s", error)


@_torch_compiler_disable
def record_attn(
    prefix: str,
    q_hnd: torch.Tensor,
    k_hnd: torch.Tensor,
    v_hnd: torch.Tensor,
    out_hnd: torch.Tensor | None,
    kind: str,
    softmax_scale: float,
) -> None:
    """Record Sage-vs-SDPA per-head diff (kind='self') or stats only."""
    if not is_enabled():
        return
    try:
        parsed = _parse_block(prefix)
        if parsed is None:
            return
        expert, block, _op = parsed
        is_self = kind == "self" and q_hnd.shape[2] == k_hnd.shape[2]
        p = _RECORDER.note_pass(expert, block, is_self)
        if kind == "self" and out_hnd is not None and is_self:
            idx = _RECORDER.subsample(q_hnd.shape[2], q_hnd.device)
            ref = F.scaled_dot_product_attention(q_hnd[:, :, idx].contiguous(), k_hnd, v_hnd, scale=softmax_scale)
            metrics = _per_head_diff(out_hnd[:, :, idx], ref).cpu().numpy()
            q_rms = _head_rms(q_hnd, idx).cpu().numpy()
            k_rms = _head_rms(k_hnd, None).cpu().numpy()
            _RECORDER._append(_RECORDER.attn, (expert, block, "self"), p, (metrics, q_rms, k_rms))
        else:
            idx = _RECORDER.subsample(q_hnd.shape[2], q_hnd.device)
            q_rms = _head_rms(q_hnd, idx).cpu().numpy()
            k_rms = _head_rms(k_hnd, None).cpu().numpy()
            out_rms = out_hnd.float().square().mean().sqrt().cpu().numpy() if out_hnd is not None else np.float32(0.0)
            _RECORDER._append(_RECORDER.attn, (expert, block, kind), p, (None, q_rms, k_rms, out_rms))
    except Exception as error:  # debug feature must never break the run
        logger.warning_once("quant_dump record_attn failed: %s", error)


@_torch_compiler_disable
def record_sdpa_stats(
    prefix: str,
    q_hnd: torch.Tensor,
    k_hnd: torch.Tensor,
    _v_hnd: torch.Tensor,
    out_hnd: torch.Tensor,
) -> None:
    """Record per-head q/k rms and output rms for a plain SDPA call (bf16 run)."""
    if not is_enabled():
        return
    try:
        parsed = _parse_block(prefix)
        if parsed is None:
            return
        expert, block, _op = parsed
        is_self = q_hnd.shape[2] == k_hnd.shape[2]
        p = _RECORDER.note_pass(expert, block, is_self)
        idx = _RECORDER.subsample(q_hnd.shape[2], q_hnd.device)
        q_rms = _head_rms(q_hnd, idx).cpu().numpy()
        k_rms = _head_rms(k_hnd, None).cpu().numpy()
        out_rms = out_hnd.float().square().mean().sqrt().cpu().numpy()
        kind = "sdpa_self" if is_self else "sdpa_cross"
        _RECORDER._append(_RECORDER.attn, (expert, block, kind), p, (None, q_rms, k_rms, out_rms))
    except Exception as error:  # debug feature must never break the run
        logger.warning_once("quant_dump record_sdpa_stats failed: %s", error)


def _op_sort_key(op: str):
    order = {
        "attn1.to_qkv": 0,
        "attn1.to_out": 1,
        "attn2.to_q": 2,
        "attn2.to_k": 3,
        "attn2.to_v": 4,
        "attn2.to_out": 5,
    }
    return (order.get(op, 6), op)


def _pad_pass_array(rows: list, tail: tuple, component: int) -> np.ndarray:
    """Per-pass records -> [pass_count, *tail] with NaN holes for skipped passes.

    ``component`` selects which field of each stored record tuple to write.
    """
    out = np.full((len(rows),) + tuple(tail), np.nan, dtype=np.float32)
    for p, record in enumerate(rows):
        if record is not None:
            out[p] = record[component]
    return out


def save() -> None:
    if not is_enabled() or _RECORDER.pass_count == 0:
        return
    out_dir = os.environ[_DUMP_DIR_ENV]
    os.makedirs(out_dir, exist_ok=True)
    tag = _tag()
    npz_path = os.path.join(out_dir, f"{tag}.npz")
    summary_path = os.path.join(out_dir, f"{tag}_summary.json")

    payload: dict[str, np.ndarray] = {}
    for (expert, block, op), rows in sorted(
        _RECORDER.lin.items(), key=lambda kv: (kv[0][0], kv[0][1], _op_sort_key(kv[0][2]))
    ):
        while len(rows) < _RECORDER.pass_count:
            rows.append(None)
        complete = [r for r in rows if r is not None]
        if not complete:
            continue
        payload[f"lin_{expert}_{block}_{op}_diff"] = _pad_pass_array(rows, complete[0][0].shape, 0)
        payload[f"lin_{expert}_{block}_{op}_in"] = _pad_pass_array(rows, complete[0][1].shape, 1)
        payload[f"lin_{expert}_{block}_{op}_inch"] = _pad_pass_array(rows, complete[0][2].shape, 2)
    for (expert, block, kind), rows in sorted(_RECORDER.attn.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        while len(rows) < _RECORDER.pass_count:
            rows.append(None)
        complete = [r for r in rows if r is not None]
        if not complete:
            continue
        base = f"attn_{expert}_{block}_{kind}"
        if complete[0][0] is not None:
            payload[f"{base}_diff"] = _pad_pass_array(rows, complete[0][0].shape, 0)
        payload[f"{base}_qk"] = _pad_pass_array(rows, complete[0][1].shape, 1)
        payload[f"{base}_krms"] = _pad_pass_array(rows, complete[0][2].shape, 2)
        if len(complete[0]) > 3:
            payload[f"{base}_outrms"] = _pad_pass_array(rows, (), 3)

    np.savez_compressed(npz_path, **payload)
    experts = sorted({e for e, _b, _o in _RECORDER.lin} | {e for e, _b, _k in _RECORDER.attn})
    blocks = sorted({b for _e, b, _o in _RECORDER.lin} | {b for _e, b, _k in _RECORDER.attn})
    summary = {
        "tag": tag,
        "passes": _RECORDER.pass_count,
        "subsample_tokens": _tokens(),
        "experts": experts,
        "blocks": blocks,
        "lin_entries": len(_RECORDER.lin),
        "attn_entries": len(_RECORDER.attn),
        "env": {
            _DUMP_DIR_ENV: out_dir,
            _DUMP_TAG_ENV: os.environ.get(_DUMP_TAG_ENV, ""),
            _DUMP_TOKENS_ENV: os.environ.get(_DUMP_TOKENS_ENV, ""),
        },
        "npz": npz_path,
        "note": (
            "pass index = transformer forward (includes the engine warmup "
            "pass); with cache-dit, cached steps compute only block 0, so "
            "non-block-0 entries only have rows on computing steps; "
            "lin_*_diff = {cos, rel_l2, max_abs, mean_abs} quant vs bf16 on "
            "subsampled tokens; attn_*_diff = per-head {cos, rel_l2, max_abs} "
            "sage v3 hybrid vs bf16 sdpa on subsampled q rows"
        ),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Quant dump: wrote {npz_path} and {summary_path}", flush=True)


if is_enabled():
    atexit.register(save)
