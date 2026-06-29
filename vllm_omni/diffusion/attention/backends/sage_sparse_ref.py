"""
Block-wise PyTorch reference implementation of SageAttention sparse kernel.

Processes one Q-block at a time using LUT-selected K blocks, avoiding
the O(n^2) memory of a full attention score matrix. This is a faithful
reimplementation of the sage_sparse math but in pure PyTorch, to isolate
whether quality issues come from the preprocess/routing or the SYCL kernel.
"""

from __future__ import annotations

import torch


def _decode_block_map(lut: torch.Tensor, valid_block_num: torch.Tensor) -> torch.Tensor:
    """
    Decode delta-encoded LUT into a boolean block mask.

    The LUT is stored as [B, H, Q, K] int32.  The first valid entry is the
    first selected block index (offset from 0).  Subsequent entries are
    deltas from the previous selected index.  Only the first
    valid_block_num entries per row are meaningful.

    Returns:
        block_mask: [B, H, Q, K] bool
    """
    B, H, Q, K = lut.shape
    device = lut.device
    cum = torch.cumsum(lut, dim=-1)
    arange = torch.arange(K, device=device).view(1, 1, 1, K)
    valid_mask = arange < valid_block_num.unsqueeze(-1)
    selected = cum.masked_fill(~valid_mask, -1)

    flat_selected = selected.view(-1)
    flat_valid = valid_mask.view(-1)
    keep = flat_valid & (flat_selected >= 0) & (flat_selected < K)

    block_mask = torch.zeros(B * H * Q * K, dtype=torch.bool, device=device)
    if keep.any():
        flat_pos = torch.arange(B * H * Q * K, device=device)[keep]
        target_idx = flat_selected[keep].long().clamp_(0, K - 1)
        row_base = (flat_pos // K) * K
        block_mask[row_base + target_idx] = True
    return block_mask.view(B, H, Q, K)


def _validate(tensor: torch.Tensor, name: str, layout: str) -> tuple[int, int, int, int]:
    assert tensor.dim() == 4, f"{name} must be 4D"
    if layout.upper() == "HND":
        return tensor.shape  # B, H, S, D
    B, S, H, D = tensor.shape
    return B, H, S, D


def _to_hnd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout.upper() == "HND":
        return tensor.contiguous()
    return tensor.transpose(1, 2).contiguous()


def _from_hnd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout.upper() == "HND":
        return tensor.contiguous()
    return tensor.transpose(1, 2).contiguous()


def sage_sparse_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    quant_block_size: int = 64,
    qscale: torch.Tensor | None = None,
    kscale: torch.Tensor | None = None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """
    Block-wise sparse int8 attention (memory-safe).

    Processes one Q-block at a time using only the K blocks selected by
    the routing metadata.  Never materialises a full [Sq, Skv] score matrix.
    """
    # Validate
    B, Hq, Sq, D = _validate(query, "query", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate(key, "key", tensor_layout)
    Bv, Hkv2, Skv2, Dv = _validate(value, "value", tensor_layout)
    assert Bk == B and Bv == B
    assert Hkv2 == Hkv and Skv2 == Skv and Dv == Dk
    assert Dk == D and D in (64, 128)
    assert quant_block_size == 64

    scale_val = float(scale) if scale is not None else 1.0 / (D ** 0.5)
    q_blocks = (Sq + quant_block_size - 1) // quant_block_size
    kv_blocks = (Skv + quant_block_size - 1) // quant_block_size
    assert lut.shape == (B, Hq, q_blocks, kv_blocks)
    assert valid_block_num.shape == (B, Hq, q_blocks)

    # Convert to HND
    q_hnd = _to_hnd(query, tensor_layout)
    k_hnd = _to_hnd(key, tensor_layout)
    v_hnd = _to_hnd(value, tensor_layout)

    # Decode block map (which K-blocks are selected for each Q-block)
    block_mask = _decode_block_map(lut, valid_block_num)  # [B, Hq, q_blocks, kv_blocks]

    # Convert qscale/kscale from [B, H, Bk, 1] to proper shapes
    if qscale.dim() == 4 and qscale.shape[-1] == 1:
        qscale_1d = qscale.view(B, Hq, q_blocks)
    else:
        qscale_1d = qscale.view(B, Hq, q_blocks)

    if kscale.dim() == 4 and kscale.shape[-1] == 1:
        kscale_1d = kscale.view(B, Hkv, kv_blocks)
    else:
        kscale_1d = kscale.view(B, Hkv, kv_blocks)

    # Output buffer
    out_hnd = torch.zeros(B, Hq, Sq, D, dtype=value.dtype, device=query.device)

    for b in range(B):
        for hq in range(Hq):
            hkv = hq // (Hq // Hkv) if Hkv < Hq else hq

            for qb in range(q_blocks):
                q_start = qb * quant_block_size
                q_end = min(q_start + quant_block_size, Sq)
                q_len = q_end - q_start

                # Load Q block, dequantize
                q_block_i8 = q_hnd[b, hq, q_start:q_end, :]  # [q_len, D]
                q_scale = qscale_1d[b, hq, qb].item()
                q_fp32 = q_block_i8.to(torch.float32) * q_scale

                # Get selected K blocks for this Q block
                selected_kb = torch.where(block_mask[b, hq, qb])[0]  # indices
                num_sel = selected_kb.numel()
                if num_sel == 0:
                    continue

                # Gather selected K blocks, dequantize
                k_blocks_list = []
                for kb in selected_kb.tolist():
                    ks = kb * quant_block_size
                    ke = min(ks + quant_block_size, Skv)
                    k_block_i8 = k_hnd[b, hkv, ks:ke, :]
                    k_s = kscale_1d[b, hkv, kb].item()
                    k_blocks_list.append(k_block_i8.to(torch.float32) * k_s)
                k_selected = torch.cat(k_blocks_list, dim=0)  # [num_sel*64 or less, D]

                # Attention scores for this Q-block against selected K blocks
                scores = torch.matmul(q_fp32, k_selected.transpose(-1, -2))  # [q_len, total_k_len_selected]
                scores *= scale_val

                # Causal mask within selected blocks (approximate: block-level)
                if is_causal:
                    # Build causal mask: only allow each Q row to attend to K rows up to its position
                    q_pos = torch.arange(q_start, q_end, device=query.device)
                    k_pos = []
                    for kb in selected_kb.tolist():
                        ks = kb * quant_block_size
                        ke = min(ks + quant_block_size, Skv)
                        k_pos.extend(range(ks, ke))
                    k_pos = torch.tensor(k_pos, device=query.device)
                    causal_mask = q_pos.unsqueeze(-1) >= k_pos.unsqueeze(0)  # [q_len, total_k_len]
                    scores = scores.masked_fill(~causal_mask, float("-inf"))

                # Softmax
                attn = torch.softmax(scores, dim=-1).to(value.dtype)  # [q_len, total_k_len]

                # Gather corresponding V blocks
                v_blocks_list = []
                for kb in selected_kb.tolist():
                    vs = kb * quant_block_size
                    ve = min(vs + quant_block_size, Skv)
                    v_blocks_list.append(v_hnd[b, hkv, vs:ve, :])
                v_selected = torch.cat(v_blocks_list, dim=0)  # [total_k_len, D]

                # Weighted sum
                out_block = torch.matmul(attn, v_selected)  # [q_len, D]
                out_hnd[b, hq, q_start:q_end, :] = out_block.to(value.dtype)

    # Convert back
    return _from_hnd(out_hnd, tensor_layout)
