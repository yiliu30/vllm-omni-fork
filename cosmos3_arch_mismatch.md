# Cosmos3-Nano: Architecture Mismatch Between Native & Diffusers Checkpoints

## The Problem

The AutoRound W4A16 quantized models at `/storage/yiliu7/cosmos3-nano-w4a16` and
`/storage/yiliu7/cosmos3-nano-w4a16-v2` cannot be loaded into the omni-wm (diffusers)
pipeline. The root cause is not quantization format — it's an **architectural mismatch**
between the native NVIDIA Cosmos3 checkpoint and the diffusers-format checkpoint.

## Two Different Checkpoint Formats

### 1. Native NVIDIA Format (`/storage/yiliu7/nvidia/Cosmos3-Nano` — BF16 reference)

Uses **`Cosmos3OmniModel`** (transformers library, `model_type="cosmos3_omni"`):

```
Cosmos3OmniModel
├── visual (Qwen3VLVisionModel, 27 blocks, hidden=1152)
│   ├── blocks.0..26
│   │   ├── attn.qkv.weight      (3456, 1152) — unified QKV
│   │   ├── attn.proj.weight     (1152, 1152)
│   │   └── mlp.linear_fc1/fc2
│   └── patch_embed, pos_embed
├── language_model (Qwen2 LLM, 36 layers, hidden=4096)
│   ├── layers.0..35
│   │   ├── self_attn.to_q/k/v   (separate Q/K/V)
│   │   ├── self_attn.to_out
│   │   ├── mlp.gate/up/down_proj
│   │   └── input_layernorm, post_attention_layernorm
│   └── embed_tokens, norm
├── lm_head
├── time_embedder                 — diffusion adapter
├── proj_in / proj_out            — diffusion latent projection
├── action_proj_in/out            — modality adapters
└── audio_proj_in/out
```

**This is a VLM (Vision-Language Model) with lightweight diffusion adapters.**
The diffusion generation pathway (noise prediction, cross-attention between text
and image latents) is minimal — the heavy lifting is done by the language model
with adapter projections.

### 2. Diffusers Format (`/storage/yiliu7/nvidia/Cosmos3-Nano` — diffusers view)

Uses **`Cosmos3OmniTransformer`** (diffusers library):

```
Cosmos3OmniTransformer (36 layers, hidden=4096)
├── layers.0..35
│   ├── self_attn
│   │   ├── to_q.weight (4096,4096)     ← text understanding path (und)
│   │   ├── to_k.weight (1024,4096)
│   │   ├── to_v.weight (1024,4096)
│   │   ├── to_out.weight (4096,4096)
│   │   ├── norm_q / norm_k
│   │   ├── add_q_proj.weight (4096,4096) ← generation path (gen)
│   │   ├── add_k_proj.weight (1024,4096)
│   │   ├── add_v_proj.weight (1024,4096)
│   │   ├── to_add_out.weight (4096,4096)
│   │   └── norm_added_q / norm_added_k
│   ├── mlp (text)
│   │   └── gate_proj / up_proj / down_proj   (12288 intermediate)
│   ├── mlp_moe_gen (generation)
│   │   └── gate_proj / up_proj / down_proj   (12288 intermediate)
│   ├── input_layernorm
│   ├── input_layernorm_moe_gen
│   ├── post_attention_layernorm
│   └── post_attention_layernorm_moe_gen
├── patch_embed, pos_embed
├── time_embedder
├── embed_tokens
└── proj_in, proj_out
```

**This is a full diffusion transformer with dual-pathway MoE attention.**
Each layer handles BOTH understanding (text causal attention) and generation
(image non-causal attention) via separate projection matrices, with MoE-routed
MLP experts.

## Key Mismatches

| Component | Native Format | Diffusers Format | Compatible? |
|---|---|---|---|
| Vision encoder | `blocks.0-26` (separate, hidden=1152) | NOT present — merged into layers | ❌ |
| Text attention | `to_q/k/v` (quantized) | `to_q/k/v` | ✅ (same dims) |
| Gen attention | ❌ not present | `add_q/k_proj`, `to_add_out` | ❌ MISSING |
| Text MLP | `gate/up/down_proj` (quantized) | `gate/up/down_proj` | ✅ (same dims) |
| Gen MLP | ❌ not present | `mlp_moe_gen.*` | ❌ MISSING |
| Gen norms | ❌ not present | `*_moe_gen` norms | ❌ MISSING |
| Time embedder | ✅ present | ✅ present | ✅ |
| Proj in/out | ✅ present | ✅ present | ✅ |

## AutoRound Quantization Status

The native-format checkpoint at `...w4a16-v2` was successfully quantized with
AutoRound (W4A16, group_size=128, sym=True) and can be loaded via:

```python
from transformers import AutoModel
from auto_round.inference.convert_model import convert_hf_model, post_init

model = AutoModel.from_pretrained(QUANT_PATH, torch_dtype=torch.bfloat16,
                                  device_map='cuda', trust_remote_code=True)
model = model.cuda()
model.config.quantization_config.block_name_to_quantize = "language_model.layers"
model, backends = convert_hf_model(model, target_device='cuda')
# post_init requires a working CUDA kernel backend
```

The Marlin kernel compilation fails on B200 (sm_100 linker error). A working
backend (e.g., pure PyTorch `auto_round:torch`) is not compatible with the
W4A16 int quantization scheme in the installed auto_round version (0.13.1).

## Solution

To get a quantized Cosmos3-Nano that works with the omni-wm pipeline, apply
AutoRound quantization directly to the **diffusers-format transformer**
(`/storage/yiliu7/nvidia/Cosmos3-Nano/transformer/`) rather than to the
native-format checkpoint.
