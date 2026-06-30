# Cosmos3-Nano W4A16 Quantization: Requirements & Guide

## Current State

We have two checkpoints:

| Path | Format | Language Model Layers | Generation Path | Loadable? |
|---|---|---|---|---|
| `/storage/yiliu7/nvidia/Cosmos3-Nano` | Diffusers pipeline | BF16 | BF16 | ✅ via diffusers `Cosmos3OmniPipeline` |
| `/storage/yiliu7/cosmos3-nano-w4a16-v2` | Native VLM | W4A16 (AutoRound) | ❌ MISSING | Partially (see below) |

### What works today

The native-format quantized checkpoint **can** be loaded and the quantized
understanding-path weights are correct:

```python
from transformers import AutoModel
from auto_round.inference.convert_model import convert_hf_model

model = AutoModel.from_pretrained(
    "/storage/yiliu7/cosmos3-nano-w4a16-v2",
    torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
)
model = model.cuda()
model.config.quantization_config.block_name_to_quantize = "language_model.layers"
model, backends = convert_hf_model(model, target_device="cuda")
# All 252 quantized layers replaced with MarlinQuantLinear (W4A16 int, group_size=128)
# post_init() fails on B200 due to Marlin kernel linker error (needs sm_100 fix)
```

The quantized weights (qweight/qzeros/scales) for the language model's `to_q/k/v`,
`to_out`, `gate/up/down_proj` are loaded correctly with `convert_hf_model`.

### What's broken

1. **Generation path weights missing**: The native VLM checkpoint has no
   `add_q_proj/k_proj/v_proj`, `to_add_out`, `mlp_moe_gen.*` weights. These
   must come from the BF16 diffusers checkpoint.

2. **Architecture mismatch**: The native format has `blocks.*` (vision encoder,
   hidden=1152) + `layers.*` (text LLM, hidden=4096). The diffusers format has
   only `layers.*` (joint dual-pathway MoE, hidden=4096). They're structurally
   incompatible — weight paths don't map 1:1.

3. **Backend compatibility**: The Marlin CUDA kernel (`gptqmodel:marlin_zp`)
   fails to compile on B200 (sm_100) — linker error for the Marlin template
   specialization with group_size=128, bits=4. The pure PyTorch backend
   (`auto_round:torch`) is not compatible with the W4A16 int quantization
   scheme in auto_round 0.13.1.

4. **Omni pipeline integration**: The omni-wm pipeline uses the diffusers
   `Cosmos3OmniDiffusersPipeline` which expects the diffusers directory structure
   (`model_index.json`, subdirectories per component). The quantized checkpoint
   is a flat directory with a single `config.json` — the pipeline can't
   initialize from it.

---

## Requirements for Proper Quantization

### 1. Quantize the diffusers-format transformer

Run AutoRound on `/storage/yiliu7/nvidia/Cosmos3-Nano/transformer/`, targeting
all quantizable linear layers (both understanding and generation paths).

**Target layers** (×36 layers):
```
layers.{0..35}.self_attn.to_q       (4096, 4096)  — und path Q
layers.{0..35}.self_attn.to_k       (1024, 4096)  — und path K
layers.{0..35}.self_attn.to_v       (1024, 4096)  — und path V
layers.{0..35}.self_attn.to_out     (4096, 4096)  — und path out
layers.{0..35}.self_attn.add_q_proj (4096, 4096)  — gen path Q
layers.{0..35}.self_attn.add_k_proj (1024, 4096)  — gen path K
layers.{0..35}.self_attn.add_v_proj (1024, 4096)  — gen path V
layers.{0..35}.self_attn.to_add_out (4096, 4096)  — gen path out
layers.{0..35}.mlp.gate_proj        (12288, 4096) — und MLP gate
layers.{0..35}.mlp.up_proj          (12288, 4096) — und MLP up
layers.{0..35}.mlp.down_proj        (4096, 12288) — und MLP down
layers.{0..35}.mlp_moe_gen.gate_proj (12288, 4096) — gen MLP gate
layers.{0..35}.mlp_moe_gen.up_proj   (12288, 4096) — gen MLP up
layers.{0..35}.mlp_moe_gen.down_proj (4096, 12288) — gen MLP down
```
Plus optionally: `time_embedder.linear_1`, `time_embedder.linear_2`

**Keep BF16** (exclude from quantization):
```
layers.{0..35}.self_attn.norm_q       (128,)     — per-head Q norm
layers.{0..35}.self_attn.norm_k       (128,)     — per-head K norm
layers.{0..35}.self_attn.norm_added_q (128,)     — gen Q norm
layers.{0..35}.self_attn.norm_added_k (128,)     — gen K norm
layers.{0..35}.input_layernorm        (4096,)    — und input norm
layers.{0..35}.post_attention_layernorm   (4096,) — und post-attn norm
layers.{0..35}.input_layernorm_moe_gen    (4096,) — gen input norm
layers.{0..35}.post_attention_layernorm_moe_gen (4096,) — gen post-attn norm
embed_tokens.weight                   (151936, 4096) — tied with lm_head
lm_head.weight                        (151936, 4096) — tied with embed_tokens
proj_in.*                             — small projections
proj_out.*                            — small projections
time_embedder.linear_1/2              — optional, keep BF16 for precision
```

### 2. Output format

AutoRound config (`quantization_config.json`):
```json
{
  "bits": 4,
  "group_size": 128,
  "sym": true,
  "data_type": "int",
  "quant_method": "auto-round",
  "block_name_to_quantize": "layers",
  "extra_config": {
    "embed_tokens": {"bits": 16, "data_type": "fp"},
    "lm_head": {"bits": 16, "data_type": "fp"},
    "norm": {"bits": 16, "data_type": "fp"},
    "norm_moe_gen": {"bits": 16, "data_type": "fp"},
    "proj_in": {"bits": 16, "data_type": "fp"},
    "proj_out": {"bits": 16, "data_type": "fp"},
    "time_embedder": {"bits": 16, "data_type": "fp"},
    "self_attn.norm_q": {"bits": 16, "data_type": "fp"},
    "self_attn.norm_k": {"bits": 16, "data_type": "fp"},
    "self_attn.norm_added_q": {"bits": 16, "data_type": "fp"},
    "self_attn.norm_added_k": {"bits": 16, "data_type": "fp"},
    "input_layernorm": {"bits": 16, "data_type": "fp"},
    "post_attention_layernorm": {"bits": 16, "data_type": "fp"},
    "input_layernorm_moe_gen": {"bits": 16, "data_type": "fp"},
    "post_attention_layernorm_moe_gen": {"bits": 16, "data_type": "fp"}
  }
}
```

The result should be a quantized transformer that fits within the existing
diffusers pipeline — just swap `Cosmos3OmniTransformer` → quantized version,
keeping the same VAE, scheduler, and tokenizer.

### 3. Backend constraints (B200 / sm_100)

| Backend | Status on B200 |
|---|---|
| `gptqmodel:marlin` | Fails — Marlin kernel linker error on sm_100 |
| `gptqmodel:exllamav2` | Unknown — may or may not compile |
| `auto_round:tritonv2` | Not compatible with W4A16 int config |
| `auto_round:torch` | Not compatible with W4A16 int config |
| `auto_round_kernel` | Unknown |

**Requirement**: A working CUDA/Triton backend on B200 that supports W4A16
int symmetric quantization with group_size=128.

---

## Size Projection

| Scenario | Transformer | VAE | Other | Total |
|---|---|---|---|---|
| BF16 (current) | 30.3GB | 1.4GB | 1.0GB | **32.7GB** |
| W4A16 all layers | ~3.5GB | 1.4GB | 1.0GB | **~5.9GB** |
| W4A16 + unload sound_tokenizer | ~3.5GB | 1.4GB | 0 | **~4.9GB** |

---

## References

- [`cosmos3_quant_guide.md`](cosmos3_quant_guide.md) — Detailed per-component breakdown
- [`cosmos3_arch_mismatch.md`](cosmos3_arch_mismatch.md) — Why the existing native-format checkpoint doesn't work with the diffusers pipeline
- BF16 model: `/storage/yiliu7/nvidia/Cosmos3-Nano`
- Quantized (incomplete): `/storage/yiliu7/cosmos3-nano-w4a16-v2`
