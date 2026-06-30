# Cosmos3-Nano: Full Pipeline Breakdown & Quantization Guide

## Pipeline Overview (T2I Mode)

The Cosmos3OmniPipeline (diffusers) consists of 6 components as declared in
`model_index.json`. Only 4 are loaded with weights for T2I text-to-image
generation:

| Component | Type | Params | BF16 Size | Used in T2I? | Quantizable? |
|---|---|---|---|---|---|
| **transformer** | Cosmos3OmniTransformer (diffusers) | 15.17B | 30.3GB | Yes — denoising U-Net equiv | ✅ 91.6% of it |
| **vae** | AutoencoderKLWan (diffusers) | 0.70B | 1.4GB | Yes — latent → image decode | Minimal gain |
| **sound_tokenizer** | Cosmos3AVAEAudioTokenizer (diffusers) | 0.50B | 1.0GB | No — audio modalities only | Not worth it |
| **text_tokenizer** | Qwen2TokenizerFast (transformers) | 0 | 0 | Yes — text → token IDs | N/A (no weights) |
| **scheduler** | UniPCMultistepScheduler (diffusers) | 0 | 0 | Yes — noise schedule algorithm | N/A (no weights) |
| vision_encoder | Qwen3VLVisionModel (transformers) | 0.58B | 1.2GB | **Not loaded for T2I** | N/A |

**Pipeline total (T2I): ~16.37B params / ~32.7GB BF16**
The transformer alone is 93% of all weights.

---

## Transformer: Per-Layer Components (×36 layers)

### Understanding Path (text → causal attention)
These exist in both the diffusers model AND the quantized native checkpoint:

| Module | Weight Shape | Params | Quantizable? |
|---|---|---|---|
| `self_attn.to_q` | (4096, 4096) | 16.8M | ✅ W4A16 |
| `self_attn.to_k` | (1024, 4096) | 4.2M | ✅ W4A16 |
| `self_attn.to_v` | (1024, 4096) | 4.2M | ✅ W4A16 |
| `self_attn.to_out` | (4096, 4096) | 16.8M | ✅ W4A16 |
| `self_attn.norm_q` | (128,) | — | ❌ norm (keep BF16) |
| `self_attn.norm_k` | (128,) | — | ❌ norm (keep BF16) |
| `mlp.gate_proj` | (12288, 4096) | 50.3M | ✅ W4A16 |
| `mlp.up_proj` | (12288, 4096) | 50.3M | ✅ W4A16 |
| `mlp.down_proj` | (4096, 12288) | 50.3M | ✅ W4A16 |
| `input_layernorm` | (4096,) | — | ❌ norm |
| `post_attention_layernorm` | (4096,) | — | ❌ norm |
| **Subtotal (und path):** | | **192.9M/layer** | **6.7B across 36 layers** |

### Generation Path (latents → full attention)
These exist in the diffusers model but are MISSING from the quantized native checkpoint:

| Module | Weight Shape | Params | Quantizable? |
|---|---|---|---|
| `self_attn.add_q_proj` | (4096, 4096) | 16.8M | ✅ W4A16 |
| `self_attn.add_k_proj` | (1024, 4096) | 4.2M | ✅ W4A16 |
| `self_attn.add_v_proj` | (1024, 4096) | 4.2M | ✅ W4A16 |
| `self_attn.to_add_out` | (4096, 4096) | 16.8M | ✅ W4A16 |
| `self_attn.norm_added_q` | (128,) | — | ❌ norm |
| `self_attn.norm_added_k` | (128,) | — | ❌ norm |
| `mlp_moe_gen.gate_proj` | (12288, 4096) | 50.3M | ✅ W4A16 |
| `mlp_moe_gen.up_proj` | (12288, 4096) | 50.3M | ✅ W4A16 |
| `mlp_moe_gen.down_proj` | (4096, 12288) | 50.3M | ✅ W4A16 |
| `input_layernorm_moe_gen` | (4096,) | — | ❌ norm |
| `post_attention_layernorm_moe_gen` | (4096,) | — | ❌ norm |
| **Subtotal (gen path):** | | **192.9M/layer** | **6.7B across 36 layers** |

---

## Non-Layer Components

| Module | Shape | Params | Quantizable? |
|---|---|---|---|
| **embed_tokens** | (151936, 4096) | 622.3M | ✅ but usually kept BF16 |
| **lm_head** | (151936, 4096) | 622.3M | ✅ but tied with embed_tokens |
| **time_embedder.linear_1** | (4096, 256) | 1.0M | ✅ |
| **time_embedder.linear_2** | (4096, 4096) | 16.8M | ✅ |
| **proj_in** | (4096, 192) | 0.8M | ✅ |
| **proj_out** | (192, 4096) | 0.8M | ✅ |
| **action_proj_in.fc** | (32, 262144) | 8.4M | ✅ |
| **action_proj_out.fc** | (32, 262144) | 8.4M | ✅ |
| **audio_proj_in** | (4096, 64) | 0.3M | ✅ |
| **audio_proj_out** | (64, 4096) | 0.3M | ✅ |
| norm, norm_moe_gen, rotary_emb, biases, etc. | small | — | ❌ norms only |

---

## Weight Distribution

| Category | Params | % of Total | Quantizable |
|---|---|---|---|
| Understanding path (36 layers) | 6.94B | 45.8% | ✅ W4A16 → ~1.7B effective |
| Generation path (36 layers) | 6.94B | 45.8% | ✅ W4A16 → ~1.7B effective |
| embed_tokens + lm_head | 1.24B | 8.2% | keep BF16 (tied weights) |
| Other (projections, adapters) | ~55M | 0.4% | keep BF16 |
| Norms, embeds, biases | negligible | — | keep BF16 |
| **Total** | **15.17B** | **100%** | **W4A16 → ~5B effective** |

---

## Quantization Recommendation

### Primary target: Transformer

Quantize BOTH understanding and generation path linear layers to W4A16.
Covers ~91.6% of transformer params (~13.88B / 15.17B).

| What | Keep |
|---|---|
| All Linear layers (per-layer qkv, attn_out, gate/up/down) | W4A16 |
| Norm layers (RMSNorm, QK norms) | BF16 |
| embed_tokens + lm_head (tied) | BF16 |
| Small projections (proj_in/out, action_proj, audio_proj) | BF16 |
| time_embedder Linear layers | BF16 |

**Expected size after W4A16 transformer + BF16 remainder:**

| Component | Original | After Q |
|---|---|---|
| Transformer quantizable linear | 13.88B × 2B = 27.8GB | ~3.5GB (4-bit) |
| Transformer non-quantizable | 1.29B × 2B = 2.6GB | 2.6GB (BF16) |
| VAE | 0.70B × 2B = 1.4GB | 1.4GB (BF16, unchanged) |
| sound_tokenizer | 0.50B × 2B = 1.0GB | 1.0GB (BF16, unused) |
| **Total** | **~32.7GB** | **~8.5GB** |

### Not worth quantizing

- **VAE (0.70B):** Already only 1.4GB. Precision-sensitive autoencoder — quantization artifacts in the VAE would corrupt output images.
- **sound_tokenizer (0.50B):** Not used at all in T2I. Unnecessary to include at load time.

### How to proceed

The existing quantized checkpoints (`...w4a16[-v2]`) only quantized the
understanding path of a VLM-only checkpoint. They're missing the generation path
entirely. To get a fully working quantized model for the omni pipeline, AutoRound
needs to be run on the full diffusers-format `Cosmos3OmniTransformer` model at
`/storage/yiliu7/nvidia/Cosmos3-Nano/transformer/`.
