<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cosmos3-Super GPU0 Video BKC

Best-known configuration for generating one real `Cosmos3-Super` video on
`GPU 0` with Omni, using the default T2V request settings from the recipe.

## Summary

- Model source: `/storage/yiliu7/nvidia/Cosmos3-Super`
- Runtime: `/home/yiliu7/workspace/omni-wm/.venv/bin/python`
- GPU: `0`
- Serving mode: Omni `/v1/videos/sync`
- Request config:
  - `size=1280x720`
  - `num_frames=189`
  - `fps=24`
  - `num_inference_steps=35`
  - `guidance_scale=6.0`
  - `flow_shift=10.0`
  - `max_sequence_length=4096`
  - `seed=17`
- Prompt assets:
  - `/storage/yiliu7/nvidia/Cosmos3-Super/assets/example_t2v_prompt.json`
  - `/storage/yiliu7/nvidia/Cosmos3-Super/assets/negative_prompt.json`

Generated artifact from the validated run:

- `/home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_t2v_default_gpu0_20260709.mp4`

Measured wall time for the full default request on one GPU with offload:

- about `372s` end-to-end

## Important Caveat

The local `Cosmos3-Super` checkpoint is incomplete for direct Omni loading:

- missing file:
  - `/storage/yiliu7/nvidia/Cosmos3-Super/vae/diffusion_pytorch_model.safetensors`

Because of that, direct serve from `/storage/yiliu7/nvidia/Cosmos3-Super`
fails during VAE load.

Working workaround used in the successful run:

- create a local mirror at:
  - `/home/yiliu7/workspace/omni-wm/.tmp/Cosmos3-Super-fixed-vae`
- keep everything from `Cosmos3-Super`
- inject the Nano VAE weights from:
  - `/storage/yiliu7/nvidia/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors`

The VAE config files are identical between the local Nano and Super
checkpoints, so this patched mirror is the current workable local path.

## BKC

Use this server command for a single-GPU `GPU 0` run:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/yiliu7/workspace/omni-wm/.venv/bin/python -m vllm.entrypoints.cli.main serve \
  /home/yiliu7/workspace/omni-wm/.tmp/Cosmos3-Super-fixed-vae \
  --omni \
  --host 127.0.0.1 \
  --port 8021 \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails \
  --enable-layerwise-offload \
  --init-timeout 2400 \
  --stage-init-timeout 2400
```

Notes:

- `--enable-layerwise-offload` is required here to make the single-GPU run
  practical.
- This is a `GPU 0` only workaround path, not the recipe's recommended
  multi-GPU serving topology.

## One-Video Run

### 1. Create the patched local mirror

```bash
rm -rf /home/yiliu7/workspace/omni-wm/.tmp/Cosmos3-Super-fixed-vae
mkdir -p /home/yiliu7/workspace/omni-wm/.tmp
cp -as /storage/yiliu7/nvidia/Cosmos3-Super \
  /home/yiliu7/workspace/omni-wm/.tmp/Cosmos3-Super-fixed-vae
ln -s /storage/yiliu7/nvidia/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors \
  /home/yiliu7/workspace/omni-wm/.tmp/Cosmos3-Super-fixed-vae/vae/diffusion_pytorch_model.safetensors
```

### 2. Start the Omni server

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/yiliu7/workspace/omni-wm/.venv/bin/python -m vllm.entrypoints.cli.main serve \
  /home/yiliu7/workspace/omni-wm/.tmp/Cosmos3-Super-fixed-vae \
  --omni \
  --host 127.0.0.1 \
  --port 8021 \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails \
  --enable-layerwise-offload \
  --init-timeout 2400 \
  --stage-init-timeout 2400
```

Wait until the server reports:

- `Pure diffusion API server initialized`
- `Starting vLLM API server`

### 3. Generate one default T2V video

```bash
PROMPT=$(jq -c . /storage/yiliu7/nvidia/Cosmos3-Super/assets/example_t2v_prompt.json)
NEGATIVE_PROMPT=$(jq -c . /storage/yiliu7/nvidia/Cosmos3-Super/assets/negative_prompt.json)

mkdir -p /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super

curl -sS -X POST http://127.0.0.1:8021/v1/videos/sync \
  -H 'Accept: video/mp4' \
  -F 'model=/home/yiliu7/workspace/omni-wm/.tmp/Cosmos3-Super-fixed-vae' \
  --form-string "prompt=$PROMPT" \
  --form-string "negative_prompt=$NEGATIVE_PROMPT" \
  -F 'size=1280x720' \
  -F 'num_frames=189' \
  -F 'fps=24' \
  -F 'num_inference_steps=35' \
  -F 'guidance_scale=6.0' \
  -F 'max_sequence_length=4096' \
  -F 'flow_shift=10.0' \
  -F 'seed=17' \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false}' \
  -o /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_t2v_default_gpu0_20260709.mp4
```

### 4. Verify output

```bash
ls -lh /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_t2v_default_gpu0_20260709.mp4
file /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_t2v_default_gpu0_20260709.mp4
```

## Cleanup

Stop the server with `Ctrl-C`.

To check that the port is released:

```bash
ss -ltnp | rg ':8021\b' || true
```

## If The Checkpoint Gets Fixed

If `Cosmos3-Super` later includes:

- `/storage/yiliu7/nvidia/Cosmos3-Super/vae/diffusion_pytorch_model.safetensors`

then use the original model path directly and skip the patched mirror.
