<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cosmos3-Super W4A16 GPU0 Video BKC

Best-known configuration for generating one real `Cosmos3-Super` T2V video
from the direct packed W4A16 checkpoint on `GPU 0` with Omni.

## Summary

- Model source:
  - `/storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed`
- Runtime:
  - `/home/yiliu7/workspace/omni-wm/.venv/bin/python`
- GPU:
  - `0`
- Serving mode:
  - Omni video API
- Validated request config:
  - `size=1280x720`
  - `num_frames=189`
  - `fps=24`
  - `num_inference_steps=35`
  - `guidance_scale=6.0`
  - `flow_shift=10.0`
  - `max_sequence_length=4096`
  - `seed=17`

Validated output artifact:

- `/home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_w4a16_packed_t2v_default_gpu0_20260709_async.mp4`

Observed final job metadata:

- `status=completed`
- `completed_at=2026-07-09 05:30:21 UTC`
- `inference_time_s=796.899`
- `stage_0_gen_ms=769578.475`
- `peak_memory_mb=38552.0`

## Important Notes

- `POST /v1/videos/sync` is not suitable for this default run here.
  - The server-side sync timeout is `600s`.
  - This model completed generation, but the sync request timed out before the
    bytes were returned.
- Use async `POST /v1/videos`, then poll `GET /v1/videos/{video_id}`, then
  download with `GET /v1/videos/{video_id}/content`.
- This direct-packed folder already has the layout needed by the original
  Cosmos3 loader:
  - `transformer/diffusion_pytorch_model-*.safetensors`
  - `transformer/diffusion_pytorch_model.safetensors.index.json`
  - `transformer/config.json` embedding `quantization_config` with `quant_method: "auto-round"`
  - `transformer/quantization_config.json` is still present, but no longer required for loading
- No local shim is required for this path.

## Serve

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/yiliu7/workspace/omni-wm/.venv/bin/python -m vllm.entrypoints.cli.main serve \
  /storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed \
  --omni \
  --host 127.0.0.1 \
  --port 8022 \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails \
  --enable-layerwise-offload \
  --init-timeout 2400 \
  --stage-init-timeout 2400
```

Wait for:

- `Pure diffusion API server initialized`
- `Starting vLLM API server`

## Generate One Video

### 1. Submit the async job

```bash
curl -sS -X POST http://127.0.0.1:8022/v1/videos \
  -F 'model=/storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed' \
  --form-string "prompt=$(jq -c . /storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed/assets/example_t2v_prompt.json)" \
  --form-string "negative_prompt=$(jq -c . /storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed/assets/negative_prompt.json)" \
  -F 'size=1280x720' \
  -F 'num_frames=189' \
  -F 'fps=24' \
  -F 'num_inference_steps=35' \
  -F 'guidance_scale=6.0' \
  -F 'max_sequence_length=4096' \
  -F 'flow_shift=10.0' \
  -F 'seed=17' \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false}'
```

The response returns a `video` object with an `id`, for example:

- `video_gen_5d43ab3ccae043a5a4155d4f0f8924f3`

### 2. Poll completion

```bash
curl -sS http://127.0.0.1:8022/v1/videos/video_gen_5d43ab3ccae043a5a4155d4f0f8924f3 | jq
```

Wait until:

- `"status": "completed"`

### 3. Download the MP4

```bash
curl -sS -L \
  http://127.0.0.1:8022/v1/videos/video_gen_5d43ab3ccae043a5a4155d4f0f8924f3/content \
  -o /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_w4a16_packed_t2v_default_gpu0_20260709_async.mp4
```

## Verification

```bash
ls -lh /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_w4a16_packed_t2v_default_gpu0_20260709_async.mp4
file /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_w4a16_packed_t2v_default_gpu0_20260709_async.mp4
curl -sS http://127.0.0.1:8022/v1/videos/video_gen_5d43ab3ccae043a5a4155d4f0f8924f3 | jq '{status,inference_time_s,peak_memory_mb,stage_durations}'
```

Expected high-level result from the validated run:

- MP4 exists and is about `9.3M`
- `status=completed`
- generation time about `13.3 min`

## Quick Smoke

Validated quick sync smoke on the direct packed path:

- output:
  - `/home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_w4a16_packed_quick_gpu0_20260709.mp4`
- request:
  - `size=1280x720`
  - `num_frames=17`
  - `fps=24`
  - `num_inference_steps=4`
  - `guidance_scale=6.0`
  - `flow_shift=10.0`
  - `max_sequence_length=4096`
  - `seed=17`
- observed wall time:
  - about `9.1s`

Command:

```bash
curl -sS -X POST http://127.0.0.1:8022/v1/videos/sync \
  -H 'Accept: video/mp4' \
  -F 'model=/storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed' \
  --form-string "prompt=$(jq -c . /storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed/assets/example_t2v_prompt.json)" \
  --form-string "negative_prompt=$(jq -c . /storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed/assets/negative_prompt.json)" \
  -F 'size=1280x720' \
  -F 'num_frames=17' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F 'guidance_scale=6.0' \
  -F 'max_sequence_length=4096' \
  -F 'flow_shift=10.0' \
  -F 'seed=17' \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false}' \
  -o /home/yiliu7/workspace/omni-wm/generated_videos/cosmos3_super/cosmos3_super_w4a16_packed_quick_gpu0_20260709.mp4
```

## Cleanup

Stop the server with `Ctrl-C`.
