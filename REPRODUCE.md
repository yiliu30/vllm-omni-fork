# Reproduce

This file collects the exact commands used to validate the Wan2.2 sparse-attention work in this tree.

## 1. Triton-XPU sparse executor regression test

```bash
cd /data/model/yiliu7/vllm-omni
source /opt/intel/oneapi/setvars.sh >/dev/null
uv run --no-sync python \
  /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/test/test_triton_sparse_prefill_e2e.py
```

## 2. Reduced Wan 9-frame run with Triton-XPU sparse self-attention

```bash
cd /data/model/yiliu7/vllm-omni
sg render -c 'bash -lc "
  set -eo pipefail
  set +u
  source /opt/intel/oneapi/setvars.sh >/dev/null
  set -u
  ulimit -n 1048576
  export SYCL_UR_USE_LEVEL_ZERO_V2=0
  export ZE_AFFINITY_MASK=0,1,2,3
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
  export SAGE_ATTN_XPU_BACKEND=sparse
  export SAGE_ATTN_TOPK=0.5
  export SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND=triton_xpu_kernel
  export AUTO_ROUND_KERNEL_PATH=/home/yiliu7/workspace/auto-round/auto_round_extension/ark
  export UV_CACHE_DIR=/tmp/uvcache
  export PYTHONUNBUFFERED=1
  uv run --no-sync python /data/model/yiliu7/vllm-omni/tmp_run_wan_per_role_attention.py \
    --attention-mode mixed \
    --model /data/model/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt \"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.\" \
    --negative-prompt \"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走\" \
    --output /tmp/wan22_triton_sparse_topk0p5_9f.mp4 \
    --height 480 \
    --width 832 \
    --num-frames 9 \
    --num-inference-steps 40 \
    --boundary-ratio 0.875 \
    --flow-shift 5.0 \
    --fps 16 \
    --seed 42
"'
```

## 3. Extract frames from the reduced Wan run

```bash
cd /data/model/yiliu7/vllm-omni
uv run --no-sync python - <<'PY'
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image

video = Path("/tmp/wan22_triton_sparse_topk0p5_9f.mp4")
out_dir = Path("/tmp/wan22_triton_sparse_topk0p5_9f_frames")
out_dir.mkdir(parents=True, exist_ok=True)
frames = iio.imread(video)
if frames.ndim == 3:
    frames = frames[None, ...]
for idx, frame in enumerate(frames, start=1):
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_dir / f"frame_{idx:03d}.png")
print(out_dir)
PY
```

## Related docs

- [WAN22_SPARSE_STATUS.md](/data/model/yiliu7/vllm-omni/WAN22_SPARSE_STATUS.md)
- [WAN22_SPARSE_DEBUG_PROCESS.md](/data/model/yiliu7/vllm-omni/WAN22_SPARSE_DEBUG_PROCESS.md)
- [WAN22_SPARSE_XPU_VS_CUDA_GAP_ANALYSIS.md](/data/model/yiliu7/vllm-omni/WAN22_SPARSE_XPU_VS_CUDA_GAP_ANALYSIS.md)
