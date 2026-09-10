#!/usr/bin/env bash
# SAGE_ATTN_3 (XPU / deepklox Sage V3 Hybrid) run for Wan2.2 T2V on yi-wan,
# with negative prompt + dual guidance scales.
set -euo pipefail

WAN_RUN_NUM_INFERENCE_STEPS="${1:-40}"
WAN_RUN_NUM_FRAMES="${2:-81}"
WAN_XPU="${3:-0}"  # device label = ZE affinity mask (0=xpu0, 1=xpu1)
source /opt/gfx-deps/env.sh
source /home/ubuntu/main_cri_toolchain/env.sh
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="${WAN_XPU}"
# Interim: prebuilt deepklox (Sage V3 Hybrid, 0908) until it is baked into the CRI image.
export PYTHONPATH="/workspace/vllm-omni-sage/tmp_yi/deepklox-sage_0908${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3
if [[ -n "${WAN_SAGE_FALLBACK_BLOCKS:-}" ]]; then
  export SAGE_ATTN_FORCE_SDPA_BLOCKS="${WAN_SAGE_FALLBACK_BLOCKS}"
  RUN_VARIANT="selective"
else
  unset SAGE_ATTN_FORCE_SDPA_BLOCKS
  RUN_VARIANT="nofallback"
fi
if [[ "${WAN_SAGE_REPORT_FALLBACKS:-0}" == "1" ]]; then
  export SAGE_ATTN_REPORT_FALLBACKS=1
else
  unset SAGE_ATTN_REPORT_FALLBACKS
fi

WAN_NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

OUTPUT_DIR="/workspace/tmp_yi_yiwan"
mkdir -p "${OUTPUT_DIR}"
OUTPUT="${OUTPUT_DIR}/wan22_output_sagev3_${RUN_VARIANT}_mxfp8_cachedit_steps${WAN_RUN_NUM_INFERENCE_STEPS}_frames${WAN_RUN_NUM_FRAMES}_xpu${WAN_XPU}_noneager_negprompt.mp4"
LOG="${OUTPUT_DIR}/wan22_sagev3_${RUN_VARIANT}_mxfp8_cachedit_steps${WAN_RUN_NUM_INFERENCE_STEPS}_frames${WAN_RUN_NUM_FRAMES}_xpu${WAN_XPU}_noneager_negprompt.log"

/opt/gfx-deps/venv/bin/python \
  /workspace/vllm-omni-inner/examples/offline_inference/text_to_video/text_to_video.py \
  --model /workspace/hf_models/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative-prompt "${WAN_NEGATIVE_PROMPT}" \
  --guidance-scale 4.0 \
  --guidance-scale-high 3.0 \
  --height 720 \
  --width 1280 \
  --num-frames "${WAN_RUN_NUM_FRAMES}" \
  --num-inference-steps "${WAN_RUN_NUM_INFERENCE_STEPS}" \
  --boundary-ratio 0.875 \
  --flow-shift 5.0 \
  --fps 16 \
  --tensor-parallel-size 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --quantization mxfp8 \
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --output "${OUTPUT}" 2>&1 | tee "${LOG}"
