#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
  set -u
fi

MODE="${MODE:-into_tensor}"
LAYOUT="${LAYOUT:-strided}"
NPROC="${NPROC:-4}"
BACKEND="${BACKEND:-xccl}"
DEVICE="${DEVICE:-xpu}"
SHAPE="${SHAPE:-1,48,48,45,80}"
WARMUP_SHAPE="${WARMUP_SHAPE:-1,48,1,32,32}"
ITERS="${ITERS:-1}"
TP_PRECOLLECTIVES="${TP_PRECOLLECTIVES:-0}"
PRODUCER_WORK="${PRODUCER_WORK:-none}"
PRODUCER_ITERS="${PRODUCER_ITERS:-0}"
SYNC_BEFORE_GATHER="${SYNC_BEFORE_GATHER:-0}"
TP_SLOW_RANKS="${TP_SLOW_RANKS:-}"
TP_DELAY_MS="${TP_DELAY_MS:-0}"
CFG_HOST_RENDEZVOUS="${CFG_HOST_RENDEZVOUS:-0}"
MODEL_HOST_RENDEZVOUS="${MODEL_HOST_RENDEZVOUS:-0}"

extra_args=()
if [[ "${SYNC_BEFORE_GATHER}" == "1" ]]; then
  extra_args+=(--sync-before-gather)
fi
if [[ "${CFG_HOST_RENDEZVOUS}" == "1" ]]; then
  extra_args+=(--cfg-host-rendezvous)
fi
if [[ "${MODEL_HOST_RENDEZVOUS}" == "1" ]]; then
  extra_args+=(--model-host-rendezvous)
fi

cd "${REPO_ROOT}"

exec torchrun --standalone --nproc_per_node="${NPROC}" \
  tools/repro/repro_xpu_cfg_allgather_hang.py \
  --backend "${BACKEND}" \
  --device "${DEVICE}" \
  --mode "${MODE}" \
  --layout "${LAYOUT}" \
  --cfg-parallel-size 2 \
  --tp-parallel-size 2 \
  --tp-precollectives "${TP_PRECOLLECTIVES}" \
  --tp-slow-ranks "${TP_SLOW_RANKS}" \
  --tp-delay-ms "${TP_DELAY_MS}" \
  --shape "${SHAPE}" \
  --warmup-shape "${WARMUP_SHAPE}" \
  --iters "${ITERS}" \
  --producer-work "${PRODUCER_WORK}" \
  --producer-iters "${PRODUCER_ITERS}" \
  --non-contiguous \
  "${extra_args[@]}"
