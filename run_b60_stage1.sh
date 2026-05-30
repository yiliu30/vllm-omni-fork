#!/bin/bash
# FLUX.1-dev Benchmark Stage 1 - Intel Arc Pro B60 x4
# Quick baseline: single-concurrency across TP configs
set -euo pipefail

PYTHON=/home/yiliu7/workspace/venvs/omni/bin/python
W4A16=/home/yiliu7/models/FLUX.1-dev-AutoRound-w4a16
BF16=/home/yiliu7/models/FLUX.1-dev
OUT=/home/yiliu7/workspace/vllm-omni/benchmarks/diffusion/results/b60_flux1_dev_stage1
PORT=8099
NUM_PROMPTS=10
STEPS=20
WIDTH=1024
HEIGHT=1024
# XPU model loading is slower than CUDA; increase timeout to avoid SIGTERM during init
STAGE_INIT_TIMEOUT=900

mkdir -p "$OUT"
cd /home/yiliu7/workspace/vllm-omni

LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo " FLUX.1-dev Benchmark - Stage 1 (Quick)"
echo " Host: $(hostname)"
echo " Date: $(date)"
echo " GPU:  Intel Arc Pro B60 x4"
echo "============================================"

kill_server() {
    pkill -f "vllm_omni.entrypoints.cli.main serve" 2>/dev/null || true
    sleep 5
}

start_and_wait() {
    local model=$1 tp=$2
    echo ""
    echo ">>> Starting server: $(basename $model) TP=$tp"
    $PYTHON -m vllm_omni.entrypoints.cli.main serve "$model" \
        --omni --tensor-parallel-size "$tp" --port $PORT --disable-log-stats \
        --stage-init-timeout $STAGE_INIT_TIMEOUT &
    SERVER_PID=$!
    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo ">>> Server healthy (waited $((i*5))s)"
            return 0
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ">>> ERROR: Server process died"
            return 1
        fi
        sleep 5
    done
    echo ">>> ERROR: Server startup timeout (300s)"
    return 1
}

run_bench() {
    local model=$1 conc=$2 label=$3
    local outfile="${OUT}/${label}_tp${TP}_c${conc}_${WIDTH}x${HEIGHT}_s${STEPS}.json"
    echo ""
    echo "--- Benchmark: $label TP=$TP C=$conc ---"
    echo "    Output: $(basename $outfile)"
    $PYTHON benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend vllm-omni \
        --base-url http://localhost:$PORT \
        --model "$model" \
        --task t2i --dataset vbench \
        --num-prompts $NUM_PROMPTS \
        --width $WIDTH --height $HEIGHT \
        --num-inference-steps $STEPS \
        --max-concurrency "$conc" \
        --warmup-requests 1 --warmup-concurrency "$conc" \
        --output-file "$outfile"
    echo "--- Done: $(basename $outfile) ---"
}

# ============ W4A16 TP=1 ============
kill_server
TP=1
if start_and_wait "$W4A16" $TP; then
    run_bench "$W4A16" 1 "w4a16"
    run_bench "$W4A16" 2 "w4a16"
    run_bench "$W4A16" 4 "w4a16"
    run_bench "$W4A16" 8 "w4a16"
fi
kill_server

# ============ W4A16 TP=2 ============
TP=2
if start_and_wait "$W4A16" $TP; then
    run_bench "$W4A16" 1 "w4a16"
fi
kill_server

# ============ W4A16 TP=4 ============
TP=4
if start_and_wait "$W4A16" $TP; then
    run_bench "$W4A16" 1 "w4a16"
fi
kill_server

# ============ BF16 TP=2 ============
TP=2
if start_and_wait "$BF16" $TP; then
    run_bench "$BF16" 1 "bf16"
fi
kill_server

# ============ BF16 TP=4 ============
TP=4
if start_and_wait "$BF16" $TP; then
    run_bench "$BF16" 1 "bf16"
fi
kill_server

echo ""
echo "============================================"
echo " Stage 1 COMPLETE: $(date)"
echo " Results in: $OUT"
echo "============================================"
echo ""
echo "Results summary:"
ls -la "$OUT"/*.json 2>/dev/null || echo "(no json files)"
