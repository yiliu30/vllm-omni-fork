#!/bin/bash
# FLUX.1-dev Benchmark Stage 2 - Full sweep
# Concurrency sweep + resolution/steps sweep
set -euo pipefail

PYTHON=/home/yiliu7/workspace/venvs/omni/bin/python
W4A16=/home/yiliu7/models/FLUX.1-dev-AutoRound-w4a16
BF16=/home/yiliu7/models/FLUX.1-dev
OUT=/home/yiliu7/workspace/vllm-omni/benchmarks/diffusion/results/b60_flux1_dev_stage2
PORT=8099
NUM_PROMPTS=10
STAGE_INIT_TIMEOUT=900

mkdir -p "$OUT"
cd /home/yiliu7/workspace/vllm-omni

LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo " FLUX.1-dev Benchmark - Stage 2 (Full)"
echo " Host: $(hostname)"
echo " Date: $(date)"
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
    local model=$1 conc=$2 label=$3 width=$4 height=$5 steps=$6
    local outfile="${OUT}/${label}_tp${TP}_c${conc}_${width}x${height}_s${steps}.json"
    echo ""
    echo "--- Benchmark: $label TP=$TP C=$conc ${width}x${height} steps=$steps ---"
    $PYTHON benchmarks/diffusion/diffusion_benchmark_serving.py \
        --backend vllm-omni \
        --base-url http://localhost:$PORT \
        --model "$model" \
        --task t2i --dataset vbench \
        --num-prompts $NUM_PROMPTS \
        --width "$width" --height "$height" \
        --num-inference-steps "$steps" \
        --max-concurrency "$conc" \
        --warmup-requests 1 --warmup-concurrency "$conc" \
        --output-file "$outfile"
    echo "--- Done: $(basename $outfile) ---"
}

# ============ W4A16 TP=2 concurrency sweep ============
kill_server
TP=2
if start_and_wait "$W4A16" $TP; then
    run_bench "$W4A16" 2 "w4a16" 1024 1024 20
    run_bench "$W4A16" 4 "w4a16" 1024 1024 20
    run_bench "$W4A16" 8 "w4a16" 1024 1024 20
fi
kill_server

# ============ W4A16 TP=4 concurrency + resolution sweep ============
TP=4
if start_and_wait "$W4A16" $TP; then
    run_bench "$W4A16" 2 "w4a16" 1024 1024 20
    run_bench "$W4A16" 4 "w4a16" 1024 1024 20
    run_bench "$W4A16" 8 "w4a16" 1024 1024 20
    # Resolution sweep
    run_bench "$W4A16" 1 "w4a16" 512 512 20
    run_bench "$W4A16" 1 "w4a16" 1024 1024 50
fi
kill_server

# ============ BF16 TP=2: SKIPPED (OOM during inference on B60) ============

# ============ BF16 TP=4 concurrency + resolution sweep ============
TP=4
if start_and_wait "$BF16" $TP; then
    run_bench "$BF16" 2 "bf16" 1024 1024 20
    run_bench "$BF16" 4 "bf16" 1024 1024 20
    run_bench "$BF16" 8 "bf16" 1024 1024 20
    # Resolution sweep
    run_bench "$BF16" 1 "bf16" 512 512 20
    run_bench "$BF16" 1 "bf16" 1024 1024 50
fi
kill_server

echo ""
echo "============================================"
echo " Stage 2 COMPLETE: $(date)"
echo " Results in: $OUT"
echo "============================================"
ls -la "$OUT"/*.json 2>/dev/null || echo "(no json files)"
