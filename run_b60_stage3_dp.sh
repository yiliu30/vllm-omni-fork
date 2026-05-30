#!/bin/bash
# FLUX.1-dev Benchmark Stage 3 - Data Parallel (W4A16 Advantage Showcase)
#
# Goal: Demonstrate W4A16's unique throughput advantage on B60 via data parallelism.
# bf16 CANNOT do DP on B60 (requires all 4 GPUs for TP=4 just to fit).
# W4A16 fits on 1 GPU → enables DP=4 for 4× throughput.
#
# Configs:
#   1. W4A16 TP=1, DP=4, C=1,4,8  (max throughput, 4 replicas)
#   2. W4A16 TP=2, DP=2, C=1,2,4  (balanced latency/throughput)
#   3. W4A16 TP=4, DP=1, C=1,4    (baseline, already known)
#   4. bf16  TP=4, DP=1, C=1,4    (baseline comparison)
#
# Expected outcome:
#   W4A16 TP=1 DP=4 @ C=4 → ~0.133 qps (2.4× better than bf16 TP=4)
#   W4A16 TP=2 DP=2 @ C=2 → ~0.087 qps (1.6× better than bf16 TP=4)
#
set -euo pipefail

PYTHON=/home/yiliu7/workspace/venvs/omni/bin/python
W4A16=/home/yiliu7/models/FLUX.1-dev-AutoRound-w4a16
BF16=/home/yiliu7/models/FLUX.1-dev
OUT=/home/yiliu7/workspace/vllm-omni/benchmarks/diffusion/results/b60_flux1_dev_stage3_dp
PORT=8099
NUM_PROMPTS=10
STEPS=20
WIDTH=1024
HEIGHT=1024
STAGE_INIT_TIMEOUT=900

mkdir -p "$OUT"
cd /home/yiliu7/workspace/vllm-omni

LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo " FLUX.1-dev Benchmark - Stage 3 (Data Parallel)"
echo " Goal: Show W4A16 throughput advantage via DP"
echo " Host: $(hostname)"
echo " Date: $(date)"
echo " GPU:  Intel Arc Pro B60 x4 (24.4 GB each)"
echo "============================================"

kill_server() {
    pkill -f "vllm_omni.entrypoints.cli.main serve" 2>/dev/null || true
    sleep 5
}

start_and_wait() {
    local model=$1 tp=$2 dp=$3
    echo ""
    echo ">>> Starting server: $(basename $model) TP=$tp DP=$dp"
    $PYTHON -m vllm_omni.entrypoints.cli.main serve "$model" \
        --omni \
        --tensor-parallel-size "$tp" \
        --data-parallel-size "$dp" \
        --port $PORT \
        --disable-log-stats \
        --stage-init-timeout $STAGE_INIT_TIMEOUT &
    SERVER_PID=$!
    # DP=4 with bf16 loading takes longer; wait up to 15 min
    local max_wait=180
    for i in $(seq 1 $max_wait); do
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
    echo ">>> ERROR: Server startup timeout ($((max_wait*5))s)"
    return 1
}

run_bench() {
    local model=$1 conc=$2 label=$3
    local outfile="${OUT}/${label}_tp${TP}_dp${DP}_c${conc}_${WIDTH}x${HEIGHT}_s${STEPS}.json"
    echo ""
    echo "--- Benchmark: $label TP=$TP DP=$DP C=$conc ${WIDTH}x${HEIGHT} steps=$STEPS ---"
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

# ============================================================
# Config 1: W4A16 TP=1, DP=4 (Maximum throughput)
# 4 independent replicas, each on 1 GPU
# Expected: ~0.133 qps at C=4 (4x single-GPU throughput)
# ============================================================
kill_server
TP=1; DP=4
if start_and_wait "$W4A16" $TP $DP; then
    run_bench "$W4A16" 1 "w4a16"   # Below saturation (only 1 of 4 replicas busy)
    run_bench "$W4A16" 4 "w4a16"   # Saturated (all 4 replicas busy)
    run_bench "$W4A16" 8 "w4a16"   # Over-saturated (2 requests queued per replica)
fi
kill_server

# ============================================================
# Config 2: W4A16 TP=2, DP=2 (Balanced latency + throughput)
# 2 replicas, each using 2 GPUs
# Expected: ~0.087 qps at C=2 (better latency than TP=1, 2x throughput)
# ============================================================
TP=2; DP=2
if start_and_wait "$W4A16" $TP $DP; then
    run_bench "$W4A16" 1 "w4a16"   # Below saturation
    run_bench "$W4A16" 2 "w4a16"   # Saturated (both replicas busy)
    run_bench "$W4A16" 4 "w4a16"   # Over-saturated
fi
kill_server

# ============================================================
# Config 3 & 4: SKIPPED — already have data from Stage 1/2:
#   W4A16 TP=4 DP=1 C=1: 19.01s, 0.0526 qps
#   W4A16 TP=4 DP=1 C=4: 63.22s, 0.0538 qps
#   bf16  TP=4 DP=1 C=1: 18.26s, 0.0548 qps
#   bf16  TP=4 DP=1 C=4: 60.65s, 0.0561 qps
# ============================================================

echo ""
echo "============================================"
echo " Stage 3 (Data Parallel) COMPLETE: $(date)"
echo " Results in: $OUT"
echo "============================================"
echo ""
echo "Results summary:"
ls -la "$OUT"/*.json 2>/dev/null || echo "(no json files)"
echo ""
echo "Expected key comparison:"
echo "  bf16  TP=4 DP=1 C=4 → ~0.056 qps (baseline)"
echo "  W4A16 TP=2 DP=2 C=2 → ~0.087 qps (1.6x throughput)"
echo "  W4A16 TP=1 DP=4 C=4 → ~0.133 qps (2.4x throughput)"
