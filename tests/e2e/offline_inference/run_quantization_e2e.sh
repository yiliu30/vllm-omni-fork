#!/bin/bash
# End-to-end test script for unified quantization framework (PR #1764).
#
# Tests FP8 quantization for:
#   1. Z-Image-Turbo  (single-stage, ~10GB VRAM)
#   2. Qwen-Image     (single-stage, ~10GB VRAM)
#   3. FLUX.1-dev     (single-stage, ~25GB VRAM, needs --quantization fp8 or OOM)
#   4. BAGEL          (multi-stage LLM+DiT, ~55GB VRAM)
#
# Usage:
#   bash tests/e2e/offline_inference/run_quantization_e2e.sh [--skip-flux] [--skip-bagel]
#
# Expected: all runs produce images without errors.
# Key check for BAGEL: FP8 only applies to diffusion stage, NOT the LLM stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/test_quant_outputs"
mkdir -p "$OUTPUT_DIR"

SKIP_FLUX=false
SKIP_BAGEL=false
for arg in "$@"; do
    case "$arg" in
        --skip-flux) SKIP_FLUX=true ;;
        --skip-bagel) SKIP_BAGEL=true ;;
    esac
done

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    shift
    echo ""
    echo "============================================================"
    echo "  TEST: $name"
    echo "============================================================"
    if "$@"; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

# ─── 1. Z-Image-Turbo ────────────────────────────────────────────────────────

run_test "Z-Image-Turbo BF16 (baseline)" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Tongyi-MAI/Z-Image-Turbo \
        --prompt "a cup of coffee on the table" \
        --seed 42 --num-inference-steps 2 \
        --height 256 --width 256 \
        --output "$OUTPUT_DIR/zimage_bf16.png"

run_test "Z-Image-Turbo FP8" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Tongyi-MAI/Z-Image-Turbo \
        --prompt "a cup of coffee on the table" \
        --seed 42 --num-inference-steps 2 \
        --height 256 --width 256 \
        --quantization fp8 \
        --output "$OUTPUT_DIR/zimage_fp8.png"

# ─── 2. Qwen-Image ───────────────────────────────────────────────────────────

run_test "Qwen-Image BF16 (baseline)" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Qwen/Qwen-Image \
        --prompt "a cup of coffee on the table" \
        --seed 42 --num-inference-steps 2 \
        --height 256 --width 256 \
        --output "$OUTPUT_DIR/qwen_bf16.png"

run_test "Qwen-Image FP8" \
    python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
        --model Qwen/Qwen-Image \
        --prompt "a cup of coffee on the table" \
        --seed 42 --num-inference-steps 2 \
        --height 256 --width 256 \
        --quantization fp8 \
        --output "$OUTPUT_DIR/qwen_fp8.png"

# ─── 3. FLUX.1-dev ───────────────────────────────────────────────────────────

if [ "$SKIP_FLUX" = true ]; then
    echo ""
    echo "  SKIP: FLUX.1-dev (--skip-flux)"
    SKIP=$((SKIP + 1))
    SKIP=$((SKIP + 1))
else
    run_test "FLUX.1-dev BF16 (baseline)" \
        python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
            --model black-forest-labs/FLUX.1-dev \
            --prompt "a cup of coffee on the table" \
            --seed 42 --num-inference-steps 4 \
            --height 512 --width 512 \
            --guidance-scale 3.5 --cfg-scale 1.0 \
            --output "$OUTPUT_DIR/flux_bf16.png"

    run_test "FLUX.1-dev FP8" \
        python "$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py" \
            --model black-forest-labs/FLUX.1-dev \
            --prompt "a cup of coffee on the table" \
            --seed 42 --num-inference-steps 4 \
            --height 512 --width 512 \
            --guidance-scale 3.5 --cfg-scale 1.0 \
            --quantization fp8 \
            --output "$OUTPUT_DIR/flux_fp8.png"
fi

# ─── 4. BAGEL (multi-stage) ──────────────────────────────────────────────────

if [ "$SKIP_BAGEL" = true ]; then
    echo ""
    echo "  SKIP: BAGEL (--skip-bagel)"
    SKIP=$((SKIP + 1))
    SKIP=$((SKIP + 1))
else
    run_test "BAGEL BF16 (baseline)" \
        python "$REPO_ROOT/examples/offline_inference/bagel/end2end.py" \
            --model ByteDance-Seed/BAGEL-7B-MoT \
            --modality text2img \
            --prompts "A cute cat" \
            --steps 15

    run_test "BAGEL FP8 (diffusion-only quantization)" \
        python "$REPO_ROOT/examples/offline_inference/bagel/end2end.py" \
            --model ByteDance-Seed/BAGEL-7B-MoT \
            --modality text2img \
            --prompts "A cute cat" \
            --steps 15 \
            --quantization fp8
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
echo "  Output images: $OUTPUT_DIR/"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
