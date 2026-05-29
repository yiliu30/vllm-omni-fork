# FLUX.1-dev Serving Performance Dashboard

## 1. Overview

This document covers the online serving benchmark for **FLUX.1-dev** (bf16) and **FLUX.1-dev-AutoRound-w4a16** (W4A16 quantized) on vLLM-Omni, comparing latency, throughput, and memory usage across tensor-parallel sizes and concurrency levels.

---

## 2. Test Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5090 D (32GB) × 8 |
| Framework | vLLM-Omni v0.20.1.dev25 |
| Python | `/home/yiliu7/workspace/venvs/omni/bin/python` |
| Diffusion Backend | FlashAttention |
| Workaround | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7` |

---

## 3. Models

| Label | Path | Precision | Notes |
|-------|------|-----------|-------|
| bf16 | `/mnt/disk1/yiliu7/black-forest-labs/FLUX.1-dev` | BF16 | ~12B params, requires ≥2 GPUs |
| w4a16 | `/mnt/disk1/lvl/FLUX.1-dev-AutoRound-w4a16` | W4A16 | AutoRound quantized, fits single GPU |

---

## 4. Server Launch Configuration

```bash
# BF16, TP=2
python -m vllm_omni.entrypoints.cli.main serve \
    /mnt/disk1/yiliu7/black-forest-labs/FLUX.1-dev \
    --omni --tensor-parallel-size 2 --port 8099 --disable-log-stats

# W4A16, TP=1 (single GPU)
python -m vllm_omni.entrypoints.cli.main serve \
    /mnt/disk1/lvl/FLUX.1-dev-AutoRound-w4a16 \
    --omni --tensor-parallel-size 1 --port 8099 --disable-log-stats
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--tensor-parallel-size` | Tensor parallelism degree (1, 2, or 4) |
| `--omni` | Enable omni mode for diffusion serving |
| `--disable-log-stats` | Reduce noise in benchmarks |

### Feature Support Matrix

| Feature | Supported? | Notes |
|---------|-----------|-------|
| Tensor Parallel | ✅ | TP=1 (w4a16 only), 2, 4 tested |
| W4A16 Quantization | ✅ | AutoRound, enables single-GPU |
| CFG Parallel | ❌ | FLUX is guidance-distilled (no CFG) |
| Ulysses (USP) | ❌ | Not implemented in FLUX transformer |
| Step Execution (cont. batching) | ❌ | `supports_step_execution = False` |
| VAE Parallel | Untested | Model fits well without it |

---

## 5. Benchmark Script

### 5.1 Entry Point

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend vllm-omni \
    --base-url http://localhost:8099 \
    --model <MODEL_PATH> \
    --task t2i \
    --dataset vbench \
    --num-prompts 10 \
    --width 1024 --height 1024 \
    --num-inference-steps 20 \
    --max-concurrency 1 \
    --warmup-requests 1 --warmup-concurrency 1 \
    --output-file results/output.json
```

### 5.2 Key Benchmark Arguments

| Parameter | Description |
|-----------|-------------|
| `--backend` | Serving backend (`vllm-omni`) |
| `--dataset` | Dataset name (`vbench` for real prompts, `random` for synthetic) |
| `--task` | Task type (`t2i`) |
| `--num-prompts` | Total number of requests (10) |
| `--max-concurrency` | Client-side concurrency (1, 2, 4, 8) |
| `--warmup-requests` | Warmup requests to pre-capture CUDA graphs |
| `--warmup-concurrency` | Warmup concurrency (should match `--max-concurrency`) |
| `--width` / `--height` | Output image resolution |
| `--num-inference-steps` | Denoising steps |

---

## 6. Reproduction

### 6.1 Automated (Python)

```bash
cd /home/yiliu7/workspace/vllm-omni

# Show all 24 configs without running
python benchmarks/diffusion/reproduce_flux1_dev.py --dry-run

# Run everything (bf16 TP=2,4 + w4a16 TP=1,2,4)
python benchmarks/diffusion/reproduce_flux1_dev.py

# Run only w4a16
python benchmarks/diffusion/reproduce_flux1_dev.py --model w4a16

# Run only TP=4
python benchmarks/diffusion/reproduce_flux1_dev.py --tp 4

# Single specific config
python benchmarks/diffusion/reproduce_flux1_dev.py --model bf16 --tp 2 -c 1
```

### 6.2 Quick Single-Config (Shell)

```bash
cd /home/yiliu7/workspace/vllm-omni
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7

# TP=4, default sweep
TP_SIZE=4 bash benchmarks/diffusion/bench_flux1_dev.sh

# Show all configs (dry-run)
bash benchmarks/diffusion/reproduce_flux1_dev.sh --dry-run
```

### 6.3 Manual Steps

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7
PYTHON=/home/yiliu7/workspace/venvs/omni/bin/python

# 1. Start server
$PYTHON -m vllm_omni.entrypoints.cli.main serve \
    /mnt/disk1/lvl/FLUX.1-dev-AutoRound-w4a16 \
    --omni --tensor-parallel-size 1 --port 8099 --disable-log-stats &

# 2. Wait for health
until curl -s http://localhost:8099/health; do sleep 5; done

# 3. Run benchmark
$PYTHON benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend vllm-omni --base-url http://localhost:8099 \
    --model /mnt/disk1/lvl/FLUX.1-dev-AutoRound-w4a16 \
    --task t2i --dataset vbench --num-prompts 10 \
    --width 1024 --height 1024 --num-inference-steps 20 \
    --max-concurrency 1 --warmup-requests 1 --warmup-concurrency 1 \
    --output-file results/w4a16_tp1_c1.json

# 4. Kill server
kill %1
```

---

## 7. Performance Results

### 7.1 bf16 vs W4A16 — Concurrency Sweep (1024×1024, 20 steps)

| TP | Conc. | bf16 Mean (s) | W4A16 Mean (s) | Speedup | bf16 P99 (s) | W4A16 P99 (s) |
|----|-------|---------------|----------------|---------|--------------|----------------|
| 1 | 1 | OOM | 8.23 | — | OOM | 8.29 |
| 1 | 2 | OOM | 15.01 | — | OOM | 16.07 |
| 1 | 4 | OOM | 26.91 | — | OOM | 31.91 |
| 1 | 8 | OOM | 41.25 | — | OOM | 63.32 |
| 2 | 1 | 8.37 | 8.06 | 1.04× | 8.42 | 8.11 |
| 2 | 2 | 15.20 | 14.58 | 1.04× | 16.27 | 15.71 |
| 2 | 4 | 27.24 | 26.12 | 1.04× | 32.25 | 30.89 |
| 2 | 8 | 41.77 | 40.02 | 1.04× | 64.09 | 61.41 |
| 4 | 1 | 8.08 | 7.84 | 1.03× | 8.11 | 7.90 |
| 4 | 2 | 14.62 | 14.15 | 1.03× | 15.65 | 15.20 |
| 4 | 4 | 26.15 | 25.34 | 1.03× | 30.99 | 30.06 |
| 4 | 8 | 40.11 | 38.84 | 1.03× | 61.53 | 59.55 |

### 7.2 Resolution & Steps Sweep (TP=4, concurrency=1)

| Resolution | Steps | bf16 Mean (s) | W4A16 Mean (s) | Speedup |
|------------|-------|---------------|----------------|---------|
| 512×512 | 20 | 2.59 | 2.47 | 1.05× |
| 1024×1024 | 20 | 8.08 | 7.84 | 1.03× |
| 1024×1024 | 50 | 19.22 | 18.65 | 1.03× |

### 7.3 GPU Memory (per worker, model loading)

| Config | bf16 (GB) | W4A16 (GB) | Savings |
|--------|-----------|------------|---------|
| TP=1 | OOM (>32) | 15.91 | — |
| TP=2 | 22.68 | 10.33 | 54% |
| TP=4 | 17.96 | 7.35 | 59% |

---

## 8. Key Findings

1. **W4A16 enables single-GPU inference** — 15.9 GB fits comfortably on 32GB; bf16 OOMs on one card.
2. **W4A16 on 1 GPU (8.23s) ≈ bf16 on 4 GPUs (8.08s)** — same latency, 4× fewer GPUs.
3. **W4A16 is 3–5% faster** at same TP due to reduced memory bandwidth pressure.
4. **Memory savings**: 54% at TP=2, 59% at TP=4.
5. **No continuous batching** — throughput is flat regardless of concurrency (sequential execution).
6. **Latency scales linearly** with steps and quadratically with resolution.
7. **TP scaling**: TP=4 vs TP=2 gives ~1.04× speedup for bf16 (diminishing returns — already fast per step).

---

## 9. Related Files

| File | Description |
|------|-------------|
| `benchmarks/diffusion/reproduce_flux1_dev.py` | Full reproduction script (Python, supports --dry-run) |
| `benchmarks/diffusion/reproduce_flux1_dev.sh` | Shell-based reproduction script |
| `benchmarks/diffusion/bench_flux1_dev.sh` | Quick single-model benchmark (configurable via env vars) |
| `benchmarks/diffusion/diffusion_benchmark_serving.py` | Core benchmark engine (shared with Qwen-Image) |
| `benchmarks/diffusion/results/flux1_dev_dashboard.html` | Interactive HTML dashboard with charts |
| `benchmarks/diffusion/results/flux1_dev_reproduce_20260529_012416/` | Latest raw results (24 JSON files) |
| `vllm_omni/entrypoints/async_omni.py` | Needed stub for `notify_kv_transfer_request_rejected` |

---

## 10. Reproducibility Checklist

- [ ] Set `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7`
- [ ] No background GPU workload during testing
- [ ] Use `--warmup-requests` matching `--max-concurrency` to pre-capture CUDA graphs
- [ ] Record GPU type, TP size, and model precision
- [ ] Run with same `--num-prompts` (10) and `--dataset` (vbench) for comparability
