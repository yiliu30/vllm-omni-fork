#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce all FLUX.1-dev benchmark results (bf16 and w4a16).

Usage:
    python reproduce_flux1_dev.py --dry-run          # show all configs
    python reproduce_flux1_dev.py                    # run all benchmarks
    python reproduce_flux1_dev.py --model w4a16      # run only quantized model
    python reproduce_flux1_dev.py --tp 4             # run only TP=4 configs
    python reproduce_flux1_dev.py --model bf16 --tp 2 -c 1  # single config
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_SCRIPT = SCRIPT_DIR / "diffusion_benchmark_serving.py"

MODELS = {
    "bf16": "/mnt/disk1/yiliu7/black-forest-labs/FLUX.1-dev",
    "w4a16": "/mnt/disk1/lvl/FLUX.1-dev-AutoRound-w4a16",
}

PYTHON = "/home/yiliu7/workspace/venvs/omni/bin/python"
PORT = 8099
BASE_URL = f"http://localhost:{PORT}"


@dataclass
class BenchConfig:
    model: str
    tp: int
    width: int
    height: int
    steps: int
    concurrency: int

    @property
    def model_name(self) -> str:
        for name, path in MODELS.items():
            if path == self.model:
                return name
        return "unknown"

    @property
    def tag(self) -> str:
        return f"{self.model_name}_tp{self.tp}_{self.width}x{self.height}_s{self.steps}_c{self.concurrency}"

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def _make_configs(model_name: str, tp_sizes: list[int]) -> list[BenchConfig]:
    """Generate standard config set for a model."""
    model_path = MODELS[model_name]
    configs = []
    for tp in tp_sizes:
        # Concurrency sweep at 1024x1024, 20 steps
        for c in [1, 2, 4, 8]:
            configs.append(BenchConfig(model_path, tp, 1024, 1024, 20, c))
    # Resolution & steps sweep at max TP, concurrency=1
    max_tp = max(tp_sizes)
    configs.append(BenchConfig(model_path, max_tp, 512, 512, 20, 1))
    configs.append(BenchConfig(model_path, max_tp, 1024, 1024, 50, 1))
    return configs


ALL_CONFIGS = [
    *_make_configs("bf16", [2, 4]),
    *_make_configs("w4a16", [1, 2, 4]),
]


def print_configs(configs: list[BenchConfig]):
    """Print all benchmark configs in a table."""
    print("=== FLUX.1-dev Benchmark Configs ===\n")
    print(f"  Models:  {', '.join(f'{k}={v}' for k, v in MODELS.items())}")
    print(f"  Python:  {PYTHON}")
    print(f"  Port:    {PORT}")
    print(f"  Dataset: vbench (10 prompts)\n")
    print(f"{'#':<4} {'Model':<8} {'TP':<4} {'Concurrency':<12} {'Resolution':<12} {'Steps':<6}")
    print(f"{'---':<4} {'---':<8} {'---':<4} {'---':<12} {'---':<12} {'---':<6}")
    for i, c in enumerate(configs, 1):
        print(f"{i:<4} {c.model_name:<8} {c.tp:<4} {c.concurrency:<12} {c.resolution:<12} {c.steps:<6}")
    groups = sorted(set((c.model_name, c.tp) for c in configs))
    print(f"\nTotal: {len(configs)} runs ({len(groups)} server launch(es))")
    for name, tp in groups:
        print(f"  - {name} TP={tp}")


def start_server(model: str, tp: int, outdir: Path) -> subprocess.Popen:
    """Launch vLLM-Omni server."""
    model_name = "unknown"
    for k, v in MODELS.items():
        if v == model:
            model_name = k
            break
    log = open(outdir / f"server_{model_name}_tp{tp}.log", "w")
    env = os.environ.copy()
    env.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7")
    cmd = [
        PYTHON, "-m", "vllm_omni.entrypoints.cli.main", "serve", model,
        "--omni",
        "--tensor-parallel-size", str(tp),
        "--port", str(PORT),
        "--disable-log-stats",
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"  Server PID={proc.pid} ({model_name}, TP={tp})")
    return proc


def wait_for_server(timeout: int = 600):
    """Wait until server health endpoint responds."""
    import urllib.request
    elapsed = 0
    while elapsed < timeout:
        try:
            urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
            print(f"  Server ready ({elapsed}s)")
            return
        except Exception:
            time.sleep(5)
            elapsed += 5
    raise TimeoutError(f"Server not ready after {timeout}s")


def kill_server(proc: subprocess.Popen):
    """Gracefully kill the server process."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print("  Server stopped")


def run_benchmark(config: BenchConfig, outdir: Path) -> dict | None:
    """Run a single benchmark and return parsed results."""
    output_file = outdir / f"{config.tag}.json"
    cmd = [
        PYTHON, str(BENCH_SCRIPT),
        "--backend", "vllm-omni",
        "--base-url", BASE_URL,
        "--model", config.model,
        "--task", "t2i",
        "--dataset", "vbench",
        "--num-prompts", "10",
        "--width", str(config.width),
        "--height", str(config.height),
        "--num-inference-steps", str(config.steps),
        "--max-concurrency", str(config.concurrency),
        "--warmup-requests", str(config.concurrency),
        "--warmup-concurrency", str(config.concurrency),
        "--output-file", str(output_file),
    ]
    env = os.environ.copy()
    env.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7")

    print(f"  [{config.tag}] running...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("FAILED")
        err_file = outdir / f"{config.tag}.err"
        err_file.write_text(result.stdout + "\n" + result.stderr)
        return None

    if output_file.exists():
        data = json.loads(output_file.read_text())
        mean = data.get("latency_mean", "?")
        p99 = data.get("latency_p99", "?")
        print(f"done (mean={mean:.2f}s, p99={p99:.2f}s)" if isinstance(mean, float) else "done")
        return data

    print("done (no output file)")
    return None


def print_comparison(results: dict[str, dict], configs: list[BenchConfig]):
    """Print side-by-side comparison of bf16 vs w4a16."""
    # Find matching pairs: same (tp, width, height, steps, concurrency)
    pairs: dict[tuple, dict[str, dict]] = {}
    for config in configs:
        key = (config.tp, config.width, config.height, config.steps, config.concurrency)
        if config.tag in results:
            pairs.setdefault(key, {})[config.model_name] = results[config.tag]

    # Only show pairs where both models have results
    comparable = {k: v for k, v in pairs.items() if len(v) >= 2}
    if not comparable:
        return

    print("\n=== bf16 vs w4a16 Comparison ===\n")
    print(f"{'TP':<4} {'Resolution':<12} {'Steps':<6} {'Conc.':<6} "
          f"{'bf16 Mean':<12} {'w4a16 Mean':<12} {'Speedup':<8} "
          f"{'bf16 Mem':<10} {'w4a16 Mem':<10} {'Mem Save':<8}")
    print("-" * 100)

    for key in sorted(comparable.keys()):
        tp, w, h, steps, conc = key
        bf16 = comparable[key].get("bf16", {})
        w4 = comparable[key].get("w4a16", {})
        bf16_mean = bf16.get("latency_mean", 0)
        w4_mean = w4.get("latency_mean", 0)
        bf16_mem = bf16.get("peak_memory_max_mb", 0)
        w4_mem = w4.get("peak_memory_max_mb", 0)

        speedup = f"{bf16_mean / w4_mean:.2f}x" if w4_mean > 0 else "N/A"
        mem_save = f"{(1 - w4_mem / bf16_mem) * 100:.0f}%" if bf16_mem > 0 else "N/A"

        print(f"{tp:<4} {w}x{h:<7} {steps:<6} {conc:<6} "
              f"{bf16_mean:<12.2f} {w4_mean:<12.2f} {speedup:<8} "
              f"{bf16_mem:<10.0f} {w4_mem:<10.0f} {mem_save:<8}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce FLUX.1-dev benchmarks (bf16 & w4a16)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show configs without running")
    parser.add_argument("--model", "-m", choices=["bf16", "w4a16", "all"], default="all",
                        help="Which model to benchmark")
    parser.add_argument("--tp", type=int, choices=[1, 2, 4], help="Run only configs for this TP size")
    parser.add_argument("--concurrency", "-c", type=int, help="Run only this concurrency level")
    parser.add_argument("--output-dir", "-o", type=str, help="Override output directory")
    args = parser.parse_args()

    # Filter configs
    configs = ALL_CONFIGS
    if args.model != "all":
        model_path = MODELS[args.model]
        configs = [c for c in configs if c.model == model_path]
    if args.tp:
        configs = [c for c in configs if c.tp == args.tp]
    if args.concurrency:
        configs = [c for c in configs if c.concurrency == args.concurrency]

    if not configs:
        print("No configs match the given filters.")
        sys.exit(1)

    if args.dry_run:
        print_configs(configs)
        return

    # Run benchmarks
    outdir = Path(args.output_dir) if args.output_dir else (
        SCRIPT_DIR / "results" / f"flux1_dev_reproduce_{datetime.now():%Y%m%d_%H%M%S}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"=== FLUX.1-dev Benchmark ===")
    print(f"  Output: {outdir}\n")

    # Group by (model, tp) to minimize server restarts
    groups: dict[tuple[str, int], list[BenchConfig]] = {}
    for c in configs:
        groups.setdefault((c.model, c.tp), []).append(c)

    results = {}
    for (model, tp), group in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        model_name = group[0].model_name
        print(f"--- {model_name} TP={tp} ({len(group)} runs) ---")
        proc = start_server(model, tp, outdir)
        try:
            wait_for_server()
            for config in group:
                data = run_benchmark(config, outdir)
                if data:
                    results[config.tag] = data
        finally:
            kill_server(proc)
        print()

    # Summary
    print(f"=== Complete: {len(results)}/{len(configs)} succeeded ===")
    print(f"Results: {outdir}")

    # Comparison table
    print_comparison(results, configs)


if __name__ == "__main__":
    main()
