"""Compare BF16 vs FP8 Qwen3-Omni: memory, speed, and output quality."""

import argparse
import gc
import json
import time

import torch
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni

PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Rust?",
    "Summarize the key ideas behind transformer architecture.",
    "How does photosynthesis work? Answer in 3 sentences.",
]

SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def build_prompt(question: str) -> dict:
    return {
        "prompt": (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "modalities": ["text"],
    }


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def run_benchmark(model_path, stage_config, label, seed=42):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    torch.cuda.reset_peak_memory_stats()

    # Load model
    t0 = time.time()
    omni = Omni(
        model=model_path,
        stage_configs_path=stage_config,
        stage_init_timeout=600,
    )
    load_time = time.time() - t0
    print(f"Load time: {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=200,
        seed=seed,
    )

    prompts = [build_prompt(q) for q in PROMPTS]

    # Warmup
    print("Warming up...")
    warmup_prompt = [build_prompt("Hello")]
    list(omni.generate(warmup_prompt, [sampling_params]))

    # Benchmark
    print(f"Running {len(PROMPTS)} prompts...")
    t0 = time.time()
    results = list(omni.generate(prompts, [sampling_params]))
    gen_time = time.time() - t0

    # Collect outputs
    outputs = []
    total_tokens = 0
    for stage_output in results:
        req_output = stage_output.request_output
        text = req_output.outputs[0].text
        n_tokens = len(req_output.outputs[0].token_ids)
        total_tokens += n_tokens
        outputs.append(text)

    peak_mem = get_gpu_memory_mb()
    tok_per_sec = total_tokens / gen_time if gen_time > 0 else 0

    summary = {
        "label": label,
        "model": model_path,
        "load_time_s": round(load_time, 1),
        "peak_memory_gib": round(peak_mem / 1024, 2),
        "total_tokens": total_tokens,
        "generation_time_s": round(gen_time, 2),
        "tokens_per_sec": round(tok_per_sec, 1),
        "outputs": outputs,
    }

    print(f"Peak memory:    {summary['peak_memory_gib']} GiB")
    print(f"Total tokens:   {total_tokens}")
    print(f"Generation:     {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")
    print(f"\nSample output (prompt 0):\n  {outputs[0][:200]}...")

    # Cleanup
    del omni
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare BF16 vs FP8 Qwen3-Omni")
    parser.add_argument(
        "--bf16-model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="BF16 model path or HF ID",
    )
    parser.add_argument(
        "--fp8-model",
        default="/root/.cache/modelscope/hub/models/asdazd/Qwen3-Omni-30B-A3B-Instruct_modelopt_FP8",
        help="FP8 model path",
    )
    parser.add_argument(
        "--bf16-config",
        default="bf16_stage_config.yaml",
        help="BF16 stage config",
    )
    parser.add_argument(
        "--fp8-config",
        default="fp8_stage_config.yaml",
        help="FP8 stage config",
    )
    parser.add_argument(
        "--skip-bf16",
        action="store_true",
        help="Skip BF16 run (if you already have results)",
    )
    parser.add_argument(
        "--skip-fp8",
        action="store_true",
        help="Skip FP8 run",
    )
    args = parser.parse_args()

    results = {}

    if not args.skip_bf16:
        results["bf16"] = run_benchmark(args.bf16_model, args.bf16_config, "BF16 Baseline")

    if not args.skip_fp8:
        results["fp8"] = run_benchmark(args.fp8_model, args.fp8_config, "FP8 (ModelOpt)")

    # Print comparison
    if "bf16" in results and "fp8" in results:
        bf16 = results["bf16"]
        fp8 = results["fp8"]
        mem_reduction = (1 - fp8["peak_memory_gib"] / bf16["peak_memory_gib"]) * 100
        speedup = fp8["tokens_per_sec"] / bf16["tokens_per_sec"] if bf16["tokens_per_sec"] > 0 else 0

        print(f"\n{'=' * 60}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Metric':<25} {'BF16':>12} {'FP8':>12} {'Delta':>12}")
        print(f"{'-' * 61}")
        bf16_mem = bf16["peak_memory_gib"]
        fp8_mem = fp8["peak_memory_gib"]
        print(f"{'Peak Memory (GiB)':<25} {bf16_mem:>12} {fp8_mem:>12} {mem_reduction:>11.0f}%")
        print(f"{'Load Time (s)':<25} {bf16['load_time_s']:>12} {fp8['load_time_s']:>12}")
        print(f"{'Tokens/sec':<25} {bf16['tokens_per_sec']:>12} {fp8['tokens_per_sec']:>12} {speedup:>11.2f}x")
        print()

        # Output comparison
        print("OUTPUT COMPARISON (greedy, temperature=0):")
        for i, prompt in enumerate(PROMPTS):
            print(f"\n--- Prompt {i}: {prompt[:60]}...")
            print(f"  BF16: {bf16['outputs'][i][:150]}...")
            print(f"  FP8:  {fp8['outputs'][i][:150]}...")

    # Save results
    out_file = "fp8_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
