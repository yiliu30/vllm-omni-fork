#!/usr/bin/env python3
"""Stage 4: CFG Parallel Benchmark - W4A16 TP=2 CFG=2 vs bf16 TP=4 CFG=1
All configs use 4 GPUs, true_cfg_scale=4.0, negative prompt.
Story: W4A16 gets CFG-guided quality at same latency bf16 gets without guidance.
"""
import subprocess, time, requests, base64, json, os, sys, signal

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_W4A16 = "/home/yiliu7/models/FLUX.1-dev-AutoRound-w4a16"
MODEL_BF16 = "/home/yiliu7/models/FLUX.1-dev"
PYTHON = "/home/yiliu7/workspace/venvs/omni/bin/python"
OUTDIR = "/home/yiliu7/workspace/vllm-omni/benchmarks/diffusion/results/b60_flux1_dev_stage4_cfg"
LOGDIR = "/home/yiliu7/workspace/vllm-omni/cfg_stage4_logs"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

PROMPT = "A majestic snow-capped mountain reflected in a crystal clear lake at golden hour, photorealistic"
NEGATIVE = "low quality, blurry, distorted, watermark, text, ugly"
TRUE_CFG_SCALE = 4.0
SEED = 42

CONFIGS = [
    {"name": "w4a16_tp2_cfg2_512x512_s20", "model": MODEL_W4A16, "tp": 2, "cfg": 2, "res": "512x512", "steps": 20},
    {"name": "bf16_tp4_cfg1_512x512_s20", "model": MODEL_BF16, "tp": 4, "cfg": 1, "res": "512x512", "steps": 20},
    {"name": "w4a16_tp2_cfg2_1024x1024_s20", "model": MODEL_W4A16, "tp": 2, "cfg": 2, "res": "1024x1024", "steps": 20},
    {"name": "bf16_tp4_cfg1_1024x1024_s20", "model": MODEL_BF16, "tp": 4, "cfg": 1, "res": "1024x1024", "steps": 20},
    {"name": "w4a16_tp2_cfg2_1536x1536_s20", "model": MODEL_W4A16, "tp": 2, "cfg": 2, "res": "1536x1536", "steps": 20},
    {"name": "bf16_tp4_cfg1_1536x1536_s20", "model": MODEL_BF16, "tp": 4, "cfg": 1, "res": "1536x1536", "steps": 20},
]

PORT = 8100
NUM_RUNS = 3  # 3 runs per config, take median


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def start_server(config):
    args = [
        PYTHON, "-m", "vllm_omni.entrypoints.cli.main", "serve", config["model"],
        "--omni", "--port", str(PORT),
        "--disable-log-stats", "--stage-init-timeout", "1200",
        "--init-timeout", "1200",
        "--enforce-eager",
        "--tensor-parallel-size", str(config["tp"]),
    ]
    if config["cfg"] > 1:
        args += ["--cfg-parallel-size", str(config["cfg"])]

    cmd_str = " ".join(args)
    log(f"  CMD: {cmd_str}")

    logpath = f"{LOGDIR}/{config['name']}_server.log"
    logfile = open(logpath, "w")
    proc = subprocess.Popen(args, stdout=logfile, stderr=subprocess.STDOUT,
                            cwd="/home/yiliu7/workspace/vllm-omni",
                            start_new_session=True)
    return proc, logfile, cmd_str


def wait_healthy(timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                elapsed = time.time() - t0
                log(f"  Server healthy after {elapsed:.0f}s")
                return True
        except:
            pass
        time.sleep(5)
    log(f"  ERROR: Server not healthy after {timeout}s")
    return False


def kill_server(proc, logfile):
    # Kill entire process group (includes spawn'd workers)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except:
        pass
    logfile.close()
    # Also cleanup any stray vllm_omni processes
    os.system("pkill -9 -f 'vllm_omni.entrypoints.cli.main serve' 2>/dev/null")
    time.sleep(15)  # let GPU resources release


def generate_image(config, run_idx):
    w, h = config["res"].split("x")
    payload = {
        "model": config["model"],
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "n": 1,
        "size": f"{w}x{h}",
        "num_inference_steps": config["steps"],
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED + run_idx,
    }

    t0 = time.time()
    try:
        resp = requests.post(f"http://localhost:{PORT}/v1/images/generations",
                             json=payload, timeout=1800)
        elapsed = time.time() - t0
        data = resp.json()

        if "data" in data and data["data"][0].get("b64_json"):
            b64 = data["data"][0]["b64_json"]
            return {"success": True, "latency": elapsed, "image_size": len(b64)}
        else:
            error = json.dumps(data)[:300]
            return {"success": False, "latency": elapsed, "error": error}
    except Exception as e:
        elapsed = time.time() - t0
        return {"success": False, "latency": elapsed, "error": str(e)}


def run_config(config):
    log(f"{'='*60}")
    log(f"Config: {config['name']}")
    log(f"  Model: {config['model']}")
    log(f"  TP={config['tp']}, CFG={config['cfg']}, Res={config['res']}, Steps={config['steps']}")
    log(f"{'='*60}")

    proc, logfile, cmd_str = start_server(config)

    if not wait_healthy():
        log(f"  FAILED: Server did not start")
        kill_server(proc, logfile)
        # Save failure result
        result = {
            "config": config,
            "cmd": cmd_str,
            "status": "server_failed",
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE,
            "true_cfg_scale": TRUE_CFG_SCALE,
        }
        with open(f"{OUTDIR}/{config['name']}.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    # Warmup run (discard)
    log(f"  Warmup run...")
    warmup = generate_image(config, run_idx=99)
    log(f"  Warmup: success={warmup['success']}, latency={warmup['latency']:.2f}s")

    if not warmup["success"]:
        log(f"  FAILED at warmup: {warmup.get('error', 'unknown')}")
        kill_server(proc, logfile)
        result = {
            "config": config,
            "cmd": cmd_str,
            "status": "generation_failed",
            "error": warmup.get("error", "unknown"),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE,
            "true_cfg_scale": TRUE_CFG_SCALE,
        }
        with open(f"{OUTDIR}/{config['name']}.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    # Actual runs
    results = []
    for i in range(NUM_RUNS):
        r = generate_image(config, run_idx=i)
        results.append(r)
        status = "OK" if r["success"] else "FAIL"
        log(f"  Run {i+1}/{NUM_RUNS}: {status}, latency={r['latency']:.2f}s")

    latencies = [r["latency"] for r in results if r["success"]]
    kill_server(proc, logfile)

    # Save results
    output = {
        "config": config,
        "cmd": cmd_str,
        "status": "success" if latencies else "all_failed",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed_base": SEED,
        "num_runs": NUM_RUNS,
        "runs": results,
        "latencies": latencies,
        "median_latency": sorted(latencies)[len(latencies)//2] if latencies else None,
        "mean_latency": sum(latencies)/len(latencies) if latencies else None,
        "min_latency": min(latencies) if latencies else None,
        "max_latency": max(latencies) if latencies else None,
    }
    with open(f"{OUTDIR}/{config['name']}.json", "w") as f:
        json.dump(output, f, indent=2)

    if latencies:
        log(f"  Result: median={output['median_latency']:.2f}s, "
            f"mean={output['mean_latency']:.2f}s, "
            f"min={output['min_latency']:.2f}s, max={output['max_latency']:.2f}s")
    log("")


def main():
    log("=" * 60)
    log("Stage 4: CFG Parallel Benchmark")
    log(f"  Configs: {len(CONFIGS)}")
    log(f"  Runs per config: {NUM_RUNS}")
    log(f"  CFG scale: {TRUE_CFG_SCALE}")
    log(f"  Output: {OUTDIR}")
    log("=" * 60)
    log("")

    # Group configs by server (same model+tp+cfg can share a server)
    # But to keep it simple, restart server for each config
    # (different resolutions/steps are request-level params)

    # Optimize: group by server config (model + tp + cfg)
    server_groups = {}
    for c in CONFIGS:
        key = (c["model"], c["tp"], c["cfg"])
        if key not in server_groups:
            server_groups[key] = []
        server_groups[key].append(c)

    for (model, tp, cfg), configs in server_groups.items():
        # Start one server for this group
        first = configs[0]
        log(f"{'='*60}")
        log(f"Server group: model={os.path.basename(model)}, TP={tp}, CFG={cfg}")
        log(f"  Configs in group: {[c['name'] for c in configs]}")
        log(f"{'='*60}")

        proc, logfile, cmd_str = start_server(first)

        if not wait_healthy():
            log(f"  FAILED: Server did not start")
            kill_server(proc, logfile)
            for c in configs:
                result = {"config": c, "cmd": cmd_str, "status": "server_failed"}
                with open(f"{OUTDIR}/{c['name']}.json", "w") as f:
                    json.dump(result, f, indent=2)
            continue

        for config in configs:
            log(f"")
            log(f"--- Testing: {config['name']} ---")

            # Warmup
            log(f"  Warmup...")
            warmup = generate_image(config, run_idx=99)
            log(f"  Warmup: success={warmup['success']}, latency={warmup['latency']:.2f}s")

            if not warmup["success"]:
                log(f"  FAILED: {warmup.get('error', 'unknown')}")
                result = {
                    "config": config, "cmd": cmd_str,
                    "status": "generation_failed",
                    "error": warmup.get("error", "unknown"),
                    "prompt": PROMPT, "negative_prompt": NEGATIVE,
                    "true_cfg_scale": TRUE_CFG_SCALE,
                }
                with open(f"{OUTDIR}/{config['name']}.json", "w") as f:
                    json.dump(result, f, indent=2)
                continue

            # Runs
            results = []
            for i in range(NUM_RUNS):
                r = generate_image(config, run_idx=i)
                results.append(r)
                status = "OK" if r["success"] else "FAIL"
                log(f"  Run {i+1}/{NUM_RUNS}: {status}, latency={r['latency']:.2f}s")

            latencies = [r["latency"] for r in results if r["success"]]
            output = {
                "config": config, "cmd": cmd_str,
                "status": "success" if latencies else "all_failed",
                "prompt": PROMPT, "negative_prompt": NEGATIVE,
                "true_cfg_scale": TRUE_CFG_SCALE, "seed_base": SEED,
                "num_runs": NUM_RUNS, "runs": results, "latencies": latencies,
                "median_latency": sorted(latencies)[len(latencies)//2] if latencies else None,
                "mean_latency": sum(latencies)/len(latencies) if latencies else None,
                "min_latency": min(latencies) if latencies else None,
                "max_latency": max(latencies) if latencies else None,
            }
            with open(f"{OUTDIR}/{config['name']}.json", "w") as f:
                json.dump(output, f, indent=2)

            if latencies:
                log(f"  >> median={output['median_latency']:.2f}s, "
                    f"mean={output['mean_latency']:.2f}s")
            log("")

        kill_server(proc, logfile)

    log("=" * 60)
    log("Stage 4 Complete!")
    log(f"Results saved to: {OUTDIR}")
    log("=" * 60)

    # Print summary table
    log("")
    log("SUMMARY:")
    log(f"{'Config':<40} {'Status':<12} {'Median(s)':<10} {'Mean(s)':<10}")
    log("-" * 72)
    for c in CONFIGS:
        fpath = f"{OUTDIR}/{c['name']}.json"
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            status = d["status"]
            med = f"{d.get('median_latency', 0):.2f}" if d.get("median_latency") else "N/A"
            mean = f"{d.get('mean_latency', 0):.2f}" if d.get("mean_latency") else "N/A"
            log(f"{c['name']:<40} {status:<12} {med:<10} {mean:<10}")


if __name__ == "__main__":
    main()
