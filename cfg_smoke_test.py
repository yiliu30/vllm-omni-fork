#!/usr/bin/env python3
"""CFG Parallel smoke test: W4A16 TP=2/CFG=1 vs W4A16 TP=2/CFG=2.
Both use true_cfg_scale=4.0 + negative_prompt.
Config A runs both CFG branches sequentially on 2 GPUs.
Config B runs them in parallel across 4 GPUs (2 TP x 2 CFG).
"""
import subprocess, time, requests, base64, json, os, sys

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "/home/yiliu7/models/FLUX.1-dev-AutoRound-w4a16"
PYTHON = "/home/yiliu7/workspace/venvs/omni/bin/python"
OUTDIR = "/home/yiliu7/workspace/vllm-omni/cfg_smoke_images"
os.makedirs(OUTDIR, exist_ok=True)

CONFIGS = [
    {
        "name": "W4A16_TP2_CFG1",
        "port": 8100,
        "args": ["--tensor-parallel-size", "2"],
        "desc": "TP=2, sequential CFG (2 GPUs)",
    },
    {
        "name": "W4A16_TP2_CFG2",
        "port": 8101,
        "args": ["--tensor-parallel-size", "2", "--cfg-parallel-size", "2"],
        "desc": "TP=2, parallel CFG (4 GPUs = 2 TP x 2 CFG)",
    },
]

PROMPT = "A majestic snow-capped mountain reflected in a crystal clear lake at golden hour, photorealistic"
NEGATIVE = "low quality, blurry, distorted, watermark, text"


def start_server(config):
    cmd = [
        PYTHON, "-m", "vllm_omni.entrypoints.cli.main", "serve", MODEL,
        "--omni", "--port", str(config["port"]),
        "--disable-log-stats", "--stage-init-timeout", "900",
    ] + config["args"]
    logpath = f"{OUTDIR}/{config['name']}_server.log"
    logfile = open(logpath, "w")
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)
    print(f"  CMD: {' '.join(cmd)}", flush=True)
    return proc, logfile


def wait_healthy(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return time.time() - t0
        except:
            pass
        time.sleep(5)
    return -1


def generate_image(port, prompt, negative_prompt, steps=20, cfg_scale=4.0, seed=42):
    t0 = time.time()
    resp = requests.post(f"http://localhost:{port}/v1/images/generations", json={
        "model": MODEL,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "num_inference_steps": steps,
        "true_cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }, timeout=600)
    elapsed = time.time() - t0
    return resp.json(), elapsed


def kill_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    results = {}
    for config in CONFIGS:
        name = config["name"]
        port = config["port"]
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {name} - {config['desc']}", flush=True)
        print(f"{'='*60}", flush=True)

        proc, logfile = start_server(config)

        startup_time = wait_healthy(port, timeout=600)
        if startup_time < 0:
            print(f"  ERROR: Server did not become healthy in 10 min!", flush=True)
            kill_server(proc)
            logfile.close()
            results[name] = {"status": "TIMEOUT"}
            time.sleep(5)
            continue

        print(f"  Server healthy after {startup_time:.0f}s", flush=True)

        # Warmup (4 steps)
        print(f"  Warmup...", flush=True)
        try:
            resp, wt = generate_image(port, PROMPT, NEGATIVE, steps=4, seed=0)
            if "data" in resp:
                print(f"  Warmup OK ({wt:.1f}s)", flush=True)
            else:
                print(f"  Warmup FAILED: {json.dumps(resp)[:300]}", flush=True)
                kill_server(proc)
                logfile.close()
                results[name] = {"status": "WARMUP_FAILED", "resp": str(resp)[:200]}
                time.sleep(5)
                continue
        except Exception as e:
            print(f"  Warmup EXCEPTION: {e}", flush=True)
            kill_server(proc)
            logfile.close()
            results[name] = {"status": "EXCEPTION", "error": str(e)}
            time.sleep(5)
            continue

        # 3 timed runs
        print(f"  Running 3 generations (20 steps, 1024x1024, cfg_scale=4.0)...", flush=True)
        times = []
        for i in range(3):
            try:
                resp, elapsed = generate_image(port, PROMPT, NEGATIVE, steps=20, seed=42+i)
                if "data" in resp:
                    img_b64 = resp["data"][0].get("b64_json", "")
                    img_bytes = base64.b64decode(img_b64)
                    img_path = f"{OUTDIR}/{name}_run{i+1}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    times.append(elapsed)
                    print(f"    Run {i+1}: {elapsed:.2f}s ({len(img_bytes)/1024:.0f} KB) -> {img_path}", flush=True)
                else:
                    print(f"    Run {i+1}: FAILED - {json.dumps(resp)[:200]}", flush=True)
            except Exception as e:
                print(f"    Run {i+1}: EXCEPTION - {e}", flush=True)

        if times:
            avg = sum(times) / len(times)
            print(f"  AVG: {avg:.2f}s ({len(times)}/3 ok)", flush=True)
            results[name] = {"status": "OK", "times": times, "avg": avg}
        else:
            results[name] = {"status": "ALL_FAILED"}

        kill_server(proc)
        logfile.close()
        time.sleep(5)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        if r["status"] == "OK":
            print(f"  {name}: avg={r['avg']:.2f}s  runs={[f'{t:.2f}s' for t in r['times']]}", flush=True)
        else:
            print(f"  {name}: {r['status']}", flush=True)

    if all(r.get("status") == "OK" for r in results.values()):
        cfg1 = results["W4A16_TP2_CFG1"]["avg"]
        cfg2 = results["W4A16_TP2_CFG2"]["avg"]
        speedup = cfg1 / cfg2
        print(f"\n  CFG Parallel Speedup: {speedup:.2f}x", flush=True)
        print(f"    TP=2 sequential CFG: {cfg1:.2f}s (2 GPUs)", flush=True)
        print(f"    TP=2 parallel  CFG:  {cfg2:.2f}s (4 GPUs)", flush=True)

    with open(f"{OUTDIR}/smoke_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTDIR}/smoke_test_results.json", flush=True)


if __name__ == "__main__":
    main()
