# Wan2.2 T2V A14B XPU Run Config

Successful end-to-end generation config for:

- Model: `/data/model/Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Platform: Intel XPU
- Attention backend: `TORCH_SDPA`
- Tensor parallel size: `4`
- CPU offload: enabled

## Command

```bash
sg render -c '
  ulimit -n 1048576
  cd /data/model/yiliu7/vllm-omni
  export SYCL_UR_USE_LEVEL_ZERO_V2=0
  export ZE_AFFINITY_MASK=0,1,2,3
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
  export UV_CACHE_DIR=/tmp/uvcache

  uv run --no-sync python examples/offline_inference/text_to_video/text_to_video.py \
    --model /data/model/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --num-inference-steps 40 \
    --boundary-ratio 0.875 \
    --flow-shift 5.0 \
    --fps 16 \
    --tensor-parallel-size 4 \
    --enable-cpu-offload \
    --vae-use-slicing \
    --vae-use-tiling \
    --enable-diffusion-pipeline-profiler \
    --output /tmp/t2v_480p_tp4_cpu_offload_sdpa.mp4 \
    --enforce-eager
'
```

## Effective Settings

- `SYCL_UR_USE_LEVEL_ZERO_V2=0`
- `ZE_AFFINITY_MASK=0,1,2,3`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`
- `UV_CACHE_DIR=/tmp/uvcache`
- `tensor_parallel_size=4`
- `cfg_parallel_size=1`
- `ulysses_degree=1`
- `ring_degree=1`
- `pipeline_parallel_size=1`
- `vae_patch_parallel_size=1`
- `enable_cpu_offload=True`
- `enable_layerwise_offload=False`
- `vae_use_slicing=True`
- `vae_use_tiling=True`
- `enforce_eager=True`

## Output

- Output file: `/tmp/t2v_480p_tp4_cpu_offload_sdpa.mp4`

## Observed Runtime

- Total generation time: `1306.34s` (`21m 46s`)
- Worker peak GPU memory reserved: `11402 MiB` (`11.13 GiB`)

## Notes

- `TORCH_SDPA` completed successfully.
- Default `FLASH_ATTN` failed with `v must be contiguous`.
- `SAGE_ATTN` originally failed because XPU SageAttention required `auto-round-lib`.
- After repairing the ARK import path, the remaining `SAGE_ATTN_XPU_BACKEND=sparse` failure was a kernel limitation in sparse preprocess, not a missing-package problem.

## SageAttention Sparse Attempt

Tried the same config with:

- `DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN`
- `SAGE_ATTN_XPU_BACKEND=sparse`

Command delta:

```bash
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
export SAGE_ATTN_XPU_BACKEND=sparse
```

Observed result on the first attempt:

- Backend resolved: `SAGE_ATTN`
- Model loading completed successfully on TP4 with CPU offload.
- Dummy warmup / first generation attempt failed before output export.

Initial failure:

```text
RuntimeError: Dummy run failed: Worker failed with error 'Current device xpu:0 is not supported'
```

Initial root cause:

- `vllm_omni/diffusion/attention/backends/sage_attn.py` prepends `AUTO_ROUND_KERNEL_PATH` to `sys.path`.
- In this environment it defaulted to the ARK source tree at `/home/yiliu7/workspace/auto-round/auto_round_extension/ark`.
- That source tree did not expose the built `auto_round_kernel_xpu` extension on import, so ARK's XPU sparse library handle stayed unset.
- The failure came from ARK import resolution, not from Wan2.2 itself.

### Retry With oneAPI setvars

Retried the same sparse SageAttention config after sourcing:

```bash
source /opt/intel/oneapi/setvars.sh
```

Observed result before the ARK path fix:

- oneAPI environment initialized successfully
- `SAGE_ATTN` still resolved correctly
- model loading still completed successfully
- dummy warmup still failed before output export with the same import/device error

### After Fixing ARK Import Resolution

After installing the XPU ARK package into `.venv` and pointing:

```bash
export AUTO_ROUND_KERNEL_PATH=/data/model/yiliu7/vllm-omni/.venv/lib/python3.13/site-packages
```

the `Current device xpu:0 is not supported` error was resolved.

Observed result after the fix:

- sparse SageAttention XPU kernels loaded successfully
- model loading completed successfully
- dummy warmup still failed, but now with a different error

Observed timings before the new failure:

- worker model loading: about `21.74s`
- dummy `Wan22Pipeline.diffuse`: about `32.46s`
- dummy `Wan22Pipeline.forward`: about `34.12s`

New failure:

```text
RuntimeError: Dummy run failed: Worker failed with error 'sparge_preprocess_topk currently supports prefill only: seq_len_q must equal seq_len_kv'
```

Root cause after the fix:

- Wan2.2 uses separate attention roles:
  - self-attention layers use role `self`
  - cross-attention layers use role `cross`
- The sparse ARK preprocess enforces `seq_len_q == seq_len_kv`.
- Wan2.2 cross-attention does not satisfy that contract because latent query length and text KV length differ.
- The failure is therefore a sparse-kernel contract mismatch on Wan2.2 cross-attention, not a general TP4/XPU/cpu-offload issue.

Relevant code references:

- `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py`
  - self-attention role: `role="self"`
  - cross-attention role: `role="cross"`
- `vllm_omni/diffusion/attention/backends/sage_attn.py`
  - XPU sparse path dispatches all selected `SAGE_ATTN` calls to the sparse backend
- `auto_round_kernel/sparse_attention.py`
  - `sparge_preprocess_topk` raises when `seq_len_q != seq_len_kv`

Verification:

- Using the same TP4 + CPU offload setup, a role-specific override with:
  - `self = SAGE_ATTN`
  - `cross = TORCH_SDPA`
- completed warmup successfully.
- Initialization completed in about `68.90s`.
- Dummy warmup timings:
  - `Wan22Pipeline.diffuse`: about `36.01s`
  - `Wan22Pipeline.vae.decode`: about `2.83s`
  - `Wan22Pipeline.forward`: about `40.62s`

Conclusion:

- `SAGE_ATTN_XPU_BACKEND=sparse` is usable for Wan2.2 self-attention after the ARK fix.
- It is not currently usable for Wan2.2 cross-attention because the sparse preprocess only supports prefill-style equal Q/KV sequence lengths.
- A practical workaround is per-role routing:
  - `self -> SAGE_ATTN`
  - `cross -> TORCH_SDPA`

## Environment-Specific Workaround

The original environment needed repair before this config was reusable directly from `.venv`:

- `triton-xpu==3.7.1`
- `vllm-xpu-kernels==0.1.3.1`
- a small compatibility shim for `XpuFusedMoe` was added to:
  `/.venv/lib/python3.13/site-packages/vllm_xpu_kernels/fused_moe_interface.py`

`PYTHONPATH=/tmp/triton-xpu-overlay` is no longer required after cleaning disk space and reinstalling the XPU runtime packages into `.venv`.

## oneAPI + Sparse Backend Environment

For the sparse SageAttention experiments that progressed past import issues, the effective environment was:

```bash
source /opt/intel/oneapi/setvars.sh
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export ZE_AFFINITY_MASK=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
export SAGE_ATTN_XPU_BACKEND=sparse
export AUTO_ROUND_KERNEL_PATH=/data/model/yiliu7/vllm-omni/.venv/lib/python3.13/site-packages
export UV_CACHE_DIR=/tmp/uvcache
ulimit -n 1048576
```

## SageAttention v1 Result

Using the same Wan2.2 TP4 + CPU-offload configuration with:

```bash
source /opt/intel/oneapi/setvars.sh
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export ZE_AFFINITY_MASK=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
export SAGE_ATTN_XPU_BACKEND=sagev1
export AUTO_ROUND_KERNEL_PATH=/data/model/yiliu7/vllm-omni/.venv/lib/python3.13/site-packages
export UV_CACHE_DIR=/tmp/uvcache
ulimit -n 1048576
```

the full run completed successfully.

Command:

```bash
sg render -c '
  source /opt/intel/oneapi/setvars.sh >/dev/null
  ulimit -n 1048576
  cd /data/model/yiliu7/vllm-omni
  export SYCL_UR_USE_LEVEL_ZERO_V2=0
  export ZE_AFFINITY_MASK=0,1,2,3
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
  export SAGE_ATTN_XPU_BACKEND=sagev1
  export AUTO_ROUND_KERNEL_PATH=/data/model/yiliu7/vllm-omni/.venv/lib/python3.13/site-packages
  export UV_CACHE_DIR=/tmp/uvcache

  uv run --no-sync python examples/offline_inference/text_to_video/text_to_video.py \
    --model /data/model/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --num-inference-steps 40 \
    --boundary-ratio 0.875 \
    --flow-shift 5.0 \
    --fps 16 \
    --tensor-parallel-size 4 \
    --enable-cpu-offload \
    --vae-use-slicing \
    --vae-use-tiling \
    --enable-diffusion-pipeline-profiler \
    --output /tmp/t2v_480p_tp4_cpu_offload_sagev1.mp4 \
    --enforce-eager
'
```

Observed result:

- Output file: `/tmp/t2v_480p_tp4_cpu_offload_sagev1.mp4`
- Total generation time: `1194.7563s` (`19m 54.76s`)
- Worker peak GPU memory reserved: `11722 MiB` (`11.45 GiB`)

Profiler:

- `Wan22Pipeline.diffuse`: about `1144.27s` to `1153.29s`
- `Wan22Pipeline.vae.decode`: about `26.06s` to `26.19s`
- `Wan22Pipeline.forward`: about `1194.10s` to `1194.23s`

Comparison against the earlier `TORCH_SDPA` run:

- `TORCH_SDPA`: `1306.34s`
- `SAGE_ATTN_XPU_BACKEND=sagev1`: `1194.76s`
- Improvement: about `111.59s` faster, roughly `8.5%`

## Sparse Retry Timing

Retried the full Wan2.2 TP4 + CPU-offload run with the repaired ARK import path and:

```bash
source /opt/intel/oneapi/setvars.sh
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export ZE_AFFINITY_MASK=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
export SAGE_ATTN_XPU_BACKEND=sparse
export AUTO_ROUND_KERNEL_PATH=/data/model/yiliu7/vllm-omni/.venv/lib/python3.13/site-packages
export UV_CACHE_DIR=/tmp/uvcache
ulimit -n 1048576
```

Observed result:

- No MP4 was produced.
- The run failed during dummy warmup, before the real 40-step generation request.
- Wall-clock process time: `1m27.696s`

Observed timings before failure:

- `diffusers_loader` weight loading: about `22.00s`
- worker model loading: about `35.08s` to `35.43s`
- dummy `Wan22Pipeline.diffuse`: about `31.41s` to `31.43s`
- dummy `Wan22Pipeline.forward`: about `33.05s`

Failure:

```text
Dummy run failed: Worker failed with error 'sparge_preprocess_topk currently supports prefill only: seq_len_q must equal seq_len_kv'
```

Interpretation:

- The repaired environment is sufficient for sparse kernel loading.
- The remaining blocker is still the sparse preprocess contract on Wan2.2 cross-attention.

## Mixed Attention Result

Retried Wan2.2 with per-role routing:

- self-attention: `SAGE_ATTN`
- cross-attention: `TORCH_SDPA`
- XPU SageAttention backend: `sparse`

This keeps sparse enabled where Wan2.2 satisfies the kernel contract and routes cross-attention away from the failing sparse preprocess path.

Command:

```bash
sg render -c '
  source /opt/intel/oneapi/setvars.sh >/dev/null
  ulimit -n 1048576
  cd /data/model/yiliu7/vllm-omni
  export SYCL_UR_USE_LEVEL_ZERO_V2=0
  export ZE_AFFINITY_MASK=0,1,2,3
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
  export SAGE_ATTN_XPU_BACKEND=sparse
  export AUTO_ROUND_KERNEL_PATH=/data/model/yiliu7/vllm-omni/.venv/lib/python3.13/site-packages
  export UV_CACHE_DIR=/tmp/uvcache

  uv run --no-sync python tmp_run_wan_per_role_attention.py \
    --model /data/model/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --num-inference-steps 40 \
    --boundary-ratio 0.875 \
    --flow-shift 5.0 \
    --fps 16 \
    --output /tmp/t2v_480p_tp4_cpu_offload_sparse_self_sdpa_cross.mp4
'
```

Observed result:

- Output file: `/tmp/t2v_480p_tp4_cpu_offload_sparse_self_sdpa_cross.mp4`
- Total generation time: `1153.1911s` (`19m 13.19s`)
- Worker peak GPU memory reserved: `11404 MiB` (`11.14 GiB`)
- Shell wall-clock time: about `20m27.145s`

Profiler:

- `Wan22Pipeline.diffuse`: about `1114.62s` to `1114.67s`
- `Wan22Pipeline.vae.decode`: about `26.13s` to `26.29s`
- `Wan22Pipeline.forward`: about `1152.54s` to `1152.67s`

Comparison:

- `TORCH_SDPA`: `1306.34s`
- `SAGE_ATTN_XPU_BACKEND=sagev1`: `1194.76s`
- `self=SAGE_ATTN(sparse), cross=TORCH_SDPA`: `1153.19s`

Relative improvement:

- vs `TORCH_SDPA`: about `153.15s` faster, roughly `11.7%`
- vs `sagev1`: about `41.57s` faster, roughly `3.5%`

Conclusion:

- Pure sparse backend is blocked by Wan2.2 cross-attention shape mismatch.
- Sparse self-attention with SDPA cross-attention is currently the fastest successful TP4 + CPU-offload configuration tested on this setup.

## Attention Call Timing Comparison

Added temporary attention-call timing instrumentation in:

- `vllm_omni/diffusion/attention/layer.py`

Measurement method:

- enabled with `DIFFUSION_ATTN_TIMING=1`
- each local attention call is timed with device synchronization before and after the backend call
- summaries are emitted once per worker process at shutdown as `ATTN_TIMING_SUMMARY`

What this timing is:

- synchronized wall time measured in Python around each attention backend call
- implementation shape:
  - `torch.xpu.synchronize()`
  - `time.perf_counter()`
  - run attention backend
  - `torch.xpu.synchronize()`
  - `time.perf_counter()`

What this timing includes:

- device kernel execution time
- launch overhead
- blocking and synchronization visible to the worker
- small Python timing overhead

What this timing is not:

- pure device-event timing
- per-kernel XPU profiler event duration
- a torch-profiler trace measurement

Important interpretation note:

- the attention totals below are the sum across all 4 TP workers
- they measure synchronized backend-call time inside worker processes
- they are useful for dense vs sparse comparison, but they are not the same as end-to-end wall-clock time

### Dense Baseline

Configuration:

- attention mode: all roles `TORCH_SDPA`
- TP4 + CPU offload

Observed result:

- total generation time: `1298.1191s`
- output: `/tmp/t2v_480p_tp4_cpu_offload_dense_attn_timing.mp4`
- log: `/tmp/wan22_dense_attn_timing.log`

Aggregated worker attention timing:

- self-attention
  - backend: `TORCH_SDPA`
  - total calls across workers: `13120`
  - total summed worker time: `1251937.334 ms`
  - average per call per worker: `95.422 ms`
- cross-attention
  - backend: `TORCH_SDPA`
  - total calls across workers: `13120`
  - total summed worker time: `23131.685 ms`
  - average per call per worker: `1.763 ms`

### Sparse Self-Attention `topk=0.1`

Configuration:

- self-attention: `SAGE_ATTN`
- cross-attention: `TORCH_SDPA`
- `SAGE_ATTN_XPU_BACKEND=sparse`
- `SAGE_ATTN_TOPK=0.1`
- TP4 + CPU offload

Observed result:

- total generation time: `1075.6684s`
- output: `/tmp/t2v_480p_tp4_cpu_offload_sparse_topk0p1_attn_timing.mp4`
- log: `/tmp/wan22_sparse_topk0p1_attn_timing.log`

Aggregated worker attention timing:

- self-attention
  - backend: `SAGE_ATTN`
  - total calls across workers: `13120`
  - total summed worker time: `425307.460 ms`
  - average per call per worker: `32.417 ms`
- cross-attention
  - backend: `TORCH_SDPA`
  - total calls across workers: `13120`
  - total summed worker time: `23315.364 ms`
  - average per call per worker: `1.777 ms`

### Comparison

- end-to-end generation
  - dense SDPA: `1298.1191s`
  - sparse self `topk=0.1`: `1075.6684s`
  - improvement: `222.4507s`
  - end-to-end speedup: `1.2068x`

- self-attention worker-time aggregate
  - dense SDPA: `1251937.334 ms`
  - sparse self `topk=0.1`: `425307.460 ms`
  - reduction: `826629.874 ms`
  - self-attention speedup: `2.9436x`

- cross-attention worker-time aggregate
  - dense SDPA: `23131.685 ms`
  - sparse self `topk=0.1`: `23315.364 ms`
  - essentially unchanged

Takeaway:

- the measured self-attention kernel/backend path got much faster, about `2.94x`
- cross-attention did not improve because it remained on `TORCH_SDPA`
- the full run improved by about `20.7%`, which is smaller than the self-attention speedup because end-to-end time still includes cross-attention, non-attention transformer work, VAE decode, text encoding, TP communication, and CPU-offload overhead

## Sparse Self-Attention `topk=0.5` With Attention Sink

Collected one additional attention-timing run with:

- self-attention: `SAGE_ATTN`
- cross-attention: `TORCH_SDPA`
- `SAGE_ATTN_XPU_BACKEND=sparse`
- `SAGE_ATTN_TOPK=0.5`
- `SAGE_ATTN_ATTENTION_SINK=1`
- TP4 + CPU offload

Observed result:

- total generation time: `1154.7674s`
- peak reserved memory: `11404 MiB`
- output: `/tmp/t2v_480p_tp4_cpu_offload_sparse_topk0p5_sink_attn_timing.mp4`
- log: `/tmp/wan22_sparse_topk0p5_sink_attn_timing.log`

Aggregated worker attention timing:

- self-attention
  - backend: `SAGE_ATTN`
  - total calls across workers: `13120`
  - total summed worker time: `740307.567 ms`
  - average per call per worker: `56.426 ms`
- cross-attention
  - backend: `TORCH_SDPA`
  - total calls across workers: `13120`
  - total summed worker time: `23294.693 ms`
  - average per call per worker: `1.776 ms`

Comparison:

- vs dense SDPA baseline
  - end-to-end speedup: `1.1241x`
  - end-to-end improvement: `143.3517s`
  - self-attention summed worker time speedup: `1.6911x`

- vs sparse self `topk=0.1`
  - end-to-end is slower by `79.0990s`
  - self-attention summed worker time is slower by `315000.107 ms`
  - cross-attention remains effectively unchanged

Summary across the measured runs:

- dense SDPA
  - total generation time: `1298.1191s`
  - self-attention summed worker time: `1251937.334 ms`
- sparse self `topk=0.1`
  - total generation time: `1075.6684s`
  - self-attention summed worker time: `425307.460 ms`
- sparse self `topk=0.5` with attention sink
  - total generation time: `1154.7674s`
  - self-attention summed worker time: `740307.567 ms`

Takeaway:

- enabling sparse self-attention with `topk=0.5` and `SAGE_ATTN_ATTENTION_SINK=1` still improves over dense SDPA
- in this Wan2.2 TP4 setup it is clearly worse than the measured `topk=0.1` run
- the main delta is in self-attention time; cross-attention remains flat because it still uses `TORCH_SDPA`

## Dense vs Sparse Comparison

Three-way comparison for the same Wan2.2 TP4 + CPU-offload setup:

| Mode | Self attention backend | Cross attention backend | Sparse config | Total generation time | Peak reserved memory | Self-attn worker-time aggregate | Cross-attn worker-time aggregate |
|---|---|---|---|---:|---:|---:|---:|
| Dense | `TORCH_SDPA` | `TORCH_SDPA` | none | `1298.1191s` | `11402 MiB` | `1251937.334 ms` | `23131.685 ms` |
| Sparse `topk=0.1` | `SAGE_ATTN` | `TORCH_SDPA` | `SAGE_ATTN_XPU_BACKEND=sparse`, `SAGE_ATTN_TOPK=0.1` | `1075.6684s` | `11404 MiB` | `425307.460 ms` | `23315.364 ms` |
| Sparse `topk=0.5` | `SAGE_ATTN` | `TORCH_SDPA` | `SAGE_ATTN_XPU_BACKEND=sparse`, `SAGE_ATTN_TOPK=0.5`, `SAGE_ATTN_ATTENTION_SINK=1` | `1154.7674s` | `11404 MiB` | `740307.567 ms` | `23294.693 ms` |

Relative summary:

- end-to-end runtime
  - sparse `topk=0.1` vs dense: `222.4507s` faster, `1.2068x`
  - sparse `topk=0.5` vs dense: `143.3517s` faster, `1.1241x`
  - sparse `topk=0.1` vs sparse `topk=0.5`: `79.0990s` faster, `1.0735x`

- self-attention worker-time aggregate
  - sparse `topk=0.1` vs dense: `826629.874 ms` less, `2.9436x`
  - sparse `topk=0.5` vs dense: `511629.767 ms` less, `1.6911x`
  - sparse `topk=0.1` vs sparse `topk=0.5`: `315000.107 ms` less, `1.7407x`

- cross-attention worker-time aggregate
  - effectively unchanged across all three runs because cross-attention stayed on `TORCH_SDPA`

## Quality Risk Findings

Moved to a standalone report:

- [WAN22_SPARSE_QUALITY_RISK_FINDINGS.md](/data/model/yiliu7/vllm-omni/WAN22_SPARSE_QUALITY_RISK_FINDINGS.md)

That file contains:

- quality-risk findings sorted by estimated e2e impact
- the current best hypothesis
- recommended validation order
