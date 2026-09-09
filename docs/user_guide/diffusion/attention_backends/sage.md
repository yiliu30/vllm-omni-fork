# SageAttention

SageAttention backends provide lossy low-precision attention for diffusion
models. Validate output quality against `TORCH_SDPA` at the same seed before
using either backend in production.

## `SAGE_ATTN`

`SAGE_ATTN` uses SageAttention 2.2 with INT8-quantized attention and FP16
accumulation.

### Installation

Install SageAttention into the same environment as vLLM-Omni:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
pip install . --no-build-isolation
```

Verify the installation:

```bash
python -c "import sageattention; print(sageattention.__file__)"
```

Select it globally:

```bash
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN
```

## `SAGE_ATTN_3`

`SAGE_ATTN_3` uses the SageAttention3 Blackwell implementation.

### SageAttention3 installation

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention/sageattention3_blackwell
python setup.py install
```

Verify the installation:

```bash
python -c "import sageattn3; print(sageattn3.__file__)"
```

```bash
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN_3
```

On CUDA, `SAGE_ATTN_3` uses the SageAttention3 Blackwell kernel and requires
an importable `sageattn3` plus a Blackwell-class GPU.

### XPU (Sage V3 Hybrid via deepklox)

On Intel XPU, `SAGE_ATTN_3` uses the DeepKloX Sage V3 Hybrid kernel (MXFP4
Q/K + MXFP8 E4M3 P/V). The kernel is available through the `deepklox`
package, which the CRI image builds with the Sage V3 Hybrid kernel enabled
(`DEEPKLOX_SAGEATTN_V3_HYBRID=1`) and bakes into the environment before
vLLM-Omni is installed.

Verify the installation:

```bash
python -c "from deepklox.sageattn_interface import sageattn_v3_hybrid; print('deepklox v3 hybrid OK')"
```

The kernel contract is: CRI (Xe3P) hardware only, contiguous BF16 HND
tensors, `batch=1`, equal query/key-value head counts, `head_dim=128`, and
non-causal self-attention. Calls outside the contract (GQA/MQA, causal
attention, `batch>1`, other head sizes, cross-attention) fall back to
PyTorch SDPA with a one-time warning.

Environment variables (read by the backend module):

- `SAGE_ATTN_FORCE_SDPA_BLOCKS` — comma-separated transformer layer indices
  that must run plain SDPA instead of the quantized kernel (selective
  fallback).
- `SAGE_ATTN_REPORT_FALLBACKS=1` — print kernel/fallback dispatch counters
  (`sage`, `forced_sdpa`, `cross_sdpa`, `sdpa_fallback`, `nonfinite_sdpa`)
  at process exit. Counters can also be inspected with
  `vllm_omni.diffusion.attention.backends.sage_attn3.get_sage_attn3_call_counts()`.
- `SAGE_ATTN_DEBUG_CHECK_FINITE=1` — after each kernel call, re-run the call
  with SDPA if the output contains non-finite values.

For common configuration and platform routing, see the
[attention backend overview](../attention_backends.md).
