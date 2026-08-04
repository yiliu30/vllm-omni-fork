# CFG Parallel XPU All-Gather Hang

Date: 2026-08-03

## Summary

Cosmos3 inference with `tensor_parallel_size=2` and
`cfg_parallel_size=2` hung when switching from tensor-parallel (TP) XCCL
collectives to the classifier-free-guidance (CFG) XCCL all-gather. The cause
was cross-process-group device ordering, not
`VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` and not the CFG tensor contents.

XCCL collective calls return after work is enqueued. Ranks can therefore
arrive at Python code for the next collective while their current XPU streams
are at different points. With orthogonal TP and CFG groups, one CFG pair could
enter its device collective while another pair was still draining TP work.
The result was a cycle between communicators that only became visible when a
later XPU synchronization waited forever.

The fix keeps the CFG tensor gather entirely on XPU/XCCL. Gloo is used only
for zero-payload control barriers that define the transition between the TP
and CFG device phases.

## Topology

The failing four-rank topology is:

```text
TP groups:  [0, 1], [2, 3]
CFG groups: [0, 2], [1, 3]
```

The real denoising tensor is:

```text
shape=(1, 48, 48, 45, 80), dtype=torch.bfloat16
```

The smaller warmup tensor, `(1, 48, 1, 32, 32)`, usually completed. The hang
appeared once real model work created enough completion skew between TP groups.

## Root Cause

The original transition was effectively:

```text
enqueue TP collectives/kernels
return to Python
enter CFG all-gather on an orthogonal XCCL communicator
```

Returning from a PyTorch distributed call only established host arrival. It
did not establish completion of the previously enqueued device work. Under
Cosmos, the two TP groups differed by roughly 0.2 to 0.6 seconds at the CFG
boundary.

An initial attempted fix placed a host barrier before
`current_stream().synchronize()`. That was insufficient. After the barrier,
rank 0 completed its local stream synchronization and launched CFG work while
rank 2 was still draining TP work. One captured failure had rank 0 finish at
`997964.676` and rank 2 finish at `997965.215`, a 539 ms gap.

The host rendezvous must come after every rank has completed its local device
phase. A second rendezvous is required after CFG device completion so a faster
CFG pair cannot re-enter TP while the other CFG pair is still finishing CFG.

## Fix

For the XPU `separate_tensors=True` CFG gather, the implementation now uses:

```text
1. Synchronize the current XPU stream to drain prior TP/SP/PP work locally.
2. Barrier across all model-parallel ranks using a Gloo control group.
3. Run all_gather_into_tensor on the CFG XPU/XCCL group.
4. Synchronize the current XPU stream to force gather completion.
5. Barrier across all model-parallel ranks before returning to TP work.
```

The rendezvous group covers `tp-sp-pp-cfg` ranks within one data-parallel
replica. Data-parallel replicas remain independent. No prediction tensor is
copied to CPU and no tensor payload is sent through Gloo.
The fenced path is enabled only when CFG crosses at least one TP, SP, or PP
device group; CFG-only configurations retain the normal gather path.

Implementation:

- `vllm_omni/diffusion/distributed/group_coordinator.py`
- `vllm_omni/diffusion/distributed/parallel_state.py`

Tracing can be enabled with:

```bash
VLLM_OMNI_CFG_GATHER_TRACE=1
```

## Standalone Reproducer

The reproducer models the real orthogonal TP/CFG topology and can inject skew
into one TP group:

- `tools/repro/repro_xpu_cfg_allgather_hang.py`
- `tools/repro/run_xpu_cfg_allgather_repro.sh`

Without a model-wide phase fence, this command timed out after Python
collective calls returned and device completion stalled:

```bash
MODE=list ITERS=1 TP_PRECOLLECTIVES=8 \
TP_SLOW_RANKS=0,2 TP_DELAY_MS=500 \
CFG_HOST_RENDEZVOUS=0 \
bash tools/repro/run_xpu_cfg_allgather_repro.sh
```

The corrected ordering completes with the full-size tensor and deliberate
four-second TP skew:

```bash
MODE=into_tensor ITERS=1 TP_PRECOLLECTIVES=8 \
TP_SLOW_RANKS=0,2 TP_DELAY_MS=500 \
SYNC_BEFORE_GATHER=1 MODEL_HOST_RENDEZVOUS=1 \
bash tools/repro/run_xpu_cfg_allgather_repro.sh
```

All ranks completed the gather in approximately 1 ms with matching checksum
`126.497086`.

## End-to-End Validation

Environment:

```text
Container: vllm-xpu-clean
Accelerators: 4 x Intel Arc Pro B60
PyTorch: 2.12 with built-in XPU/XCCL
Intel GPU driver: 1.15.38308+1
oneCCL: 2021.17.2
```

One-step and four-step W4A16 Cosmos3 runs both completed using TP2 and CFG2:

```bash
NUM_INFERENCE_STEPS=4 VLLM_OMNI_CFG_GATHER_TRACE=1 \
bash run-cosmos-default.sh
```

The four-step run completed all denoising, decode, video output, and worker
shutdown with exit code 0:

```text
Pipeline time: 109.53 s
Total generation time: 235.4435 s
Output: /home/yiliu4/workspace/generated_videos/cosmos3_nano/
  cosmos3_nano_bf16_sparse_topk0p5_qtile256_qblk256_exampleprompt_negprompt_defaultfs_20260803T155911Z.mp4
Log: /home/yiliu4/workspace/generated_videos/cosmos3_nano/
  cosmos3_nano_bf16_sparse_topk0p5_qtile256_qblk256_exampleprompt_negprompt_defaultfs_20260803T155911Z.log
```

Observed control-barrier waits were approximately 0.2 to 0.6 seconds per
denoising step, reflecting real TP completion skew. Once all ranks arrived,
the XCCL gather, device completion, and release barrier took approximately
3 to 5 ms.

Focused tests also passed:

```text
102 passed, 1 skipped
```

## Performance Impact

This fix introduces explicit phase boundaries, so it can reduce overlap that
previously appeared possible. That overlap was not valid with the orthogonal
XCCL communicator ordering and was the source of the deadlock. The measured
wait is mostly existing TP imbalance exposed at the boundary; the gather and
control overhead after all ranks arrive is only a few milliseconds.

This is materially different from the rejected CPU/Gloo payload fallback,
which copied the full prediction tensor XPU-to-CPU, performed the gather on
Gloo, and copied the results CPU-to-XPU. The implemented path keeps the large
tensor on XPU and uses Gloo only to exchange barrier control messages.

## Rejected Hypotheses and Experiments

- Increasing `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` did not address the device
  deadlock.
- vLLM-style tensor-list `dist.all_gather` also stalled at later device
  completion; changing the Python collective API was not sufficient.
- Chunking the gather into 1 MiB pieces did not solve communicator ordering.
  CFG group `[1, 3]` completed while `[0, 2]` remained stuck.
- XPU broadcast of the large tensor hung in the standalone reproducer.
- A CFG-pair-only host rendezvous could not coordinate the two orthogonal CFG
  pairs.
- A model-wide barrier before local stream synchronization still allowed ranks
  to enter the CFG device phase at different times.
- CPU/Gloo tensor gathering worked as a correctness experiment but was rejected
  as the production solution due to data movement and synchronization cost.
