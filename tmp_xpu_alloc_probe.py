"""Cheap, reversible XPU allocation probe.

Question it answers: is the dense-run OOM a per-process allocator ceiling
(reproducible -> our config to fix) or transient external contention on specific
physical GPUs (wait/retry)? Tries to allocate a growing buffer on each of the
4 logical devices under the SAME ZE_AFFINITY_MASK the dense run uses.
"""
import os
import torch

mask = os.environ.get("ZE_AFFINITY_MASK", "(unset)")
print(f"ZE_AFFINITY_MASK={mask}  -> logical xpu:i maps in mask order")
print(f"xpu available={torch.xpu.is_available()} device_count={torch.xpu.device_count()}")

# 256 MiB fp16 chunks; just probe whether the FIRST chunk lands (free vs held).
CHUNK_MIB = 256
MAX_MIB = 512
elems = CHUNK_MIB * 1024 * 1024 // 2  # fp16 = 2 bytes

for d in range(torch.xpu.device_count()):
    dev = f"xpu:{d}"
    held = []
    got = 0
    err = None
    try:
        while got < MAX_MIB:
            held.append(torch.empty(elems, dtype=torch.float16, device=dev))
            got += CHUNK_MIB
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    free_b, total_b = torch.xpu.mem_get_info(d)
    print(f"  {dev}: allocated_ok={got/1024:.2f} GiB  "
          f"mem_get_info(free={free_b/2**30:.2f} GiB, total={total_b/2**30:.2f} GiB)  "
          f"{'STOP@ ' + err if err else 'reached cap (no failure)'}")
    del held
    torch.xpu.empty_cache()

print("done")
