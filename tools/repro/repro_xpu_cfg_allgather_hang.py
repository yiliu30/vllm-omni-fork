#!/usr/bin/env python3
"""Standalone PyTorch reproducer for XPU CFG all-gather completion hangs.

Run inside an XPU/oneCCL environment, for example:

  torchrun --standalone --nproc_per_node=4 \
    tools/repro/repro_xpu_cfg_allgather_hang.py --mode into_tensor

The vLLM-Omni hang was observed with world_size=4, tensor_parallel_size=2,
cfg_parallel_size=2, and strided CFG groups [0, 2] and [1, 3]. The Python
collective call returned, but the first torch.xpu.synchronize() after the
collective did not complete on one CFG pair.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def log(rank: int, message: str) -> None:
    now = dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{now} rank={rank}: {message}", flush=True)


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from exc
    if not shape:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    return shape


def dtype_from_name(name: str) -> torch.dtype:
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return dtypes[name]


def build_cfg_groups(world_size: int, cfg_parallel_size: int, layout: str) -> list[list[int]]:
    if world_size % cfg_parallel_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by cfg_parallel_size={cfg_parallel_size}")

    num_groups = world_size // cfg_parallel_size
    if layout == "strided":
        return [[base + i * num_groups for i in range(cfg_parallel_size)] for base in range(num_groups)]
    if layout == "contiguous":
        return [
            list(range(base, base + cfg_parallel_size))
            for base in range(0, world_size, cfg_parallel_size)
        ]
    raise ValueError(f"unknown layout {layout!r}")


def build_contiguous_groups(world_size: int, group_size: int) -> list[list[int]]:
    if world_size % group_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by group_size={group_size}")
    return [list(range(base, base + group_size)) for base in range(0, world_size, group_size)]


def make_input(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    noncontiguous: bool,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim

    base = torch.arange(numel, dtype=torch.float32, device=device).reshape(shape)
    tensor = (base.remainder(251) + rank).to(dtype)
    if not noncontiguous:
        return tensor.contiguous()

    # Keep the final logical shape but make strides non-contiguous, matching the
    # Cosmos CFG tensor before GroupCoordinator calls input_.contiguous().
    expanded = torch.empty(shape[:-1] + (shape[-1] * 2,), dtype=dtype, device=device)
    expanded[..., ::2] = tensor
    return expanded[..., ::2]


def add_producer_work(x: torch.Tensor, mode: str, iters: int) -> torch.Tensor:
    if mode == "none" or iters <= 0:
        return x

    if mode == "elementwise":
        y = x
        for _ in range(iters):
            y = (y.float().mul(1.0001).add_(0.01).sin_()).to(x.dtype)
        return y

    if mode == "matmul":
        # Produce the final tensor through queued GEMMs, then broadcast a small
        # derived value across the target shape. This stresses stream dependency
        # handling without requiring a huge matrix allocation.
        hidden = min(4096, max(512, x.shape[-1] * 16))
        a = torch.randn((hidden, hidden), dtype=x.dtype, device=x.device)
        b = torch.randn((hidden, hidden), dtype=x.dtype, device=x.device)
        y = a
        for _ in range(iters):
            y = y @ b
            y = y / y.float().abs().mean().clamp_min(1.0).to(x.dtype)
        return x + y.flatten()[0].to(x.dtype)

    raise ValueError(f"unknown producer work mode {mode!r}")


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="xccl", help="distributed backend for device collectives")
    parser.add_argument("--device", default="xpu", choices=["xpu", "cuda", "cpu"])
    parser.add_argument("--mode", default="into_tensor", choices=["into_tensor", "list", "cpu_gloo"])
    parser.add_argument("--layout", default="strided", choices=["strided", "contiguous"])
    parser.add_argument("--cfg-parallel-size", type=int, default=2)
    parser.add_argument("--tp-parallel-size", type=int, default=2)
    parser.add_argument("--tp-precollectives", type=int, default=0)
    parser.add_argument(
        "--tp-slow-ranks",
        default="",
        help="comma-separated global ranks delayed between TP collectives",
    )
    parser.add_argument("--tp-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--cfg-host-rendezvous",
        action="store_true",
        help="rendezvous on the CFG Gloo group before launching XPU all-gather",
    )
    parser.add_argument(
        "--model-host-rendezvous",
        action="store_true",
        help="fence the XPU collective with all-model-rank Gloo barriers",
    )
    parser.add_argument("--shape", type=parse_shape, default=parse_shape("1,48,48,45,80"))
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup-shape", type=parse_shape, default=parse_shape("1,48,1,32,32"))
    parser.add_argument("--non-contiguous", action="store_true")
    parser.add_argument("--producer-work", default="none", choices=["none", "elementwise", "matmul"])
    parser.add_argument("--producer-iters", type=int, default=0)
    parser.add_argument("--sync-before-gather", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=300)
    args = parser.parse_args()
    slow_ranks = {int(value) for value in args.tp_slow_ranks.split(",") if value.strip()}

    if args.backend == "ccl":
        try:
            import oneccl_bindings_for_pytorch  # noqa: F401
        except ImportError as exc:
            print(f"Failed to import oneccl_bindings_for_pytorch: {exc}", file=sys.stderr, flush=True)
            return 2

    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_sec))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("torch.xpu is not available")
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    tp_groups = build_contiguous_groups(world_size, args.tp_parallel_size)
    my_tp_group = None
    my_tp_ranks = None
    for ranks in tp_groups:
        group = dist.new_group(ranks=ranks, backend=args.backend)
        if rank in ranks:
            my_tp_group = group
            my_tp_ranks = ranks

    cfg_groups = build_cfg_groups(world_size, args.cfg_parallel_size, args.layout)
    my_group = None
    my_cpu_group = None
    my_group_ranks = None
    for ranks in cfg_groups:
        group = dist.new_group(ranks=ranks, backend=args.backend)
        cpu_group = dist.new_group(ranks=ranks, backend="gloo")
        if rank in ranks:
            my_group = group
            my_cpu_group = cpu_group
            my_group_ranks = ranks

    model_cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    assert my_group is not None
    assert my_cpu_group is not None
    assert my_group_ranks is not None
    assert my_tp_group is not None
    assert my_tp_ranks is not None
    group_world = dist.get_world_size(my_group)
    dtype = dtype_from_name(args.dtype)

    log(
        rank,
        "initialized "
        f"backend={args.backend} device={device} mode={args.mode} "
        f"layout={args.layout} tp_group={my_tp_ranks} cfg_group={my_group_ranks} "
        f"shape={args.shape} dtype={dtype}",
    )

    def run_one(shape: tuple[int, ...], label: str) -> None:
        x = make_input(shape, dtype, device, rank, args.non_contiguous)
        x = add_producer_work(x, args.producer_work, args.producer_iters)
        if args.tp_precollectives > 0:
            tp_tensor = x.contiguous()
            for _ in range(args.tp_precollectives):
                dist.all_reduce(tp_tensor, op=dist.ReduceOp.SUM, group=my_tp_group)
                tp_tensor = (tp_tensor / float(args.tp_parallel_size)).to(dtype)
                if rank in slow_ranks and args.tp_delay_ms > 0:
                    time.sleep(args.tp_delay_ms / 1000.0)
            x = tp_tensor
        log(rank, f"{label}: input shape={tuple(x.shape)} contiguous={x.is_contiguous()}")
        if args.sync_before_gather:
            sync_start = time.monotonic()
            synchronize(device)
            log(rank, f"{label}: pre-gather device synchronize returned after {time.monotonic() - sync_start:.3f}s")

        if args.model_host_rendezvous:
            rendezvous_start = time.monotonic()
            dist.barrier(group=model_cpu_group)
            log(
                rank,
                f"{label}: model host rendezvous returned after "
                f"{time.monotonic() - rendezvous_start:.3f}s",
            )
        elif args.cfg_host_rendezvous:
            rendezvous_start = time.monotonic()
            dist.barrier(group=my_cpu_group)
            log(
                rank,
                f"{label}: CFG host rendezvous returned after "
                f"{time.monotonic() - rendezvous_start:.3f}s",
            )
        start = time.monotonic()
        if args.mode == "into_tensor":
            out_shape = list(x.shape)
            out_shape[0] *= group_world
            out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(out, x.contiguous(), group=my_group)
            gathered = out.view(group_world, *x.shape)
        elif args.mode == "list":
            x_contig = x.contiguous()
            tensor_list = [torch.empty_like(x_contig) for _ in range(group_world)]
            dist.all_gather(tensor_list, x_contig, group=my_group)
            gathered = torch.stack(tensor_list)
        else:
            x_cpu = x.contiguous().cpu()
            tensor_list_cpu = [torch.empty_like(x_cpu) for _ in range(group_world)]
            dist.all_gather(tensor_list_cpu, x_cpu, group=my_cpu_group)
            gathered = torch.stack([tensor.to(device) for tensor in tensor_list_cpu])

        log(rank, f"{label}: collective returned after {time.monotonic() - start:.3f}s")
        sync_start = time.monotonic()
        synchronize(device)
        log(rank, f"{label}: device synchronize returned after {time.monotonic() - sync_start:.3f}s")
        if args.model_host_rendezvous:
            release_start = time.monotonic()
            dist.barrier(group=model_cpu_group)
            log(
                rank,
                f"{label}: model host release returned after "
                f"{time.monotonic() - release_start:.3f}s",
            )

        # Touch the result so incorrect or incomplete data is visible without a
        # huge print. This also forces any deferred copy in cpu_gloo mode.
        checksum = gathered.float().mean().item()
        log(rank, f"{label}: checksum={checksum:.6f}")
        dist.barrier()

    try:
        run_one(args.warmup_shape, "warmup")
        for idx in range(args.iters):
            run_one(args.shape, f"large[{idx}]")
        log(rank, "done")
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
