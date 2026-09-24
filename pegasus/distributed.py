"""Small native-PyTorch DDP helpers for standalone Koopman training."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(requested_device: str = "cuda") -> DistributedContext:
    """Initialize from ``torchrun`` environment variables when WORLD_SIZE>1."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    requested = torch.device(requested_device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if distributed:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} visible CUDA devices"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = requested if requested.index is not None else torch.device("cuda", 0)
            torch.cuda.set_device(device)
        backend = "nccl" if distributed else None
    else:
        device = requested
        backend = "gloo" if distributed else None

    if distributed and not dist.is_initialized():
        init_kwargs: dict[str, Any] = {"backend": backend, "init_method": "env://"}
        # Bind NCCL to this rank's GPU. Omitting device_id makes barriers warn
        # "device used by this process is currently unknown" and can hang.
        if device.type == "cuda":
            init_kwargs["device_id"] = device
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            # Older PyTorch without device_id support.
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)
    return DistributedContext(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def barrier(ctx: DistributedContext) -> None:
    if not ctx.distributed:
        return
    if ctx.device.type == "cuda":
        dist.barrier(device_ids=[ctx.local_rank])
    else:
        dist.barrier()


def broadcast_object(value: Any, ctx: DistributedContext, *, src: int = 0) -> Any:
    if not ctx.distributed:
        return value
    # Ensure this rank's CUDA device is current before object collectives so
    # NCCL does not pick an unbound/default device.
    if ctx.device.type == "cuda":
        torch.cuda.set_device(ctx.local_rank)
    payload = [value if ctx.rank == src else None]
    kwargs: dict[str, Any] = {"src": src}
    if ctx.device.type == "cuda":
        kwargs["device"] = ctx.device
    dist.broadcast_object_list(payload, **kwargs)
    return payload[0]


def reduce_metric_sums(
    sums: Mapping[str, float],
    steps: int,
    ctx: DistributedContext,
) -> tuple[dict[str, float], int]:
    """Sum epoch metric accumulators and step counts over all ranks."""
    keys = sorted(sums)
    device = ctx.device
    values = [float(sums[k]) for k in keys] + [float(steps)]
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if ctx.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    out = {key: float(tensor[i].item()) for i, key in enumerate(keys)}
    global_steps = int(round(float(tensor[-1].item())))
    return out, global_steps


def max_float(value: float, ctx: DistributedContext) -> float:
    tensor = torch.tensor(float(value), device=ctx.device, dtype=torch.float64)
    if ctx.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "DistributedContext",
    "initialize_distributed",
    "barrier",
    "broadcast_object",
    "reduce_metric_sums",
    "max_float",
    "cleanup_distributed",
]
