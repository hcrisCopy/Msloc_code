"""DDP execution for official ActionFormer's list of complete video features.

Unlike a generic DataParallel scatter, this never splits
a video's [C,T] tensor along C. Model, losses and optimizer remain upstream's.
"""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def parse_device_ids(value):
    """Accept comma-separated CLI IDs or a YAML list, retaining their order."""
    try:
        values = value.split(",") if isinstance(value, str) else value
        ids = [int(str(item).strip()) for item in values]
    except (TypeError, ValueError) as error:
        raise ValueError("--device-ids must be comma-separated integers, e.g. 0,1,2,3") from error
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError("--device-ids must contain nonnegative, unique GPU IDs")
    return ids


def validate_cuda_devices(device_ids):
    if not torch.cuda.is_available() or max(device_ids) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device IDs {device_ids} unavailable; visible GPU count={torch.cuda.device_count()}")


def assigned_device(device, device_ids, local_rank, distributed):
    target = torch.device(device)
    if device_ids:
        return torch.device("cuda", device_ids[local_rank if distributed else 0])
    if target.type == "cuda" and distributed:
        return torch.device("cuda", local_rank)
    return target


class DistributedContext:
    def __init__(self, device, device_ids=None):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        if device_ids and len(device_ids) != self.world_size:
            raise ValueError("--device-ids count must match the number of training processes")
        self.device = assigned_device(device, device_ids, self.local_rank, self.distributed)
        if self.device.type == "cuda":
            index = self.device.index or 0
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {index} unavailable; check --device-ids")
            self.device = torch.device("cuda", index)
            torch.cuda.set_device(self.device)
        elif self.device.type != "cpu":
            raise ValueError("Training supports CUDA or CPU devices")
        if self.distributed:
            # FileStore is also useful for local CPU process tests on Windows.
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo",
                init_method=os.environ.get("MSLOC_AF_DIST_INIT_METHOD", "env://"),
                rank=self.rank, world_size=self.world_size)

    @property
    def primary(self):
        return self.rank == 0

    def print(self, *args, **kwargs):
        if self.primary:
            print(*args, **kwargs)

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def wrap(self, model):
        if not self.distributed:
            return model
        return DistributedDataParallel(
            model, device_ids=[self.device.index] if self.device.type == "cuda" else None,
            broadcast_buffers=False)

    def mean_losses(self, losses):
        values = torch.stack([value.detach() for value in losses.values()])
        if self.distributed:
            dist.all_reduce(values)
            values /= self.world_size
        return dict(zip(losses, values.tolist()))

    def require_finite(self, value, message):
        finite = torch.isfinite(value.detach()).all().to(device=self.device, dtype=torch.int32)
        if self.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(message)

    def gather_states(self, local_state):
        if not self.distributed:
            return [local_state]
        states = [None] * self.world_size if self.primary else None
        dist.gather_object(local_state, states, dst=0)
        return states

    def broadcast(self, value):
        values = [value]
        if self.distributed:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    def primary_call(self, function):
        """Run shared-file operations once and propagate failures to every rank."""
        result, failure = None, None
        if self.primary:
            try:
                result = function()
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
        result, failure = self.broadcast((result, failure))
        if failure is not None:
            raise RuntimeError(failure)
        return result

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def epoch_video_order(length, global_batch_size, world_size, rank, seed, completed_batches=0):
    """Disjoint full-video shards; drop only the final incomplete GLOBAL batch.

    Consecutive global batches match a deterministic single-process permutation.
    All ranks have the same batch count, including when resuming mid-epoch.
    """
    if global_batch_size < world_size or global_batch_size % world_size:
        raise ValueError("--batch-size is GLOBAL: it must be >= GPU count and divisible by GPU count")
    if not 0 <= rank < world_size:
        raise ValueError("Invalid distributed rank")
    total_batches = length // global_batch_size
    if total_batches == 0:
        raise ValueError("Official drop_last=True leaves no global training batch; lower --batch-size")
    if not 0 <= completed_batches <= total_batches:
        raise ValueError("Invalid resumed batch cursor")
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    start = completed_batches * global_batch_size
    return order[start:total_batches * global_batch_size][rank::world_size], total_batches
