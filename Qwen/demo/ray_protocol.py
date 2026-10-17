"""Versioned synchronous rollout and weight-transfer contracts (no model imports)."""

import hashlib
import json
import random


class WeightVersion:
    def __init__(self):
        self.ready = -1
        self.pending = None
        self.expected = []
        self.receipts = []

    def begin(self, version, chunks):
        if self.pending is not None or version <= self.ready or not chunks:
            raise ValueError("Invalid or stale weight transaction.")
        self.pending, self.expected, self.receipts = version, chunks, []

    def accept(self, version, index, names, checksum):
        if (version != self.pending or index != len(self.receipts) or index >= len(self.expected)
                or names != self.expected[index]):
            raise ValueError("Out-of-order, incomplete or wrong-version weight chunk.")
        self.receipts.append(checksum)

    def commit(self, version, checksums):
        if version != self.pending or len(self.receipts) != len(self.expected) or self.receipts != checksums:
            raise ValueError("Weight barrier cannot commit: receipts differ.")
        self.ready, self.pending = version, None

    def require(self, version):
        if self.pending is not None or self.ready != version:
            raise ValueError(f"Stale rollout: required {version}, ready {self.ready}, pending {self.pending}")


def tensor_checksum(items):
    """Hash names, shapes, dtypes and every transmitted byte, including BF16 tensors."""
    import torch
    digest = hashlib.sha256()
    for name, value in items:
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_layout(gpus, stage):
    """Use distinct GPUs for role replicas; retain all roles when debugging on one GPU."""
    if gpus < 1:
        raise ValueError("At least one GPU is required.")
    if gpus == 1:
        roles = [("trainer", 0, 0, 0.34), ("rollout", 0, 0, 0.33), ("scorer", 0, 0, 0.33)]
    elif gpus == 2:
        roles = [("trainer", 0, 0, 1.0), ("rollout", 0, 1, 0.5), ("scorer", 0, 1, 0.5)]
    else:
        trainers = max(1, gpus // 2)
        rollouts = (gpus - trainers + 1) // 2
        roles = [("trainer", i, i, 1.0) for i in range(trainers)]
        roles += [("rollout", i, trainers + i, 1.0) for i in range(rollouts)]
        roles += [("scorer", i, trainers + rollouts + i, 1.0)
                  for i in range(gpus - trainers - rollouts)]
    if stage == "grpo":
        roles += [("reward", rank, slot, 0.0) for role, rank, slot, _ in list(roles) if role == "scorer"]
    return [{"role": role, "rank": rank, "slot": slot, "gpu": gpu} for role, rank, slot, gpu in roles]


def batch_indices(size, step, count):
    """Deterministic sampling with no dependence on actor scheduling or prior failures."""
    if count > size:
        raise ValueError("Dataset is smaller than the unique input batch.")
    return random.Random(42 + step).sample(range(size), count)


def validate_jobs(jobs, version, total):
    if len(jobs) != total or [j["index"] for j in jobs] != list(range(total)):
        raise ValueError("Missing, duplicate or reordered rollout records.")
    for job in jobs:
        if job["version"] != version or not job["tokens"]:
            raise ValueError("Wrong-version or empty completion.")


def verify_signal(job, signal):
    for key in ("index", "version", "tokens"):
        if signal[key] != job[key]:
            raise ValueError(f"Teacher/reference signal does not match rollout: {key}")
