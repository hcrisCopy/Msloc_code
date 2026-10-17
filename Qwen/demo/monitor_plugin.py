"""Observe official trainers without replacing their optimization algorithms.

Phase timers are inclusive and synchronized; never sum nested phase durations.
"""

import functools
import json
import os
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback

from demo_utils import check_versions


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def emit(value):
    output = Path(os.environ["MSLOC_DEMO_RUN"])
    rank = int(os.environ.get("RANK", "0"))
    with (output / f"events_rank_{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def probe(model):
    values = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            values.append((name, parameter.detach().reshape(-1)[:32].float().cpu().tolist()))
            if len(values) == 8:
                break
    if not values:
        raise RuntimeError("No trainable parameters.")
    return values


class DemoMonitor(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        sync()
        self.last_end = self.train_begin = time.perf_counter()
        self.previous = probe(model)
        emit({"event": "session", "step": state.global_step,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in model.parameters())})

    def on_step_begin(self, args, state, control, **kwargs):
        sync()
        torch.cuda.reset_peak_memory_stats()
        self.begin = time.perf_counter()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        sync()
        now = time.perf_counter()
        elapsed = now - self.last_end
        update_elapsed = now - self.begin
        self.last_end = now
        current = probe(model)
        free, total = torch.cuda.mem_get_info()
        emit({"event": "step", "step": state.global_step, "seconds": elapsed,
              "update_seconds": update_elapsed, "device_used_gib": (total - free) / 2**30,
              "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "parameter_probe_changed": current != self.previous})
        self.previous = current
        if state.is_world_process_zero:
            print(f"\n[Demo] step={state.global_step} time={elapsed:.3f}s "
                  f"allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        sync()
        self.optimizer_begin = time.perf_counter()

    def on_optimizer_step(self, args, state, control, **kwargs):
        sync()
        emit({"event": "phase", "step": state.global_step + 1, "phase": "optimizer",
              "seconds": time.perf_counter() - self.optimizer_begin})

    def on_train_end(self, args, state, control, **kwargs):
        sync()
        emit({"event": "train_end", "step": state.global_step,
              "seconds": time.perf_counter() - self.train_begin})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            emit({"event": "metrics", "step": state.global_step, "values": logs})


def wrap(cls, method, label, check_sync=False):
    original = getattr(cls, method)

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        sync()
        begin = time.perf_counter()
        result = original(self, *args, **kwargs)
        sync()
        emit({"event": "phase", "step": self.state.global_step + 1,
              "phase": label, "seconds": time.perf_counter() - begin})
        if check_sync and self.use_fast_infer and self._last_loaded_step != self.state.global_step:
            raise RuntimeError("vLLM rollout did not synchronize to the current optimizer step.")
        if label == "weight_sync":
            emit({"event": "weight_sync", "step": self.state.global_step,
                  "parameter_probe": probe(self.model)})
        return result

    setattr(cls, method, observed)


def install():
    check_versions()
    if getattr(Trainer, "_msloc_demo_monitor", False):
        return
    Trainer._msloc_demo_monitor = True
    original = Trainer.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.add_callback(DemoMonitor())
        backward = self.accelerator.backward

        @functools.wraps(backward)
        def timed_backward(*values, **options):
            sync()
            begin = time.perf_counter()
            result = backward(*values, **options)
            sync()
            emit({"event": "phase", "step": self.state.global_step + 1, "phase": "backward",
                  "seconds": time.perf_counter() - begin})
            return result

        self.accelerator.backward = timed_backward

    Trainer.__init__ = initialize
    from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin
    from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    wrap(RolloutTrainerMixin, "_generate_completions", "generate_including_sync", check_sync=True)
    wrap(RolloutTrainerMixin, "_move_model_to_vllm", "weight_sync")
    wrap(GRPOTrainer, "_score_completions", "reward")
    wrap(GRPOTrainer, "compute_loss", "loss_forward_including_reference")
    wrap(GKDTrainer, "compute_loss", "loss_forward_including_teacher")
    # Wrap the base training_step; subclasses ultimately delegate here.
    wrap(Trainer, "training_step", "training_microstep_including_rollout")


if os.environ.get("MSLOC_DEMO_RUN"):
    install()
