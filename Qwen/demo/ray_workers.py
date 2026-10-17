"""Actual Ray role implementations; the driver wraps these classes with ray.remote."""

import os
import random
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from demo_utils import ROOT, digest
from ray_protocol import WeightVersion, tensor_checksum, verify_signal
from ray_models import (build_model, effective_weights, encode_prompt, grpo_loss, learning_rate_factor,
                        opsd_loss, response_logits, selected_logps, trainable_probe)


def signal_metadata(signal):
    """Keep large teacher logits in Ray's object store instead of fetching them into the driver."""
    result = {k: signal[k] for k in ("index", "version", "tokens")}
    tensor = signal["logits"] if "logits" in signal else signal["logps"]
    if tensor.shape[0] != len(signal["tokens"]):
        raise ValueError("Invalid teacher/reference signal.")
    return result


class Worker:
    def __init__(self, config, rank, role, gpu=True):
        os.chdir(ROOT)
        self.cfg, self.rank, self.role = config, rank, role
        self.events = []
        self.gpu = gpu
        torch.set_num_threads(config["cpus_per_worker"])
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if gpu:
            if torch.cuda.device_count() != 1:
                raise RuntimeError("Each GPU actor must see exactly its assigned GPU.")
            torch.cuda.set_device(0)
            torch.cuda.manual_seed_all(42)

    @contextmanager
    def timing(self, phase, version):
        if self.gpu:
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.gpu:
                torch.cuda.synchronize()
            self.events.append({"role": self.role, "rank": self.rank, "phase": phase,
                                "version": version, "start": started, "end": time.perf_counter()})

    def info(self):
        result = {"role": self.role, "rank": self.rank, "pid": os.getpid(),
                  "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
        if self.gpu:
            prop = torch.cuda.get_device_properties(0)
            result.update(gpu_name=prop.name, total_memory=prop.total_memory)
        return result

    def drain(self):
        result, self.events = self.events, []
        memory = None
        if self.gpu:
            free, total = torch.cuda.mem_get_info()
            memory = {"device_used_gib": (total - free) / 2**30,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            torch.cuda.reset_peak_memory_stats()
        return {"role": self.role, "rank": self.rank, "events": result, "memory": memory}


class TrainerWorker(Worker):
    def __init__(self, config, rank, resume):
        super().__init__(config, rank, "trainer")
        path, adapter = config["model"], config["adapter"]
        if resume:
            if config["tuner"] == "full":
                path, adapter = str(Path(resume) / "model"), "none"
            else:
                adapter = str(Path(resume) / "model")
        self.model, self.processor, self.template = build_model(config, path, adapter, trainable=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Trainer has no trainable parameters.")
        self.optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: learning_rate_factor(step, config["steps"]))
        self.version = 0
        if resume:
            state = torch.load(Path(resume) / f"trainer-{rank}.pt", map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(torch.load(Path(resume) / "optimizer.pt", map_location="cpu", weights_only=False))
            self.scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"])
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            self.version = state["version"]
        self.distributed = self.model
        self.world = 1
        self.store = None
        self.exported = []

    def rendezvous(self, world):
        import ray
        host = ray.util.get_node_ip_address()
        self.store = torch.distributed.TCPStore(host, 0, world, True,
                    timeout=timedelta(seconds=self.cfg["timeout_seconds"]), wait_for_workers=False)
        return host, self.store.port

    def connect(self, host, port, world):
        self.world = world
        if world > 1:
            if self.store is None:
                self.store = torch.distributed.TCPStore(host, port, world, False,
                             timeout=timedelta(seconds=self.cfg["timeout_seconds"]))
            torch.distributed.init_process_group("nccl", store=self.store, rank=self.rank, world_size=world,
                        timeout=timedelta(seconds=self.cfg["timeout_seconds"]), device_id=torch.device("cuda:0"))
            self.distributed = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[0], broadcast_buffers=False, find_unused_parameters=False)
        return {"rank": self.rank, "version": self.version,
                "probe": tensor_checksum(trainable_probe(self.model))}

    @torch.no_grad()
    def manifest(self, version, full):
        if version != self.version:
            raise ValueError("Cannot export an old/new student version.")
        changed = set()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if ".lora_" in name:
                    changed.add(name.split(".lora_", 1)[0].removeprefix("base_model.model.") + ".weight")
                else:
                    changed.add(name.removeprefix("base_model.model.").replace(".base_layer", ""))
        chunks, current, size = [], [], 0
        limit = self.cfg["sync_chunk_mb"] * 1024**2
        for name, tensor in effective_weights(self.model, None if full else changed):
            amount = tensor.numel() * tensor.element_size()
            if current and size + amount > limit:
                chunks.append(current)
                current, size = [], 0
            current.append(name)
            size += amount
        if current:
            chunks.append(current)
        if not chunks:
            raise ValueError("No weights selected for synchronization.")
        self.exported = []
        self.export_chunks = chunks
        return {"version": version, "chunks": chunks, "full": full}

    @torch.no_grad()
    def export_chunk(self, version, index):
        if version != self.version or index != len(self.exported):
            raise ValueError("Weight export order/version mismatch.")
        with self.timing("weight_export", version):
            names = self.export_chunks[index]
            weights = dict(effective_weights(self.model, set(names)))
            items = [(name, weights[name].cpu().contiguous().clone()) for name in names]
            checksum = tensor_checksum(items)
            self.exported.append(checksum)
            return {"version": version, "index": index, "weights": items, "checksum": checksum}

    def export_receipts(self):
        return self.exported

    @torch.no_grad()
    def probe(self, row, version):
        if version != self.version:
            raise ValueError("Probe used wrong student version.")
        with self.timing("sync_probe_source", version):
            self.model.eval()
            tokens = self.processor.tokenizer.encode(" normal video", add_special_tokens=False)
            logits = response_logits(self.model, self.template, row, tokens, self.cfg["max_length"])
            values = selected_logps(logits, tokens).cpu().tolist()
            prompt = encode_prompt(self.template, row)["input_ids"]
            return {"tokens": tokens, "logps": values, "prompt_digest": digest(list(prompt)), "version": version}

    def update(self, jobs, signals, advantages, version, batch_size):
        import ray
        if version != self.version or not jobs or len(jobs) != len(signals) or len(jobs) != len(advantages):
            raise ValueError("Trainer batch/version mismatch.")
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        before = tensor_checksum(trainable_probe(self.model))
        losses, kls = [], []
        with self.timing("train_update", version):
            for index, (job, signal_ref, advantage) in enumerate(zip(jobs, signals, advantages)):
                signal = ray.get(signal_ref)
                verify_signal(job, signal)
                context = self.distributed.no_sync() if self.world > 1 and index < len(jobs) - 1 else nullcontext()
                with context:
                    logits = response_logits(self.distributed, self.template, job["row"], job["tokens"], self.cfg["max_length"])
                    if self.cfg["stage"] == "opsd":
                        loss = opsd_loss(logits, signal["logits"].to("cuda:0"))
                    else:
                        logps = selected_logps(logits, job["tokens"], temperature=0.8)
                        loss, kl = grpo_loss(logps, signal["logps"].to("cuda:0"), advantage)
                        kls.append(float(kl))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss.")
                    losses.append(float(loss.detach()))
                    # DDP averages ranks; this factor gives a global per-sequence mean even for uneven shards.
                    (loss * self.world / batch_size).backward()
                del signal, logits, loss
            norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                                  1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()
            self.version += 1
        self.model.eval()
        after = tensor_checksum(trainable_probe(self.model))
        return {"rank": self.rank, "version": self.version, "loss_sum": sum(losses), "samples": len(jobs),
                "kl_sum": sum(kls), "grad_norm": float(norm), "lr": self.optimizer.param_groups[0]["lr"],
                "probe_changed": before != after, "probe": after}

    def save(self, directory):
        path = Path(directory)
        with self.timing("checkpoint", self.version):
            torch.save({"version": self.version, "scheduler": self.scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state()}, path / f"trainer-{self.rank}.pt")
            if self.rank == 0:
                torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
                self.model.save_pretrained(path / "model", safe_serialization=True)
                self.processor.save_pretrained(path / "model")
            return {"rank": self.rank, "version": self.version,
                    "trainable_digest": tensor_checksum([(n, p) for n, p in self.model.named_parameters() if p.requires_grad])}


class RolloutWorker(Worker):
    def __init__(self, config, rank):
        super().__init__(config, rank, "rollout")
        self.versions = WeightVersion()
        self.verified = -1
        if config["backend"] == "vllm":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            from swift.infer_engine import VllmEngine
            self.engine = VllmEngine(config["model"], model_type="qwen3_5", torch_dtype=torch.bfloat16,
                    tensor_parallel_size=1, distributed_executor_backend="uni", use_async_engine=False,
                    gpu_memory_utilization=config["vllm_memory"], max_model_len=config["max_length"],
                    max_num_seqs=config["global_batch_size"], enable_lora=False, seed=42,
                    enable_prefix_caching=False, mm_processor_cache_gb=0,
                    engine_kwargs={"generation_config": "vllm"})
            from swift import get_template
            self.processor = self.engine.processor
            self.template = get_template(self.processor, enable_thinking=False)
            self.template.set_mode("vllm")
        else:
            self.model, self.processor, self.template = build_model(config, config["model"])
            from transformers import GenerationConfig
            # A received checkpoint's decoding defaults must not change the comparison.
            self.generation_config = GenerationConfig(do_sample=True, num_beams=1,
                max_new_tokens=config["max_new_tokens"], temperature=1.0 if config["stage"] == "opsd" else 0.8,
                top_p=1.0 if config["stage"] == "opsd" else 0.95, top_k=0, use_cache=True,
                repetition_penalty=1.0, eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id, return_dict_in_generate=True)

    def begin_sync(self, manifest):
        self.versions.begin(manifest["version"], manifest["chunks"])
        self.verified = -1

    @torch.no_grad()
    def load_chunk(self, payload):
        version, index, weights = payload["version"], payload["index"], payload["weights"]
        with self.timing("weight_apply", version):
            checksum = tensor_checksum(weights)
            if checksum != payload["checksum"]:
                raise ValueError("Weight payload checksum mismatch.")
            # Check transaction identity before mutating any model state.
            self.versions.accept(version, index, [n for n, _ in weights], checksum)
            if self.cfg["backend"] == "vllm":
                loaded = self.engine.inner_model.load_weights((n, t.to("cuda:0")) for n, t in weights)
                if not loaded:
                    raise RuntimeError("vLLM reported no loaded parameters for this chunk.")
            else:
                state = self.model.state_dict()
                for name, tensor in weights:
                    if name not in state or state[name].shape != tensor.shape or state[name].dtype != tensor.dtype:
                        raise ValueError(f"HF weight schema mismatch: {name}")
                    state[name].copy_(tensor.to("cuda:0"))
                    if not torch.equal(state[name].cpu(), tensor):
                        raise ValueError(f"HF weight readback mismatch: {name}")
            return {"version": version, "index": index, "checksum": checksum,
                    "bytes": sum(t.numel() * t.element_size() for _, t in weights)}

    def commit_sync(self, version, checksums):
        with self.timing("weight_commit", version):
            if self.cfg["backend"] == "vllm":
                # Same post-load operation as swift, but propagate failures instead of suppressing them.
                from vllm.model_executor.model_loader.utils import process_weights_after_loading
                process_weights_after_loading(self.engine.inner_model, self.engine.engine.model_config, torch.device("cuda:0"))
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.reset_encoder_cache()
            self.versions.commit(version, checksums)
        return {"rank": self.rank, "version": version, "chunks": len(checksums)}

    def vllm_requests(self, requests):
        """Use swift's multimodal request bridge and vLLM's synchronous engine loop."""
        from vllm import SamplingParams
        pending = set()
        for key, row, tokens, seed, probe in requests:
            encoded = encode_prompt(self.template, row)
            encoded["input_ids"] = list(encoded["input_ids"]) + tokens
            sampling = SamplingParams(max_tokens=1 if probe else self.cfg["max_new_tokens"],
                       temperature=1.0 if probe or self.cfg["stage"] == "opsd" else 0.8,
                       top_p=1.0 if probe or self.cfg["stage"] == "opsd" else 0.95, top_k=-1,
                       seed=seed, prompt_logprobs=1 if probe else None,
                       stop_token_ids=[self.processor.tokenizer.eos_token_id])
            self.engine._add_request(encoded, sampling, key)
            pending.add(key)
        results = {}
        while self.engine.engine.has_unfinished_requests():
            for result in self.engine.engine.step():
                if result.finished:
                    if result.request_id not in pending or result.request_id in results:
                        raise RuntimeError("Unexpected or duplicate vLLM completion.")
                    results[result.request_id] = result
        if set(results) != pending:
            raise RuntimeError("vLLM did not finish every request.")
        return results

    @torch.no_grad()
    def verify(self, row, expected):
        version, tokens = expected["version"], expected["tokens"]
        self.versions.require(version)
        with self.timing("sync_probe_receiver", version):
            if self.cfg["backend"] == "vllm":
                result = self.vllm_requests([("probe", row, tokens, 42, True)])["probe"]
                ids = result.prompt_token_ids
                if ids[-len(tokens):] != tokens:
                    raise ValueError("vLLM probe token IDs changed.")
                values = [result.prompt_logprobs[len(ids) - len(tokens) + i][token].logprob for i, token in enumerate(tokens)]
                prompt = ids[:-len(tokens)]
            else:
                logits = response_logits(self.model, self.template, row, tokens, self.cfg["max_length"])
                values = selected_logps(logits, tokens).cpu().tolist()
                prompt = encode_prompt(self.template, row)["input_ids"]
            if digest(list(prompt)) != expected["prompt_digest"]:
                raise ValueError("Rollout/trainer expanded prompt IDs differ.")
            errors = [abs(a - b) for a, b in zip(values, expected["logps"])]
            if len(values) != len(tokens) or not all(np.isfinite(values)) or max(errors) > self.cfg["sync_atol"]:
                raise ValueError(f"Weight-sync forward probe failed: errors={errors}; tolerance={self.cfg['sync_atol']}")
            self.verified = version
            return {"rank": self.rank, "version": version, "max_logp_error": max(errors), "verified": True}

    @torch.no_grad()
    def generate(self, jobs, version):
        self.versions.require(version)
        if self.verified != version:
            raise ValueError("Generation requires the forward-probe barrier.")
        with self.timing("rollout", version):
            if self.cfg["backend"] == "vllm":
                results = self.vllm_requests([(str(j["index"]), j["row"], [], j["seed"], False) for j in jobs])
            for job in jobs:
                if self.cfg["backend"] == "vllm":
                    answer = results[str(job["index"])].outputs[0]
                    tokens = list(answer.token_ids)
                    reason = answer.finish_reason
                else:
                    torch.manual_seed(job["seed"])
                    torch.cuda.manual_seed_all(job["seed"])
                    encoded = encode_prompt(self.template, job["row"])
                    prompt_length = len(encoded["input_ids"])
                    encoded.pop("labels", None)
                    encoded.pop("loss_scale", None)
                    batch = self.template.data_collator([encoded])
                    batch.pop("labels", None)
                    batch.pop("loss_scale", None)
                    batch = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    output = self.template.generate(self.model, **batch, generation_config=self.generation_config)
                    tokens = output.sequences[0, prompt_length:].cpu().tolist()
                    reason = "stop" if tokens and tokens[-1] == self.processor.tokenizer.eos_token_id else "length"
                if not tokens:
                    raise ValueError("Empty generated token sequence.")
                job.update(tokens=tokens, raw=self.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                           finish_reason=reason, version=version, rollout_rank=self.rank)
        return jobs


class ScorerWorker(Worker):
    def __init__(self, config, rank):
        super().__init__(config, rank, "teacher" if config["stage"] == "opsd" else "reference")
        teacher = config["stage"] == "opsd"
        path = config["teacher_model"] if teacher else config["model"]
        adapter = config["teacher_adapter"] if teacher else config["adapter"]
        self.model, self.processor, self.template = build_model(config, path, adapter)

    @torch.no_grad()
    def score(self, job):
        teacher = self.cfg["stage"] == "opsd"
        with self.timing("teacher_forward" if teacher else "reference_forward", job["version"]):
            logits = response_logits(self.model, self.template, job["row"], job["tokens"], self.cfg["max_length"], teacher)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite teacher/reference logits.")
            result = {k: job[k] for k in ("index", "version", "tokens")}
            result["logits" if teacher else "logps"] = (logits if teacher else selected_logps(logits, job["tokens"], 0.8)).cpu()
            return result


class RewardWorker(Worker):
    def __init__(self, config, rank):
        super().__init__(config, rank, "reward", gpu=False)
        os.environ["MSLOC_GRPO_NLI_MODEL"] = config["nli_model"]
        from grpo_rewards import LocalizationReward, FormatReward, ExplanationReward, EntailmentExplanationJudge
        self.rewards = [LocalizationReward(), FormatReward(), ExplanationReward()]
        # Include model startup before the measured training loop, even if early answers are invalid.
        self.rewards[2].judge = EntailmentExplanationJudge(nli_model_path=config["nli_model"], nli_device="cpu", nli_batch_size=32)

    def score(self, job):
        with self.timing("reward", job["version"]):
            kwargs = {key: [job["row"][key]] for key in ("target_kind", "target_relative", "clip_duration", "evidence")}
            scores = [reward([job["raw"]], **kwargs)[0] for reward in self.rewards]
            if not all(np.isfinite(scores)):
                raise FloatingPointError("Non-finite reward.")
            return {"index": job["index"], "version": job["version"], "scores": scores}
