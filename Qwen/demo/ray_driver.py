"""Synchronous Ray multi-role OPSD/GRPO; models live inside persistent actors."""

import argparse
import importlib.metadata
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from demo_utils import (check_versions, data_identity, device_ids, environment, model_identity,
                        prepare_output, read_json, read_rows, relative, require_root, sha256,
                        source_identity, write_json, write_rows)
from ray_protocol import batch_indices, make_layout, validate_jobs


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["opsd", "grpo"], required=True)
    p.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    p.add_argument("--schedule", choices=["serial", "parallel"], default="parallel",
                   help="Dispatch independent rollout/scoring work serially or concurrently; DDP is always collective")
    for key in ("model", "data", "output"):
        p.add_argument("--" + key, required=True)
    for key in ("adapter", "teacher-model", "teacher-adapter", "teacher-gate", "alignment", "nli-model"):
        p.add_argument("--" + key, default="none")
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--tuner", choices=["lora", "full"], default="lora")
    for key, default in (("lora-rank", 16), ("lora-alpha", 32), ("steps", 32), ("save-steps", 8),
                         ("global-batch-size", 8), ("num-generations", 4), ("max-length", 8192),
                         ("max-new-tokens", 256), ("sync-chunk-mb", 256), ("cpus-per-worker", 2),
                         ("timeout-seconds", 1800)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--vllm-memory", type=float, default=0.35)
    p.add_argument("--sync-atol", type=float, default=0.1, help="Maximum absolute log-probability error in every sync probe")
    p.add_argument("--object-store-gb", type=float, default=8)
    p.add_argument("--resume", choices=["none", "auto"], default="none")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def validate(args):
    count = len(device_ids(args.devices))
    if min(args.steps, args.save_steps, args.global_batch_size, args.lora_rank, args.lora_alpha,
           args.max_new_tokens, args.sync_chunk_mb, args.cpus_per_worker, args.timeout_seconds) <= 0:
        raise ValueError("Counts must be positive.")
    if (not all(math.isfinite(x) and x > 0 for x in (args.learning_rate, args.sync_atol, args.object_store_gb))
            or not 0 < args.vllm_memory < 1 or args.max_length <= args.max_new_tokens):
        raise ValueError("Invalid length, learning rate, memory or synchronization tolerance.")
    if args.tuner == "full" and args.adapter != "none":
        raise ValueError("Merge an adapter into complete HF weights before full tuning.")
    layout = make_layout(count, args.stage)
    if args.global_batch_size < sum(r["role"] == "trainer" for r in layout):
        raise ValueError("Every trainer needs at least one completion per update.")
    if len(layout) * args.cpus_per_worker > (os.cpu_count() or 1):
        raise ValueError("CPU reservation exceeds this server; adjust --cpus-per-worker.")
    if args.stage == "grpo" and (args.num_generations < 2 or args.global_batch_size % args.num_generations):
        raise ValueError("GRPO batch must divide into groups of at least two completions.")
    weights = model_identity(args.model, args.adapter)
    if args.adapter != "none":
        cfg = read_json(Path(args.adapter) / "adapter_config.json")
        if (cfg["r"] != args.lora_rank or cfg["lora_alpha"] != args.lora_alpha
                or cfg.get("modules_to_save") or cfg.get("use_dora") or cfg.get("bias", "none") != "none"):
            raise ValueError("Adapter rank/alpha or supported language-LoRA configuration mismatch.")
        if any("visual" in str(t) for t in cfg.get("target_modules", [])):
            raise ValueError("Visual LoRA is unsupported.")
    data, sources = data_identity(args.data), source_identity()
    versions = check_versions(args.backend == "vllm")
    versions.update(ray=importlib.metadata.version("ray"), torch=importlib.metadata.version("torch"))
    if versions["ray"] != "2.59.0":
        raise ValueError("Install requirements-ray.txt; Ray 2.59.0 is required.")
    extra, protected = {}, [args.model, args.data]
    if args.adapter != "none":
        protected.append(args.adapter)
    dataset = Path(args.data) / f"{args.stage}.jsonl"
    if args.stage == "opsd":
        teacher = model_identity(args.teacher_model, args.teacher_adapter)
        for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
            if sha256(Path(args.model) / name) != sha256(Path(args.teacher_model) / name):
                raise ValueError(f"Teacher/student tokenizer or processor differs: {name}")
        gate = read_json(args.teacher_gate)
        if (gate["passed"] is not True or gate["student"]["weights"] != weights
                or gate["teacher"]["weights"] != teacher or gate["student"]["data"] != data
                or gate["student"]["source"] != sources):
            raise ValueError("Teacher gate does not match weights/data/demo sources.")
        dataset = Path(gate["dataset"])
        if sha256(dataset) != gate["dataset_sha256"]:
            raise ValueError("Teacher-filtered dataset changed.")
        extra.update(teacher=teacher, gate=sha256(args.teacher_gate))
        protected += [args.teacher_model, args.teacher_gate, dataset]
        if args.teacher_adapter != "none":
            protected.append(args.teacher_adapter)
    else:
        extra["nli"] = {name: sha256(Path(args.nli_model) / name) for name in
                        ("config.json", "model.safetensors", "tokenizer.json")}
        protected.append(args.nli_model)
    if args.backend == "vllm":
        alignment = read_json(args.alignment)
        if (alignment["passed"] is not True or alignment["config"]["weights"] != model_identity(args.model)
                or alignment["config"]["data"] != data or alignment["config"]["source"] != sources
                or alignment["config"]["max_length"] != args.max_length):
            raise ValueError("Re-run actual HF/vLLM input alignment for this model/data/source.")
        extra["alignment"] = sha256(args.alignment)
        protected.append(args.alignment)
    rows = read_rows(dataset)
    unique = args.global_batch_size // (args.num_generations if args.stage == "grpo" else 1)
    if unique > len(rows):
        raise ValueError("Too few filtered proposals for one batch.")
    cfg = {k: v for k, v in vars(args).items() if k not in ("resume", "clean", "plan_only")}
    config = {"kind": "ray_training", "schema": 1, "parameters": cfg, "weights": weights,
              "data": data, "dataset_sha256": sha256(dataset), "sources": sources, "versions": versions,
              "extra": extra, "layout": layout, "unique_inputs_per_batch": unique,
              "method": {"seed": 42, "opsd_beta": 0.5, "opsd_lmbda": 1, "sft_alpha": 0,
                         "grpo_beta": 0.02, "clip": 0.2, "reward_weights": [1.0, 0.1, 0.3],
                         "num_iterations": 1, "reduction": "mean_per_sequence", "dropout": 0.0,
                         "optimizer": "AdamW", "weight_decay": 0.0, "max_grad_norm": 1.0}}
    return config, rows, protected


def resume_checkpoint(output):
    paths = [p for p in output.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()
             and (p / "COMPLETE.json").is_file()]
    if not paths:
        raise FileNotFoundError("No committed checkpoint; use a new output or explicit --clean.")
    path = max(paths, key=lambda p: int(p.name.split("-")[-1]))
    manifest = read_json(path / "COMPLETE.json")
    if manifest["run_sha256"] != sha256(output / "run.json"):
        raise ValueError("Checkpoint belongs to a different run.")
    if manifest["step_sha256"] != sha256(output / "steps" / f"step-{manifest['version']:06d}.json"):
        raise ValueError("Checkpoint step record is missing or changed.")
    for name, expected in manifest["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return relative(path), manifest["version"]


def main():
    require_root()
    args = arguments()
    config, rows, protected = validate(args)
    print(f"Ray multi-role | {args.stage} | {args.backend} | {args.schedule} | batch={args.global_batch_size}")
    print("Placement:", config["layout"])
    if args.plan_only:
        return
    output = prepare_output(args.output, config, args.resume, args.clean, protected)
    write_rows(output / "train.jsonl", rows)
    resume, first_version = resume_checkpoint(output) if args.resume == "auto" else (None, 0)
    env = environment(args.devices)
    subprocess.run([sys.executable, "Qwen/demo/check_lengths.py", "--model", args.model,
                    "--dataset", relative(output / "train.jsonl"), "--stage", args.stage,
                    "--max-length", str(args.max_length), "--max-new-tokens", str(args.max_new_tokens),
                    "--output", relative(output / "length_check.json")], env=env, check=True)
    os.environ.update(env)
    if not Path("/dev/shm").is_dir() or shutil.disk_usage("/dev/shm").free < args.object_store_gb * 1024**3:
        raise ValueError("Ray benchmark requires enough /dev/shm for --object-store-gb; enlarge shared memory or set an explicit fitting budget.")
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from ray_workers import TrainerWorker, RolloutWorker, ScorerWorker, RewardWorker, signal_metadata
    from tqdm import tqdm
    session = uuid.uuid4().hex
    sessions = read_json(output / "sessions.json") if (output / "sessions.json").exists() else []
    sessions.append({"id": session, "start_version": first_version})
    write_json(output / "sessions.json", sessions)
    ray.init(address="local", num_gpus=len(device_ids(args.devices)),
             num_cpus=len(config["layout"]) * args.cpus_per_worker,
             object_store_memory=int(args.object_store_gb * 1024**3), include_dashboard=False, log_to_driver=True,
             _system_config={"automatic_object_spilling_enabled": False})
    pool = {name: [] for name in ("trainer", "rollout", "scorer", "reward")}
    all_actors, pg = [], None
    metadata_task = ray.remote(num_cpus=0)(signal_metadata)

    def get(refs):
        return ray.get(refs, timeout=args.timeout_seconds)

    def sync_weights(version, full=False):
        started = time.perf_counter()
        manifest = get(pool["trainer"][0].manifest.remote(version, full))
        get([a.begin_sync.remote(manifest) for a in pool["rollout"]])
        transferred = 0
        for index in range(len(manifest["chunks"])):
            payload = pool["trainer"][0].export_chunk.remote(version, index)
            receipts = get([a.load_chunk.remote(payload) for a in pool["rollout"]])
            if len({r["checksum"] for r in receipts}) != 1:
                raise ValueError("Rollout replicas received different weights.")
            transferred += receipts[0]["bytes"]
            del payload
        checksums = get(pool["trainer"][0].export_receipts.remote())
        acknowledgments = get([a.commit_sync.remote(version, checksums) for a in pool["rollout"]])
        applied = time.perf_counter()
        expected = get(pool["trainer"][0].probe.remote(rows[0], version))
        probes = get([a.verify.remote(rows[0], expected) for a in pool["rollout"]])
        return {"version": version, "chunks": len(checksums), "bytes_per_receiver": transferred,
                "checksums": checksums, "acknowledgments": acknowledgments, "probes": probes,
                "transfer_seconds": applied-started, "probe_seconds": time.perf_counter()-applied,
                "seconds": time.perf_counter()-started}

    def checkpoint(version):
        pending = output / f"checkpoint-{version}-{session}.partial"
        pending.mkdir()
        statuses = get([a.save.remote(relative(pending)) for a in pool["trainer"]])
        if any(s["version"] != version for s in statuses) or len({s["trainable_digest"] for s in statuses}) != 1:
            raise ValueError("Trainer replicas disagree at checkpoint barrier.")
        files = {p.relative_to(pending).as_posix(): sha256(p) for p in pending.rglob("*") if p.is_file()}
        return pending, {"version": version, "session": session, "files": files,
                         "trainers": statuses, "run_sha256": sha256(output / "run.json")}

    initialized = time.perf_counter()
    try:
        bundles = [{"GPU": 1, "CPU": args.cpus_per_worker * sum(r["slot"] == slot for r in config["layout"])}
                   for slot in range(len(device_ids(args.devices)))]
        pg = placement_group(bundles, strategy="STRICT_PACK")
        get(pg.ready())
        classes = {"trainer": TrainerWorker, "rollout": RolloutWorker, "scorer": ScorerWorker, "reward": RewardWorker}
        for spec in config["layout"]:
            role = spec["role"]
            actor_type = ray.remote(num_gpus=spec["gpu"], num_cpus=args.cpus_per_worker,
                                   max_restarts=0, max_task_retries=0)(classes[role])
            placement = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=spec["slot"],
                                                         placement_group_capture_child_tasks=False)
            params = (config["parameters"], spec["rank"], resume) if role == "trainer" else (config["parameters"], spec["rank"])
            actor = actor_type.options(scheduling_strategy=placement).remote(*params)
            pool[role].append(actor)
            all_actors.append(actor)
        write_json(output / "hardware.json", get([a.info.remote() for a in all_actors]))
        trainers = len(pool["trainer"])
        host, port = get(pool["trainer"][0].rendezvous.remote(trainers)) if trainers > 1 else ("", 0)
        ready = get([a.connect.remote(host, port, trainers) for a in pool["trainer"]])
        if any(r["version"] != first_version for r in ready) or len({r["probe"] for r in ready}) != 1:
            raise ValueError("Initial trainer replicas disagree.")
        write_json(output / f"initial_sync-{session}.json", sync_weights(first_version, full=True))
        get([a.drain.remote() for a in all_actors])
        startup, loop_start = time.perf_counter()-initialized, time.perf_counter()
        final_checkpoint = resume
        write_json(output / "status.json", {"state": "running", "version": first_version, "session": session})
        for version in tqdm(range(first_version, args.steps), desc=f"Ray {args.stage}", unit="update", initial=first_version, total=args.steps):
            started = time.perf_counter()
            indices = batch_indices(len(rows), version, config["unique_inputs_per_batch"])
            jobs = []
            repetitions = args.num_generations if args.stage == "grpo" else 1
            for group, row_index in enumerate(indices):
                for sample in range(repetitions):
                    index = len(jobs)
                    jobs.append({"index": index, "group": group, "sample": sample, "row": rows[row_index],
                                 "seed": 42+version*args.global_batch_size+index, "version": version})
            requests, results = [], []
            for rank, actor in enumerate(pool["rollout"]):
                shard = jobs[rank::len(pool["rollout"])]
                if shard:
                    ref = actor.generate.remote(shard, version)
                    if args.schedule == "serial":
                        results.extend(get(ref))
                    else:
                        requests.append(ref)
            for result in get(requests):
                results.extend(result)
            jobs = sorted(results, key=lambda j: j["index"])
            validate_jobs(jobs, version, args.global_batch_size)
            generated = time.perf_counter()
            signals, metas, reward_refs = [], [], []
            for job in jobs:
                ref = pool["scorer"][job["index"] % len(pool["scorer"])].score.remote(job)
                signals.append(ref)
                meta = metadata_task.remote(ref)
                if args.schedule == "serial":
                    get(meta)
                metas.append(meta)
            if args.stage == "grpo":
                for job in jobs:
                    ref = pool["reward"][job["index"] % len(pool["reward"])].score.remote(job)
                    if args.schedule == "serial":
                        get(ref)
                    reward_refs.append(ref)
            metadata = get(metas)
            if any(m["index"] != j["index"] or m["version"] != version or m["tokens"] != j["tokens"] for m, j in zip(metadata, jobs)):
                raise ValueError("Scoring barrier received mismatched sample identities.")
            advantages, reward_metrics = [0.0]*len(jobs), {}
            if args.stage == "grpo":
                import torch
                from swift.rl_core.advantage import compute_advantages
                scored = get(reward_refs)
                if any(s["index"] != j["index"] or s["version"] != version for s, j in zip(scored, jobs)):
                    raise ValueError("Reward result identity mismatch.")
                components = torch.tensor([s["scores"] for s in scored], dtype=torch.float32)
                advantage_tensor, rewards = compute_advantages(components, torch.tensor([1.0, 0.1, 0.3]), repetitions,
                                                             advantage_estimator="grpo", scale_rewards="group")
                advantages = advantage_tensor.tolist()
                reward_metrics = {"mean": float(rewards.mean()), "components": components.mean(0).tolist(),
                                  "zero_std_fraction": float((rewards.reshape(-1, repetitions).std(1) == 0).float().mean())}
                for job, comp, score, advantage in zip(jobs, components.tolist(), rewards.tolist(), advantages):
                    job.update(reward_components=comp, reward=score, advantage=advantage)
            scored_at = time.perf_counter()
            updates = get([a.update.remote(jobs[r::trainers], signals[r::trainers], advantages[r::trainers],
                                          version, len(jobs)) for r, a in enumerate(pool["trainer"])])
            if any(u["version"] != version+1 for u in updates) or len({u["probe"] for u in updates}) != 1:
                raise ValueError("Trainer replicas disagree after update.")
            updated = time.perf_counter()
            synchronization = sync_weights(version+1)
            synchronized = time.perf_counter()
            checkpoint_time = 0.0
            pending_checkpoint = None
            if (version+1) % args.save_steps == 0 or version+1 == args.steps:
                pending_checkpoint = checkpoint(version+1)
                checkpoint_time = time.perf_counter()-synchronized
            traces = get([a.drain.remote() for a in all_actors])
            record = {"step": version+1, "session": session, "start": started, "end": time.perf_counter(),
                      "input_version": version, "output_version": version+1, "batch_indices": indices,
                      "loss": sum(u["loss_sum"] for u in updates)/len(jobs), "reward_metrics": reward_metrics,
                      "updates": updates, "sync": synchronization, "traces": traces,
                      "phases": {"rollout": generated-started, "score": scored_at-generated,
                                 "train": updated-scored_at, "sync": synchronized-updated, "checkpoint": checkpoint_time},
                      "jobs": [{k: v for k, v in j.items() if k != "row"} | {"sample_id": j["row"]["sample_id"]} for j in jobs]}
            write_json(output / "steps" / f"step-{version+1:06d}.json", record)
            if pending_checkpoint is not None:
                pending, manifest = pending_checkpoint
                manifest["step_sha256"] = sha256(output / "steps" / f"step-{version+1:06d}.json")
                write_json(pending / "COMPLETE.json", manifest)
                destination = output / f"checkpoint-{version+1}"
                pending.rename(destination)
                final_checkpoint = relative(destination)
            write_json(output / "status.json", {"state": "running", "version": version+1, "session": session})
            print(f"\n[Ray] step={version+1} loss={record['loss']:.5f} time={record['end']-started:.2f}s "
                  f"sync={synchronization['seconds']:.2f}s version={version}->{version+1}", flush=True)
            del signals, metas, reward_refs
        write_json(output / "status.json", {"state": "complete", "version": args.steps, "session": session,
                   "checkpoint": final_checkpoint, "startup_seconds": startup,
                   "loop_start": loop_start, "loop_end": time.perf_counter(), "loop_seconds": time.perf_counter()-loop_start})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed_or_interrupted", "error": str(error), "session": session})
        raise
    finally:
        for actor in all_actors:
            ray.kill(actor, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()
    subprocess.run([sys.executable, "Qwen/demo/ray_report.py", "--runs", args.output,
                    "--output", relative(output / "report"), "--warmup-steps", "4"], check=True)


if __name__ == "__main__":
    main()
