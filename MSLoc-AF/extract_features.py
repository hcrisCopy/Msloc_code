"""Cache paired pre/post-Mamba window features without changing the original PG."""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from queue import Empty

import numpy as np
import torch

from common import (feature_id, load_records, make_windows, read_json, sample_indices,
                    sha256, validate_entry, write_json)
from extraction_checks import (DEFAULT_FRAME_COUNT_TOLERANCE, check_frame_count,
                               require_compatible_cache)
from cli_config import parse_stage_args
from display import tqdm
from parallel import assigned_device, parse_device_ids, validate_cuda_devices


def aggregate_tokens(tokens, batch_size):
    # Identical aggregation to DeMamba.py's res.mean(1).view(b, -1).
    return tokens.mean(1).reshape(batch_size, -1)


def read_clip(paths, crop_youku):
    import cv2
    frames = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if crop_youku and "youku" in str(path):
            height, width = image.shape[:2]
            if width > height:
                cut = int(width * 0.15)
                if cut:
                    image = image[:, cut:width-cut]
            else:
                cut = int(height * 0.15)
                if cut:
                    image = image[cut:height-cut, :]
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        frames.append(image.transpose(2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(np.stack(frames)))


def load_pg(checkpoint, weights, device):
    original = Path(__file__).resolve().parents[1] / "MSLoc-PG"
    os.environ["XCLIP_WEIGHTS_PATH"] = str(Path(weights).resolve())
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(original))
    from models.DeMamba import XCLIP_DeMamba
    model = XCLIP_DeMamba(class_num=4)
    # Checkpoints must be trusted local training artifacts.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="Shared trained four-class PG checkpoint")
    parser.add_argument("--xclip-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-ids", help="GPU IDs, e.g. 0,1,2,3; distribute complete videos across GPUs")
    parser.add_argument("--batch-size", type=int, default=2, help="Windows per forward PER GPU")
    parser.add_argument("--frame-fps", type=float, default=8.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--frame-count-tolerance", type=int, default=DEFAULT_FRAME_COUNT_TOLERANCE,
                        help="Allowed absolute frame-count error; default 4 frames (0.5s at 8 FPS)")
    parser.add_argument("--no-crop-youku", action="store_true")
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-videos", type=int, default=0, help="Smoke runs only; 0 means all")
    args = parse_stage_args(parser,"extract")
    if (args.batch_size <= 0 or args.frame_fps <= 0 or args.frames_per_window <= 0
            or args.frame_count_tolerance < 0 or args.max_videos < 0):
        parser.error("Invalid batch size, FPS, sample count, tolerance or video limit")
    if args.device_ids is not None:
        try:
            args.device_ids = parse_device_ids(args.device_ids)
        except ValueError as error:
            parser.error(str(error))
        if not args.device.startswith("cuda"):
            parser.error("--device-ids selects CUDA GPUs; omit it when using --device cpu")
    return args


class PairedExtractor:
    """One original PG replica and one local hook per extraction process."""
    def __init__(self, args, spec, signature, model_loader=None, clip_loader=None):
        self.args, self.spec, self.signature = args, spec, signature
        self.model_loader, self.clip_loader = model_loader or load_pg, clip_loader or read_clip
        self.model, self.handle, self.captured = None, None, {}

    def close(self):
        if self.handle is not None:
            self.handle.remove()

    def extract(self, record):
        args, spec = self.args, self.spec
        video = record["video_path"]
        frame_dir = Path(args.frames_root) / Path(video.replace("\\", "/")).with_suffix("")
        if not frame_dir.resolve().is_relative_to(Path(args.frames_root).resolve()):
            raise ValueError(f"Frame path escapes root: {video}")
        paths = sorted(frame_dir.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))
        if not paths or [int(p.stem.split("_")[-1]) for p in paths] != list(range(1,len(paths)+1)):
            raise ValueError(f"Missing/non-contiguous frame_1.jpg... files: {frame_dir}")
        count_check = check_frame_count(video,float(record["duration"]),args.frame_fps,
                                        len(paths),args.frame_count_tolerance)
        windows = make_windows(record["duration"],args.window_seconds,args.stride_seconds)
        indices = [sample_indices(start,end,args.frame_fps,len(paths),args.frames_per_window)
                   for start,end in windows]
        if self.model is None:
            self.model = self.model_loader(args.checkpoint,args.xclip_weights,args.device)
            expected_frames = getattr(self.model.encoder.config,"num_frames",args.frames_per_window)
            if args.frames_per_window != expected_frames:
                raise ValueError(f"XCLIP expects {expected_frames} frames per window")
            def capture(_module, inputs, output):
                self.captured["xclip"] = inputs[0].detach()
                self.captured["mamba"] = output.detach()
            self.handle = self.model.mamba.register_forward_hook(capture)
        features = {name:[] for name in ("xclip","mamba")}
        with torch.inference_mode():
            for first in tqdm(range(0,len(windows),args.batch_size),desc="Extract windows",
                              unit="batch",leave=False,dynamic_ncols=True,disable=not args.show_progress):
                clip_indices = indices[first:first+args.batch_size]
                batch = torch.stack([self.clip_loader([paths[i] for i in ids],not args.no_crop_youku)
                                     for ids in clip_indices]).to(args.device)
                self.model(batch)
                for name in features:
                    value = aggregate_tokens(self.captured[name],len(clip_indices))
                    if value.shape[1] != spec["feature_dim"] or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid {name} features: {video}, {value.shape}")
                    stored = value.float().cpu().numpy().astype(np.dtype(args.storage_dtype))
                    if not np.isfinite(stored).all():
                        raise ValueError("Storage overflow; retry with --storage-dtype float32")
                    features[name].append(stored)
                self.captured.clear()
        root, identity = Path(args.output), feature_id(video)
        for name in features:
            folder = root/name
            folder.mkdir(exist_ok=True)
            path = folder/(identity+".npy")
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream,np.concatenate(features[name]),allow_pickle=False)
            temporary.replace(path)
        entry = dict(feature_id=identity,duration=float(record["duration"]),
            num_windows=len(windows),num_source_frames=len(paths),frame_count_check=count_check,
            extractor_sha256=spec["extractor_sha256"],window_bounds_seconds=windows,
            sampled_frame_indices=indices,
            nominal_centers_seconds=[start+args.window_seconds/2 for start,_ in windows])
        # A durable marker is written ONLY after both feature files are complete.
        # It recovers a finished video even if the parent dies before manifest save.
        write_json(root/"video_commits"/(identity+".json"),
                   dict(video_id=video,cache_signature=self.signature,entry=entry))
        return entry


def _extraction_worker(rank, args, shards, spec, signature, queue, model_loader, clip_loader):
    args = copy.copy(args)
    device = assigned_device(args.device,args.device_ids,rank,distributed=True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.device, args.show_progress = str(device), False
    extractor = PairedExtractor(args,spec,signature,model_loader,clip_loader)
    try:
        for record in shards[rank]:
            entry = extractor.extract(record)
            queue.put((record["video_path"],entry))
    finally:
        extractor.close()


def extract_parallel(args, pending, spec, signature, processes, commit, pbar,
                     model_loader=None, clip_loader=None):
    count = min(processes,len(pending))
    shards = [pending[rank::count] for rank in range(count)]
    queue = torch.multiprocessing.get_context("spawn").Queue()
    context = torch.multiprocessing.spawn(_extraction_worker,
        args=(args,shards,spec,signature,queue,model_loader,clip_loader),nprocs=count,join=False)
    print(f"Parallel extraction: {count} workers; windows per GPU batch={args.batch_size}",flush=True)
    def receive(item):
        video,entry = item
        commit(video,entry)
        pbar.update(1)
        pbar.set_postfix(windows=entry["num_windows"],video=Path(video).name)
    try:
        while True:
            try:
                receive(queue.get(timeout=0.1))
            except Empty:
                pass
            if context.join(timeout=0):
                break
        while True:
            try:
                receive(queue.get_nowait())
            except Empty:
                break
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join()
        queue.close()
        queue.join_thread()


def run_extraction(args, processes=None, model_loader=None, clip_loader=None):
    make_windows(1, args.window_seconds, args.stride_seconds)
    records = {}
    for path in args.annotations:
        for record in load_records(path):
            old = records.get(record["video_path"])
            if old is not None and old != record:
                raise ValueError(f"Conflicting annotations: {record['video_path']}")
            records[record["video_path"]] = record
    selected = list(records.values())[:args.max_videos or None]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    model_path = Path(__file__).resolve().parents[1] / "MSLoc-PG/models/DeMamba.py"
    spec = dict(checkpoint_path=str(Path(args.checkpoint).resolve()),
                checkpoint_sha256=sha256(args.checkpoint),
                xclip_weights_path=str(Path(args.xclip_weights).resolve()),
                source_sha256=sha256(model_path),
                extractor_sha256=sha256(__file__),
                common_sha256=sha256(Path(__file__).parent / "common.py"),
                pscan_sha256=sha256(model_path.parent / "clip/pscan.py"),
                frame_fps=args.frame_fps, window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds, frames_per_window=args.frames_per_window,
                sampler="legacy_pg_inclusive_endpoint_v1", image_size=224,
                normalization="imagenet", crop_youku=not args.no_crop_youku,
                aggregation="legacy_mamba_sequence_mean_then_flatten_no_global",
                feature_dim=196*768, storage_dtype=args.storage_dtype,
                frame_count_tolerance=args.frame_count_tolerance,
                frames_root=str(Path(args.frames_root).resolve()))
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else dict(
        schema_version=1, spec=spec, videos={})
    require_compatible_cache(manifest["spec"], spec)
    # Keep an existing spec/fingerprint intact: count validation does not change
    # cached tensors, the sampling clock, or the weights used for extraction.
    validation_run = dict(extractor_sha256=spec["extractor_sha256"],
        checks_sha256=sha256(Path(__file__).parent/"extraction_checks.py"),
        frame_count_tolerance=args.frame_count_tolerance, sampling_changed=False)
    audit = manifest.setdefault("frame_count_validation_runs", [])
    if validation_run not in audit:
        audit.append(validation_run)
        if manifest_path.exists():
            write_json(manifest_path, manifest)
    # Recovery identity must also work before the first video is committed.
    signature = hashlib.sha256(json.dumps(manifest["spec"],sort_keys=True).encode()).hexdigest()
    # Only this parent process writes manifest.json. GPU workers write unique
    # per-video tensors/markers and send their completed entries to the parent.
    write_json(manifest_path,manifest)
    selected_by_video = {record["video_path"]:record for record in selected}
    def commit(video, entry):
        if video not in selected_by_video or entry["feature_id"] != feature_id(video):
            raise ValueError(f"Invalid extraction commit: {video}")
        manifest["videos"][video] = entry
        validate_entry(root,manifest,selected_by_video[video])
        write_json(manifest_path,manifest)
    cached_count, recovered, pending = 0, 0, []
    for record in tqdm(selected,desc="Checking feature cache",unit="video",
                       dynamic_ncols=True,disable=not args.show_progress):
        video = record["video_path"]
        marker_path = root/"video_commits"/(feature_id(video)+".json")
        if video not in manifest["videos"] and marker_path.exists():
            marker = read_json(marker_path)
            if marker["video_id"] != video or marker["cache_signature"] != signature:
                raise ValueError(f"Completed-video marker belongs to a different cache: {marker_path}")
            producer = dict(manifest["spec"],extractor_sha256=marker["entry"].get("extractor_sha256"))
            require_compatible_cache(producer,spec)
            commit(video,marker["entry"])
            recovered += 1
        if video in manifest["videos"]:
            entry = validate_entry(root,manifest,record)
            check_frame_count(video,float(record["duration"]),args.frame_fps,
                              entry["num_source_frames"],args.frame_count_tolerance)
            cached_count += 1
        else:
            pending.append(record)
    print("******* Extracting video features *******",flush=True)
    print(f"Config: {args.config or '(command-line arguments)'}")
    print(f"Source FPS={args.frame_fps:g}; window={args.window_seconds:g}s; "
          f"stride={args.stride_seconds:g}s; sampled frames={args.frames_per_window}")
    print(f"Feature cache: {root}")
    print(f"Extraction progress: {cached_count}/{len(selected)} videos cached; recovered={recovered}",flush=True)
    processes = processes or (len(args.device_ids) if args.device_ids else 1)
    with tqdm(total=len(selected),initial=cached_count,desc="Extract features",unit="video",
              dynamic_ncols=True,disable=not args.show_progress) as pbar:
        pbar.set_postfix(cached=cached_count)
        if pending and processes > 1:
            extract_parallel(args,pending,spec,signature,processes,commit,pbar,model_loader,clip_loader)
        elif pending:
            local_args = copy.copy(args)
            device = assigned_device(args.device,args.device_ids,0,distributed=False)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            local_args.device = str(device)
            extractor = PairedExtractor(local_args,spec,signature,model_loader,clip_loader)
            try:
                for record in pending:
                    entry = extractor.extract(record)
                    commit(record["video_path"],entry)
                    pbar.update(1)
                    pbar.set_postfix(cached=cached_count,windows=entry["num_windows"],
                                     video=Path(record["video_path"]).name)
            finally:
                extractor.close()
    if any(record["video_path"] not in manifest["videos"] for record in selected):
        raise RuntimeError("Incomplete extraction; rerun the same command to recover completed videos")
    print(f"[ok] Paired feature extraction finished: {manifest_path}")


def main():
    args = parse_args()
    if args.device_ids:
        validate_cuda_devices(args.device_ids)
    run_extraction(args)


if __name__ == "__main__":
    main()
