"""Select patch neurons with the existing paired effect and XCLIP clip grouping."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.UniFormerV2_DeMamba import BACKBONE, FEATURE_LOCATION, MEAN, STD, load_encoder
from probe_dinov3_neurons import RunningVectorStats, group_pairs, load_progress, save_progress
from probe_xclip_neurons import read_frame_pairs
from train import parse_device_ids


def load_frame(root, pair, real, crop_youku):
    video_key = "normal_video" if real else "fake_video"
    frame_key = "normal_frame_file" if real else "fake_frame_file"
    path = root / Path(pair[video_key]).with_suffix("") / pair[frame_key]
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unreadable paired frame: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_youku and "youku" in str(path).lower():
        height, width = image.shape[:2]
        image = (image[:, int(width * .15):int(width * .85)] if width > height
                 else image[int(height * .15):int(height * .85), :])
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    image = (image - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def signature(args):
    value = {"backbone": BACKBONE, "frame_root": str(args.frame_root.resolve()),
             "crop_youku": args.crop_youku, "amp": args.amp,
             "image_batch_size": args.image_batch_size,
             "max_frame_pairs": args.max_frame_pairs,
             "frames_per_clip": 8,
             "feature_location": FEATURE_LOCATION}
    for key in ("pairs", "uniformerv2_weights_path"):
        path = getattr(args, key)
        stat = path.stat()
        value[key] = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return json.dumps(value, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--uniformerv2-weights-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-ids", type=parse_device_ids, default=list(range(8)))
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--max-frame-pairs", type=int, default=0)
    parser.add_argument("--final-neuron-count", type=int, default=768)
    parser.add_argument("--crop-youku", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Fail on missing or unreadable frames (always enabled)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.image_batch_size < 1 or args.max_frame_pairs < 0 or args.final_neuron_count != 768:
        parser.error("image-batch-size must be positive, max-frame-pairs non-negative and final-neuron-count 768")
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "uniformerv2_neuron_scores.npz"
    selector_path = args.output_dir / "uniformerv2_neuron_indices.json"
    progress_path = args.output_dir / "uniformerv2_probe_progress.npz"
    if args.clean:
        for path in (score_path, selector_path, progress_path):
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
    run_signature = signature(args)
    if args.resume and selector_path.is_file() and score_path.is_file():
        selector = json.loads(selector_path.read_text(encoding="utf-8"))
        if selector["signature"] != run_signature:
            raise ValueError("Completed probe belongs to different inputs; use --clean")
        print(f"[reuse] Completed probe: {selector_path}")
        return
    if not torch.cuda.is_available() or max(args.device_ids) >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA devices are unavailable: {args.device_ids}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.set_device(args.device_ids[0])
    device = torch.device(f"cuda:{args.device_ids[0]}")
    pairs = read_frame_pairs(args.pairs)
    if args.max_frame_pairs:
        pairs = pairs[:args.max_frame_pairs]
    groups = group_pairs(pairs)
    stats = {layer: RunningVectorStats(768) for layer in range(1, 13)}
    start, frames, videos = 0, 0, 0
    if args.resume and progress_path.is_file():
        start, stats, failures, frames, videos = load_progress(progress_path, run_signature)
        if failures:
            raise ValueError("Strict UniFormerV2 probe cannot resume a run containing skipped pairs")
    encoder = load_encoder(args.uniformerv2_weights_path).to(device).eval().requires_grad_(False)
    if len(args.device_ids) > 1:
        encoder = torch.nn.DataParallel(encoder, device_ids=args.device_ids)
    with tqdm(total=len(groups), initial=start, desc="Probing UniFormerV2", unit="video") as bar:
        for group_index in range(start, len(groups)):
            group = groups[group_index]
            sums = {layer: np.zeros(768, np.float64) for layer in stats}
            for first in range(0, len(group), args.image_batch_size):
                rows = group[first:first + args.image_batch_size]
                clips, counts = [], []
                for offset in range(0, len(rows), 8):
                    chunk = rows[offset:offset + 8]
                    counts.append(len(chunk))
                    for real in (False, True):
                        images = [load_frame(args.frame_root, pair, real, args.crop_youku) for pair in chunk]
                        # Match XCLIP: pad each fake/real sequence separately.
                        images.extend([images[-1]] * (8 - len(images)))
                        clips.append(torch.stack(images, dim=1))
                batch = torch.stack(clips).to(device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    _, states = encoder(batch, return_features=True)
                if len(states) != 12:
                    raise RuntimeError("UniFormerV2 did not return all 12 patch states")
                for layer, state in enumerate(states, 1):
                    if state.shape != (2 * len(counts), 8, 196, 768):
                        raise RuntimeError(f"Invalid block {layer} shape: {tuple(state.shape)}")
                    activation = state.float().mean(dim=2)
                    for index, count in enumerate(counts):
                        # Duplicated padding positions never enter the statistic.
                        delta = activation[2 * index, :count] - activation[2 * index + 1, :count]
                        sums[layer] += delta.sum(dim=0).cpu().numpy()
            for layer in stats:
                stats[layer].update(sums[layer] / len(group))
            frames += len(group)
            videos += 1
            save_progress(progress_path, run_signature, group_index + 1, stats, [], frames, videos)
            bar.update(1)
            bar.set_postfix(pairs=frames)
    if videos < 2:
        raise ValueError("At least two source videos are required for paired-effect selection")
    arrays, layers = {}, {}
    for layer, state in stats.items():
        signed = state.signed_effect()
        score = np.abs(signed)
        arrays[f"frame_layer_{layer:02d}_score"] = score.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_signed_effect"] = signed.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_mean_delta"] = state.mean.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_video_count"] = np.asarray(state.count, dtype=np.int32)
        layers[str(layer)] = sorted(np.argsort(score)[::-1][:64].tolist())
    temporary = score_path.with_name(score_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(score_path)
    selector = {"schema_version": 1, "backbone": BACKBONE,
                "feature_location": FEATURE_LOCATION, "excluded_tokens": ["CLS"],
                "selection_unit": "per_video_mean_of_paired_fake_minus_real_frames",
                "selection_rule": "top absolute paired sensitivity within each layer",
                "final_neuron_count": 768, "neurons_per_layer": 64,
                "num_hidden_layers": 12, "hidden_size": 768, "image_size": 224,
                "patch_size": 16, "frames_per_window": 8, "layers": layers,
                "processed_frame_pairs": frames, "processed_videos": videos,
                "signature": run_signature}
    temporary = selector_path.with_name(selector_path.name + ".tmp")
    temporary.write_text(json.dumps(selector, indent=2), encoding="utf-8")
    temporary.replace(selector_path)
    progress_path.unlink(missing_ok=True)
    print(f"[ok] Selected 768 neurons from {frames} pairs / {videos} videos: {selector_path}")


if __name__ == "__main__":
    main()
