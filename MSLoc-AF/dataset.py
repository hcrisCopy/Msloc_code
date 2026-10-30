"""Video-level ActionFormer dataset retaining authentic videos as empty targets."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from common import (check_manifest, fake_segments, feature_signature, load_features,
                    load_records, read_json, seconds_to_grid, validate_entry)
from af_backend import data_utilities, official_preset


class FeatureDataset(Dataset):
    def __init__(self, annotations, features, variant, show_progress=False,is_training=False,
                 max_seq_len=None):
        self.annotations = str(Path(annotations).resolve())
        self.root = Path(features)
        self.records = load_records(annotations)
        self.manifest = read_json(self.root / "manifest.json")
        check_manifest(self.manifest)
        self.signature = feature_signature(self.manifest)
        self.variant = variant
        self.spec = self.manifest["spec"]
        self.is_training = is_training
        self.official_data_config = official_preset()["dataset"]
        self.max_seq_len = max_seq_len or self.official_data_config["max_seq_len"]
        self.input_dim = self.spec["feature_dim"] * (2 if variant == "fusion" else 1)
        self.entries = [validate_entry(self.root, self.manifest, r) for r in
                        tqdm(self.records,desc="Preparing video features",unit="video",disable=not show_progress)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record, entry = self.records[index], self.entries[index]
        features = load_features(self.root, entry, self.variant)
        segments = fake_segments(record)
        grid = seconds_to_grid(segments, self.spec["stride_seconds"], self.spec["window_seconds"])
        fps = self.spec["frame_fps"]
        data = dict(video_id=record["video_path"],
                    feats=torch.from_numpy(np.ascontiguousarray(features.T)),
                    segments=torch.from_numpy(grid),
                    labels=torch.zeros(len(segments), dtype=torch.long),
                    fps=fps, duration=float(record["duration"]),
                    feat_stride=fps*self.spec["stride_seconds"],
                    # IMPORTANT: physical clip extent in the source-frame clock,
                    # NOT the number of sampled frames (8).
                    feat_num_frames=fps*self.spec["window_seconds"])
        if self.is_training:
            # Exact upstream crop implementation; real videos are a required TASLE
            # extension and have no action to retain, so disable has_action there.
            data = data_utilities().truncate_feats(data,self.max_seq_len,
                self.official_data_config["trunc_thresh"],
                0.5*data["feat_num_frames"]/data["feat_stride"],
                self.official_data_config["crop_ratio"],has_action=bool(len(segments)))
        return data

    def supervision_report(self):
        uncovered = []
        segment_count = 0
        for record, entry in zip(self.records, self.entries):
            centers = np.arange(entry["num_windows"]) * self.spec["stride_seconds"]
            centers += self.spec["window_seconds"] / 2
            for segment in fake_segments(record):
                segment_count += 1
                if not ((centers > segment[0]) & (centers < segment[1])).any():
                    uncovered.append(dict(video_path=record["video_path"], segment=segment))
        return dict(videos=len(self), real_videos=sum(r["type"] == "real" for r in self.records),
                    fake_segments=segment_count, segments_without_finest_grid_center=uncovered,
                    max_windows=max(e["num_windows"] for e in self.entries))


def collate_videos(batch):
    return data_utilities().trivial_batch_collator(batch)
