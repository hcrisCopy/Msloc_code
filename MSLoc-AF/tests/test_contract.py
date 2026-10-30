"""Checks for timing, real-video targets, feature pairing and projection gradients."""
import ast
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from af_backend import batched_nms, build_model, model_config, require_official_nms
from common import (feature_id, fake_segments, grid_to_seconds, load_features,
                    make_windows, sample_indices, seconds_to_grid, write_json)
from dataset import FeatureDataset
from extract_features import aggregate_tokens


def fixture(folder, per_kind=1):
    root = Path(folder)
    spec = dict(frame_fps=8.0, window_seconds=2.0, stride_seconds=2.0,
                frames_per_window=8, feature_dim=8, storage_dtype="float16",
                aggregation="legacy_mamba_sequence_mean_then_flatten_no_global",
                checkpoint_sha256="synthetic_test_only")
    manifest = dict(schema_version=1, spec=spec, videos={})
    splits = {}
    for split in ("train", "val", "test"):
        records = []
        for sample in range(2*per_kind):
            kind = "real" if sample < per_kind else "fake"
            suffix = "" if per_kind == 1 else str(sample % per_kind)
            video = f"{split}/{kind}{suffix}.mp4"
            record = dict(video_path=video, type=kind, duration=8.0,
                          tool_domain="fixture", annotations=[] if kind == "real" else [
                              dict(segment=[0.5, 5.5], segment_label="fake")])
            records.append(record)
            ident = feature_id(video)
            manifest["videos"][video] = dict(feature_id=ident, duration=8.0, num_windows=4)
            for name, offset in (("xclip", 0), ("mamba", 1)):
                (root / name).mkdir(exist_ok=True)
                values = np.random.default_rng(7).normal(size=(4, 8)) + offset
                np.save(root / name / (ident+".npy"), values.astype(np.float16))
        splits[split] = root / (split+".json")
        write_json(splits[split], records)
    write_json(root / "manifest.json", manifest)
    return splits


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_physical_frame_clock_and_round_trip(self):
        segments = [[0, 0.5], [1.25, 3.75], [9, 10.25]]
        grid = seconds_to_grid(segments, 0.5, 2)
        np.testing.assert_allclose(grid_to_seconds(grid, 0.5, 2, 10.25), segments)
        # 8 sampled frames are NOT 8 physical frames: 2 seconds at 8 fps is 16.
        self.assertEqual(float(seconds_to_grid([[1, 3]], 2, 2)[0, 0]), 0)
        windows = make_windows(4.25, 2, 2)
        self.assertEqual(windows, [(0, 2), (2, 4), (4, 4.25)])
        self.assertEqual(sample_indices(0, 2, 8, 40, 8), list(range(0, 16, 2)))
        self.assertEqual(sample_indices(4, 4.25, 8, 34, 8), [32,33,33,33,33,33,33,33])

    def test_original_forward_hook_and_classification_feature_agree(self):
        # Execute the ORIGINAL forward with a stub encoder, avoiding missing GPU
        # weights/dependencies. This detects incorrect interpretation of its layout.
        path = ROOT.parent / "MSLoc-PG/models/DeMamba.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "XCLIP_DeMamba")
        forward = copy.deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
        functions = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"create_reorder_index", "reorder_data"}]
        namespace = dict(torch=torch, math=math)
        module = ast.fix_missing_locations(ast.Module(body=functions+[forward], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
        class Encoder(torch.nn.Module):
            def forward(self, images, output_hidden_states):
                n = len(images)
                return dict(last_hidden_state=torch.arange(n*197*2).reshape(n,197,2).float(),
                            pooler_output=torch.zeros(n,2))
        class Mixer(torch.nn.Module):
            def forward(self, x):
                return x+10
        class PG(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.mamba = Mixer()
                self.fc_norm = self.fc_norm2 = self.fc1 = self.dropout = torch.nn.Identity()
                self.fusing_ratios = 1
            forward = namespace["forward"]
        model = PG()
        captured = {}
        def hook(module, args, output):
            captured["x"] = args[0]
            captured["h"] = output
        handle = model.mamba.register_forward_hook(hook)
        output = model(torch.zeros(2,8,3,2,2))
        handle.remove()
        self.assertEqual(captured["x"].shape, (2*196,8,2))
        torch.testing.assert_close(output[:,2:], aggregate_tokens(captured["h"],2))
        torch.testing.assert_close(aggregate_tokens(captured["h"],2), aggregate_tokens(captured["x"],2)+10)
        # Preserve and explicitly characterize the existing ordering.
        source = model.encoder(torch.zeros(16,3,2,2), True)["last_hidden_state"][:,1:]
        torch.testing.assert_close(captured["x"][0], source[0,:8])

    def test_dataset_pairing_empty_real_targets_and_fusion(self):
        with tempfile.TemporaryDirectory() as folder:
            splits = fixture(folder)
            x = FeatureDataset(splits["train"], folder, "xclip")
            h = FeatureDataset(splits["train"], folder, "mamba")
            fused = FeatureDataset(splits["train"], folder, "fusion")
            self.assertEqual(x[0]["segments"].shape, (0,2))
            self.assertEqual(x[0]["labels"].shape, (0,))
            self.assertEqual(x[0]["feat_num_frames"], 16)
            torch.testing.assert_close(fused[0]["feats"], torch.cat([x[0]["feats"],h[0]["feats"]]))
            self.assertEqual(fake_segments(dict(type="fake",duration=3,annotations=[])), [[0.,3.]])

    def test_official_model_backward_including_projection_for_all_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            splits = fixture(folder)
            for variant in ("xclip", "mamba", "fusion"):
                dataset = FeatureDataset(splits["train"], folder, variant)
                config = model_config(width=8,max_seq_len=32,attention_window=5,branches=2)
                model = build_model(dataset.input_dim, config)
                model.train()
                # A completely negative batch must also support backward.
                negative = model([dataset[0]])
                self.assertEqual(float(negative["reg_loss"]), 0)
                negative["final_loss"].backward()
                model.zero_grad(set_to_none=True)
                losses = model([dataset[0],dataset[1]])
                self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
                losses["final_loss"].backward()
                self.assertGreater(float(model.backbone.proj[0].conv.weight.grad.abs().sum()),0)
                model.eval()
                model.test_nms_method = "none"  # This test covers decoding, not the compiled NMS.
                prediction = model([dataset[1]])[0]
                self.assertTrue(torch.isfinite(prediction["segments"]).all())
                self.assertTrue((prediction["segments"] >= 0).all())
                self.assertTrue((prediction["segments"] <= 8).all())

    def test_continuous_hard_nms_and_empty(self):
        try:
            require_official_nms()
        except RuntimeError as error:
            self.skipTest(str(error))
        segments = torch.tensor([[0.,2.],[0.,2.],[3.,4.]])
        scores = torch.tensor([0.9,0.8,0.7])
        labels = torch.zeros(3,dtype=torch.long)
        out, kept, _ = batched_nms(segments,scores,labels,0.5,0.1,100,use_soft_nms=False,voting_thresh=0)
        torch.testing.assert_close(kept, torch.tensor([0.9,0.7]))
        self.assertEqual(len(out),2)
        self.assertEqual(len(batched_nms(segments[:0],scores[:0],labels[:0],0.5,0.1,100)[0]),0)

    def test_official_postprocessing_uses_physical_clip_extent(self):
        model = build_model(8,model_config(width=8,max_seq_len=32,attention_window=5,branches=2))
        model.test_nms_method = "none"
        prediction = dict(video_id="v",segments=torch.tensor([[-0.5,0.25],[1.,2.]]),
                          scores=torch.tensor([0.9,0.8]),labels=torch.zeros(2,dtype=torch.long),
                          fps=8.,duration=8.,feat_stride=16.,feat_num_frames=16.)
        out = model.postprocessing([prediction])[0]
        torch.testing.assert_close(out["segments"],torch.tensor([[0.,1.5],[3.,5.]]))


if __name__ == "__main__":
    unittest.main()
