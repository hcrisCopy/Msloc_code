"""Regression checks for fixed small-data selection and extraction interruption."""
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_contract import ROOT, fixture
from common import feature_signature, load_records, read_json, write_json
from extraction_checks import LEGACY_EXTRACTOR_HASHES
from extract_features import main as extract_main
from subset import prepare_subset


class RestartTests(unittest.TestCase):
    def test_train_test_subset_never_requires_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            splits = fixture(root, per_kind=4)
            splits["val"].unlink()
            sources = {name:splits[name] for name in ("train","test")}
            chosen = prepare_subset(sources,root/"subset",10,42)
            self.assertEqual(set(chosen),{"train","test"})
            self.assertEqual([len(load_records(chosen[n])) for n in ("train","test")],[8,2])
            self.assertFalse((root/"subset/val.json").exists())
            self.assertEqual(read_json(root/"subset/subset.json")["counts"],{"train":8,"test":2})

    def test_ten_means_total_and_subset_is_fixed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            splits = fixture(root, per_kind=3)
            first = prepare_subset(splits,root/"subset",10,42)
            mtimes = {name:path.stat().st_mtime_ns for name,path in first.items()}
            second = prepare_subset(splits,root/"subset",10,42)
            self.assertEqual([len(load_records(second[name])) for name in ("train","val","test")],[6,2,2])
            self.assertEqual(mtimes,{name:path.stat().st_mtime_ns for name,path in second.items()})
            ids = []
            for path in second.values():
                records = load_records(path)
                self.assertEqual({r["type"] for r in records},{"real","fake"})
                ids += [r["video_path"] for r in records]
            self.assertEqual(len(ids),len(set(ids)))
            with self.assertRaises(ValueError):
                prepare_subset(splits,root/"subset",10,7)

    def test_extraction_commits_per_video_and_resumes_without_rewriting(self):
        torch.set_num_threads(1)
        class Mixer(torch.nn.Module):
            def forward(self,x):
                return x+1
        class PG(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mamba = Mixer()
                self.encoder = types.SimpleNamespace(config=types.SimpleNamespace(num_frames=8))
            def forward(self,images):
                b = len(images)
                self.mamba(torch.zeros(b*196,8,768))
                return torch.zeros(b,4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [dict(video_path=f"v{i}.mp4",type="fake",duration=2.,annotations=[]) for i in range(2)]
            anno = root/"data.json"
            write_json(anno,records)
            for i in range(2):
                path = root/"frames"/f"v{i}"
                path.mkdir(parents=True)
                for n in range(1,17):
                    (path/f"frame_{n}.jpg").touch()
            weights = root/"stub.pth"
            weights.write_bytes(b"mock PG checkpoint for cache contract test")
            argv = ["extract_features.py","--annotations",str(anno),"--frames-root",str(root/"frames"),
                    "--checkpoint",str(weights),"--xclip-weights",str(root/"weights"),
                    "--output",str(root/"cache"),"--device","cpu","--batch-size","1","--no-progress"]
            dummy = torch.zeros(8,3,2,2)
            with patch.object(sys,"argv",argv), patch("extract_features.load_pg",return_value=PG()), \
                    patch("extract_features.read_clip",side_effect=[dummy,RuntimeError("simulated interruption")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                    extract_main()
            manifest = read_json(root/"cache/manifest.json")
            self.assertEqual(list(manifest["videos"]),["v0.mp4"])
            # Simulate the real cache created before this validation-only patch.
            manifest["spec"]["extractor_sha256"] = sorted(LEGACY_EXTRACTOR_HASHES)[0]
            manifest["spec"]["frame_count_tolerance"] = 2
            manifest.pop("frame_count_validation_runs",None)
            manifest["videos"]["v0.mp4"].pop("frame_count_check",None)
            manifest["videos"]["v0.mp4"].pop("extractor_sha256",None)
            old_signature = feature_signature(manifest)
            write_json(root/"cache/manifest.json",manifest)
            first = next((root/"cache/xclip").glob("*.npy"))
            mtime = first.stat().st_mtime_ns
            with patch.object(sys,"argv",argv), patch("extract_features.load_pg",return_value=PG()), \
                    patch("extract_features.read_clip",return_value=dummy),contextlib.redirect_stdout(io.StringIO()):
                extract_main()
            restored = read_json(root/"cache/manifest.json")
            self.assertEqual(len(restored["videos"]),2)
            self.assertEqual(feature_signature(restored),old_signature)
            self.assertEqual(restored["spec"],manifest["spec"])
            self.assertEqual(restored["frame_count_validation_runs"][0]["frame_count_tolerance"],4)
            self.assertEqual(first.stat().st_mtime_ns,mtime)
            with patch.object(sys,"argv",argv), patch("extract_features.load_pg",side_effect=AssertionError("must skip")), \
                    contextlib.redirect_stdout(io.StringIO()):
                extract_main()
            self.assertEqual(first.stat().st_mtime_ns,mtime)


if __name__ == "__main__":
    unittest.main()
