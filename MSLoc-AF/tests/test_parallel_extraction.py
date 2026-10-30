"""Real spawned CPU workers with a synthetic PG; no real-weight GPU claim."""
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_contract import ROOT
import extract_features
from common import feature_id, feature_signature, read_json, write_json
from extraction_checks import LEGACY_EXTRACTOR_HASHES


class SyntheticMixer(torch.nn.Module):
    def forward(self, tokens):
        return tokens + 3


class SyntheticPG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = types.SimpleNamespace(config=types.SimpleNamespace(num_frames=8))
        self.mamba = SyntheticMixer()

    def forward(self, images):
        values = images[:,0,0,0,0]
        tokens = values[:,None,None,None].expand(-1,196,8,768).reshape(-1,8,768)
        self.mamba(tokens)
        return torch.zeros(len(images),4)


def synthetic_load_pg(checkpoint, weights, device):
    return SyntheticPG().to(device).eval()


def synthetic_read_clip(paths, crop_youku):
    first = paths[0]
    if (first.parent/"fail_clip").exists():
        raise RuntimeError("simulated interrupted clip")
    index = int(first.parent.name.removeprefix("v"))
    frame = int(first.stem.split("_")[-1])
    return torch.full((8,3,2,2),float(index*100+frame))


def extraction_fixture(root):
    records = []
    for index in range(6):
        duration = 4. if index % 2 == 0 else 2.
        records.append(dict(video_path=f"v{index}.mp4",type="fake",duration=duration,annotations=[]))
        folder = root/"frames"/f"v{index}"
        folder.mkdir(parents=True)
        for frame in range(1,int(duration*8)+1):
            (folder/f"frame_{frame}.jpg").touch()
    annotations = root/"annotations.json"
    write_json(annotations,records)
    (root/"checkpoint.pth").write_bytes(b"Synthetic PG weights for orchestration tests only")
    return annotations,records


def run_cpu_extraction(root, output, processes=2, batch_size=2, *extra, expect_failure=False):
    command = [sys.executable,str(Path(__file__).resolve()),"--cpu-extraction",str(processes),
        "--annotations",str(root/"annotations.json"),"--frames-root",str(root/"frames"),
        "--checkpoint",str(root/"checkpoint.pth"),"--xclip-weights",str(root/"weights"),
        "--output",str(output),"--device","cpu","--batch-size",str(batch_size),*map(str,extra)]
    result = subprocess.run(command,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONIOENCODING="utf-8"),
        capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120)
    if not expect_failure and result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    if expect_failure and not result.returncode:
        raise AssertionError("Expected simulated extraction interruption")
    return result.stdout+result.stderr


class ParallelExtractionTests(unittest.TestCase):
    def test_parse_gpu_ids_and_per_gpu_window_batch(self):
        argv = ["extract_features.py","--config",str(ROOT/"configs/features.yaml"),
                "--device-ids","3,1","--batch-size","4"]
        with patch.object(sys,"argv",argv):
            args = extract_features.parse_args()
        self.assertEqual(args.device_ids,[3,1])
        self.assertEqual(args.batch_size,4)
        self.assertEqual(args.frame_fps,8.)

    def test_spawned_paired_features_match_single_gpu_algorithm_and_legacy_cache_resumes(self):
        with tempfile.TemporaryDirectory(prefix="msloc_extract_") as temporary:
            root = Path(temporary)
            _,records = extraction_fixture(root)
            single,parallel = root/"single",root/"parallel"
            # Different window batch sizes also test per-window independence.
            run_cpu_extraction(root,single,1,1)
            logs = run_cpu_extraction(root,parallel,2,2)
            self.assertIn("Parallel extraction: 2 workers",logs)
            self.assertIn("Extract features",logs)
            one,two = read_json(single/"manifest.json"),read_json(parallel/"manifest.json")
            self.assertEqual(one["spec"],two["spec"])
            self.assertEqual(one["videos"],two["videos"])
            self.assertEqual(len(two["videos"]),6)
            for index,record in enumerate(records):
                identity = feature_id(record["video_path"])
                arrays = {}
                for variant in ("xclip","mamba"):
                    first = np.load(single/variant/(identity+".npy"))
                    second = np.load(parallel/variant/(identity+".npy"))
                    np.testing.assert_array_equal(first,second)
                    arrays[variant] = second
                expected = np.asarray([index*100+1+16*n for n in range(len(arrays["xclip"]))])
                np.testing.assert_array_equal(arrays["xclip"][:,0],expected)
                np.testing.assert_array_equal(arrays["mamba"],arrays["xclip"]+3)
            preserved = {path:path.stat().st_mtime_ns for variant in ("xclip","mamba")
                         for path in (parallel/variant).glob("*.npy")}
            # Existing single-GPU cache signatures remain unchanged. No model
            # load or tensor rewrite is needed just to enable more GPUs.
            two["spec"]["extractor_sha256"] = "8b229afd0700e23a11ad492fbb9c5c467a2d1f2795e5306ab41667d24ad8f95b"
            self.assertIn(two["spec"]["extractor_sha256"],LEGACY_EXTRACTOR_HASHES)
            signature = feature_signature(two)
            write_json(parallel/"manifest.json",two)
            logs = run_cpu_extraction(root,parallel,3,2)
            self.assertIn("6/6 videos cached",logs)
            self.assertNotIn("Parallel extraction:",logs)
            self.assertEqual(feature_signature(read_json(parallel/"manifest.json")),signature)
            for path,mtime in preserved.items():
                self.assertEqual(path.stat().st_mtime_ns,mtime)

    def test_recovers_durable_pair_after_parent_gap_and_after_worker_failure(self):
        with tempfile.TemporaryDirectory(prefix="msloc_recover_") as temporary:
            root = Path(temporary)
            _,records = extraction_fixture(root)
            output = root/"cache"
            # Complete v0/v1, then simulate the parent dying before publishing
            # v1 into manifest. Its durable paired commit must recover intact.
            run_cpu_extraction(root,output,1,2,"--max-videos","2")
            manifest = read_json(output/"manifest.json")
            manifest["videos"].pop("v1.mp4")
            write_json(output/"manifest.json",manifest)
            preserved = {path:path.stat().st_mtime_ns for variant in ("xclip","mamba")
                         for path in (output/variant).glob("*.npy")}
            failure = root/"frames/v2/fail_clip"
            failure.touch()
            logs = run_cpu_extraction(root,output,2,2,expect_failure=True)
            self.assertIn("simulated interrupted clip",logs)
            interrupted = read_json(output/"manifest.json")
            self.assertIn("v1.mp4",interrupted["videos"])
            self.assertNotIn("v2.mp4",interrupted["videos"])
            for path,mtime in preserved.items():
                self.assertEqual(path.stat().st_mtime_ns,mtime)
            failure.unlink()
            run_cpu_extraction(root,output,3,2)
            restored = read_json(output/"manifest.json")
            self.assertEqual(set(restored["videos"]),{r["video_path"] for r in records})
            self.assertEqual(len(list((output/"video_commits").glob("*.json"))),6)
            for path,mtime in preserved.items():
                self.assertEqual(path.stat().st_mtime_ns,mtime)
            # Paired markers must not recover tensors from another spec.
            restored["videos"].pop("v0.mp4")
            marker_path = output/"video_commits"/(feature_id("v0.mp4")+".json")
            marker = read_json(marker_path)
            marker["cache_signature"] = "different-cache"
            write_json(marker_path,marker)
            write_json(output/"manifest.json",restored)
            logs = run_cpu_extraction(root,output,2,2,expect_failure=True)
            self.assertIn("different cache",logs)


if __name__ == "__main__":
    if "--cpu-extraction" in sys.argv:
        position = sys.argv.index("--cpu-extraction")
        processes = int(sys.argv[position+1])
        del sys.argv[position:position+2]
        extract_features.run_extraction(extract_features.parse_args(),processes=processes,
            model_loader=synthetic_load_pg,clip_loader=synthetic_read_clip)
    else:
        unittest.main()
