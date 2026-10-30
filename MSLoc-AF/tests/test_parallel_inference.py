"""Real CPU spawn/decode tests; disabling NMS here ONLY isolates orchestration.

Production eval.py always preflights the official C++ extension. These tests do
not claim to validate Soft-NMS when that binary is absent.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_contract import ROOT, fixture
import predict
from af_backend import PROTOCOL, build_model, model_config, require_official_nms
from common import read_json, sha256, write_json
from dataset import FeatureDataset
from engine import save_torch
from runtime_compat import LAUNCHER_SOURCE_TRANSITIONS


def exported_test_model(root, annotations):
    torch.manual_seed(9)
    dataset = FeatureDataset(annotations,root,"xclip")
    config = model_config(width=8,max_seq_len=32,attention_window=5,branches=2,input_dim=8)
    config["test_cfg"]["nms_method"] = "none"  # TEST ONLY; production configs remain Soft-NMS.
    model = build_model(8,config)
    torch.nn.init.constant_(model.cls_head.cls_head.conv.bias,1.)
    torch.nn.init.constant_(model.reg_head.offset_head.conv.bias,0.5)
    checkpoint = root/"test_only_no_nms.pt"
    sources = {name:sha256(ROOT/name) for name in (
        "af_backend.py","common.py","dataset.py","engine.py","metrics.py")}
    # Existing trained weights remain loadable after the import-path-only fix.
    sources["af_backend.py"] = LAUNCHER_SOURCE_TRANSITIONS["af_backend.py"][0]
    save_torch(checkpoint,dict(model=model.state_dict(),model_config=config,protocol=PROTOCOL,
        input_dim=8,variant="xclip",feature_signature=dataset.signature,epoch=1,score_threshold=0.5,
        selection="test_only",parameters=sum(p.numel() for p in model.parameters()),
        upstream_source_sha256=sha256(ROOT/"vendor/SOURCE.json"),
        metric_source_sha256=sha256(ROOT.parent/"scripts/evaluate_long.py"),
        source_sha256=sources))
    return checkpoint


def run_cpu_prediction(annotations, features, checkpoint, output, processes, resume=False):
    command = [sys.executable,str(Path(__file__).resolve()),"--cpu-inference",str(processes),
        "--annotations",str(annotations),"--features",str(features),"--checkpoint",str(checkpoint),
        "--output",str(output),"--device","cpu"]
    if resume:
        command.append("--resume")
    result = subprocess.run(command,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONIOENCODING="utf-8"),
        capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120)
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    return result.stdout+result.stderr


class ParallelInferenceTests(unittest.TestCase):
    def test_eval_device_ids_and_shards_cover_every_pending_video_once(self):
        argv = ["eval.py","--config",str(ROOT/"configs/ActionFormer_xclip.yaml"),
                "--device-ids","0,1,2,3,4,5,6,7","--resume"]
        with patch.object(sys,"argv",argv):
            args = predict.parse_args()
        self.assertEqual(args.device_ids,list(range(8)))
        self.assertTrue(args.resume)
        for pending in ([0,2,4],list(range(13))):
            shards = predict.inference_shards(pending,8)
            flattened = [index for shard in shards for index in shard]
            self.assertEqual(sorted(flattened),pending)
            self.assertEqual(len(flattened),len(set(flattened)))
            self.assertTrue(all(shards))

    def test_missing_nms_never_leaves_vendor_utils_ahead_of_local_metrics(self):
        original_path = sys.path.copy()
        with patch("af_backend.importlib.import_module",side_effect=ImportError("simulated missing binary")):
            with self.assertRaisesRegex(RuntimeError,"NMS is not built"):
                require_official_nms()
        self.assertEqual(sys.path,original_path)

    def test_parallel_decode_matches_single_process_and_resumes_across_process_counts(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="msloc_infer_") as temporary:
            root = Path(temporary)
            splits = fixture(root,per_kind=3)
            checkpoint = exported_test_model(root,splits["test"])
            # Exercise the native process launcher and the actual model's eval
            # forward/decode. C++ NMS is covered separately by official tests.
            single, parallel = root/"single",root/"parallel"
            run_cpu_prediction(splits["test"],root,checkpoint,single,1)
            logs = run_cpu_prediction(splits["test"],root,checkpoint,parallel,2)
            self.assertIn("Parallel inference: 2 workers",logs)
            self.assertIn("Testing",logs)
            self.assertEqual(len(list((parallel/"video_progress").glob("*.json"))),6)
            for name in ("raw_predictions.json","predictions.json","metrics.json"):
                self.assertEqual(read_json(parallel/name),read_json(single/name))
            raw = read_json(parallel/"raw_predictions.json")
            self.assertTrue(any(prediction["segments"] for prediction in raw.values()))
            files = sorted((parallel/"video_progress").glob("*.json"))
            preserved = {path:path.stat().st_mtime_ns for path in files[:4]}
            for path in files[4:]:
                path.unlink()
            # A cache from the previous single-GPU predictor can also resume.
            spec = read_json(parallel/"inference_config.json")
            spec["predictor_sha256"] = LAUNCHER_SOURCE_TRANSITIONS["predict.py"][0]
            write_json(parallel/"inference_config.json",spec)
            (parallel/"metrics.json").unlink()
            (parallel/"predictions.json").unlink()
            logs = run_cpu_prediction(splits["test"],root,checkpoint,parallel,1,resume=True)
            self.assertIn("4/6 videos already committed",logs)
            for path,mtime in preserved.items():
                self.assertEqual(path.stat().st_mtime_ns,mtime)
            self.assertEqual(read_json(parallel/"raw_predictions.json"),read_json(single/"raw_predictions.json"))
            self.assertEqual(read_json(parallel/"metrics.json"),read_json(single/"metrics.json"))
            mtime = (parallel/"metrics.json").stat().st_mtime_ns
            logs = run_cpu_prediction(splits["test"],root,checkpoint,parallel,2,resume=True)
            self.assertIn("Evaluation already complete",logs)
            self.assertEqual((parallel/"metrics.json").stat().st_mtime_ns,mtime)


if __name__ == "__main__":
    if "--cpu-inference" in sys.argv:
        position = sys.argv.index("--cpu-inference")
        processes = int(sys.argv[position+1])
        del sys.argv[position:position+2]
        # Explicitly scoped to this test driver and a TEST-ONLY no-NMS model.
        with patch.object(predict,"require_official_nms",return_value=None):
            predict.run_prediction(predict.parse_args(),processes=processes)
    else:
        unittest.main()
