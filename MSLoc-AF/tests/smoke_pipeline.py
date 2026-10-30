"""End-to-end synthetic CLI smoke test. Does not claim real-data model accuracy."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from test_contract import ROOT, fixture
from common import feature_id, read_json, sha256
from subset import prepare_subset
from runtime_compat import LEGACY_SOURCE_HASHES
from af_backend import require_official_nms, PROTOCOL


def run(script, *args, markers=()):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(ROOT/script), *map(str,args)],env=env,
                            capture_output=True,text=True,encoding="utf-8",errors="replace")
    if result.returncode:
        print(result.stdout,result.stderr)
        result.check_returncode()
    for marker in markers:
        assert marker in result.stdout+result.stderr, (script,marker,result.stdout,result.stderr)
    print(f"PASS {script}: {', '.join(markers) or 'completed'}",flush=True)
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="msloc_af_smoke_") as temporary:
        root = Path(temporary)
        splits = fixture(root,per_kind=4)
        val_records = read_json(splits["val"])
        splits["val"].unlink()  # The default pipeline must work with no validation file at all.
        sources = {name:splits[name] for name in ("train","test")}
        run("prepare_subset.py","--train-json",sources["train"],"--test-json",sources["test"],
            "--output",root/"subset","--max-videos",10)
        subset = {name:root/"subset"/f"{name}.json" for name in ("train","test")}
        configs = {}
        for variant in ("xclip","mamba","fusion"):
            config = dict(variant=variant,train_json_path=str(subset["train"]),
                test_json_path=str(subset["test"]),features_path=str(root),save_dir=str(root/"runs"/variant),
                score_threshold=0.37,max_epoch=1,train_batch_size=1,
                warmup_epochs=0,width=8,max_seq_len=32,attention_window=5,branches=2,device="cpu")
            configs[variant] = root/f"{variant}.yaml"
            configs[variant].write_text(yaml.safe_dump(config),encoding="utf-8")
        direct = ["--config",configs["xclip"]]
        run("train.py",*direct,"--max-updates-this-run",1,markers=("Training epoch 1/1","Saved after"))
        paused = torch.load(root/"runs/xclip/last.pt",map_location="cpu",weights_only=False)
        assert paused["epoch"] == 0 and paused["batch_in_epoch"] == 1
        from common import write_json
        run("train.py",*direct,"--resume",markers=("Restored:","Train 1/1","loss=","lr="))
        for variant in ("mamba","fusion"):
            run("train.py","--config",configs[variant],markers=("Train 1/1","cls_loss=","reg_loss="))
        run("train.py",*direct,"--output",root/"reference")
        restored = torch.load(root/"runs/xclip/last.pt",map_location="cpu",weights_only=False)
        reference = torch.load(root/"reference/last.pt",map_location="cpu",weights_only=False)
        assert restored["global_updates"] == reference["global_updates"] == 8
        torch.testing.assert_close(restored["torch_rng"],reference["torch_rng"],rtol=0,atol=0)
        for name in ("model","ema"):
            for key,value in reference[name].items():
                torch.testing.assert_close(restored[name][key],value,rtol=1e-6,atol=1e-7)
        for variant in ("xclip","mamba","fusion"):
            folder = root/"runs"/variant
            final = torch.load(folder/"final.pt",map_location="cpu",weights_only=False)
            assert final["selection"] == "last_epoch_ema" and final["score_threshold"] == 0.37
            assert final["epoch"] == 1 and final["validation_metrics"] is None
            assert final["protocol"] == PROTOCOL
            assert not (folder/"best.pt").exists()
            assert not (folder/"val_predictions_best.json").exists()
            assert set(read_json(folder/"supervision_report.json")) == {"train"}
            assert all("validation" not in entry for entry in read_json(folder/"history.json"))
        try:
            require_official_nms()
        except RuntimeError as error:
            print(f"PASS: three official-module CLI training/resume checks. SKIP inference/Soft-NMS: {error}")
            return
        def evaluate(variant,markers=()):
            run("eval.py","--config",configs[variant],"--model_path",root/"runs"/variant/"final.pt",
                "--output_dir",root/"runs"/variant/"test","--resume",markers=markers)
        for variant in configs:
            evaluate(variant,markers=("Temporal-segmentation model evaluation","Testing", "Classification accuracy:",
                                     "Avg Loc F1:","[ok] Evaluation pipeline finished."))
        output = root/"runs/xclip/test"
        old_predictions = read_json(output/"predictions.json")
        progress = output/"video_progress"
        complete = list(progress.glob("*.json"))
        preserved_mtime = complete[0].stat().st_mtime_ns
        complete[1].unlink()
        (output/"metrics.json").unlink()
        evaluate("xclip",markers=("1/2 videos already committed","Testing"))
        assert read_json(output/"predictions.json") == old_predictions
        assert complete[0].stat().st_mtime_ns == preserved_mtime
        last_hashes = {v:sha256(root/"runs"/v/"last.pt") for v in ("xclip","mamba","fusion")}
        metric_mtimes = {v:(root/"runs"/v/"test/metrics.json").stat().st_mtime_ns for v in last_hashes}
        for variant in configs:
            run("train.py","--config",configs[variant],"--resume",markers=("Training already complete",))
            evaluate(variant,markers=("Evaluation already complete",))
        for variant in last_hashes:
            assert sha256(root/"runs"/variant/"last.pt") == last_hashes[variant]
            assert (root/"runs"/variant/"test/metrics.json").stat().st_mtime_ns == metric_mtimes[variant]
        run("summarize.py","--run-root",root/"runs")
        assert (root/"runs/comparison.csv").is_file()
        # Explicit validation remains supported, but is never required by default.
        write_json(splits["val"],val_records)
        run("train.py",*direct,"--output",root/"optional_val","--val-json",splits["val"],
            "--thresholds",0.05,0.5)
        selected = torch.load(root/"optional_val/best.pt",map_location="cpu",weights_only=False)
        assert selected["selection"] == "validation_best_ema"
        assert not (root/"optional_val/final.pt").exists()
        print("PASS: separate YAML-driven official-module train/eval; source-style progress/logs; "
              "no-val 8/2 protocol; mid-epoch resume; partial inference resume; optional val works")


if __name__ == "__main__":
    main()
