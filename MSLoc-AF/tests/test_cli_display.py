import argparse
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import yaml

from test_contract import ROOT
from cli_config import parse_stage_args
from common import sha256
from display import print_evaluation_metrics
from runtime_compat import (LEGACY_SOURCE_HASHES, PRESENTATION_SOURCE_HASHES,
                            require_compatible_inference, require_compatible_run,
                            same_computation_source)


class CLIDisplayTests(unittest.TestCase):
    def test_three_configs_keep_identical_training_parameters(self):
        configs = []
        for variant in ("xclip","mamba","fusion"):
            config = yaml.safe_load((ROOT/"configs"/f"ActionFormer_{variant}.yaml").read_text())
            self.assertEqual(config.pop("variant"),variant)
            self.assertTrue(config.pop("save_dir").endswith("/"+variant))
            configs.append(config)
        self.assertEqual(configs[0],configs[1])
        self.assertEqual(configs[0],configs[2])
        parser = argparse.ArgumentParser()
        parser.add_argument("--train-json",required=True)
        parser.add_argument("--epochs",type=int,default=30)
        parser.add_argument("--variant",choices=["xclip","mamba","fusion"],required=True)
        parser.add_argument("--output",required=True)
        with patch.object(sys,"argv",["train.py","--config",str(ROOT/"configs/ActionFormer_mamba.yaml"),
                                      "--epochs","1","--no-progress"]):
            args = parse_stage_args(parser,"train")
        self.assertEqual(args.epochs,1)
        self.assertEqual(args.variant,"mamba")
        self.assertFalse(args.show_progress)
        self.assertTrue(args.train_json.endswith("train_all_1209.json"))

    def test_presentation_compatibility_is_explicit_and_rejects_computation_changes(self):
        for name,hashes in PRESENTATION_SOURCE_HASHES.items():
            current = sha256(ROOT/name)
            old = sorted(LEGACY_SOURCE_HASHES[name])[0]
            self.assertEqual(same_computation_source(name,old,current),current in hashes)
            self.assertFalse(same_computation_source(name,"unknown",current))
            self.assertFalse(same_computation_source(name,old,"changed_computation"))
        current = dict(lr=1e-4,source_sha256={n:sorted(h)[0] for n,h in PRESENTATION_SOURCE_HASHES.items()})
        old = dict(lr=1e-4,source_sha256={n:sorted(LEGACY_SOURCE_HASHES[n])[0] for n in PRESENTATION_SOURCE_HASHES})
        require_compatible_run(old,current)
        with self.assertRaises(ValueError):
            require_compatible_run(old,dict(current,lr=2e-4))
        require_compatible_inference(dict(checkpoint_sha256="A",predictor_sha256=sorted(LEGACY_SOURCE_HASHES["predict.py"])[0]),
            dict(checkpoint_sha256="A",predictor_sha256=sorted(PRESENTATION_SOURCE_HASHES["predict.py"])[0]))
        with self.assertRaises(ValueError):
            require_compatible_inference(dict(checkpoint_sha256="B",predictor_sha256=sorted(LEGACY_SOURCE_HASHES["predict.py"])[0]),
                dict(checkpoint_sha256="A",predictor_sha256=sha256(ROOT/"predict.py")))

    def test_report_uses_source_style_and_honest_metric_labels(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_evaluation_metrics(dict(Total=dict(Loc_F1=0.25,Loc_IoU=0.5,Det_Acc=0.75)),
                                     [dict(type="real"),dict(type="fake")])
        text = output.getvalue()
        for marker in ("Evaluation summary","Temporal segmentation evaluation","F1: 0.2500",
                       "Classification accuracy: 0.7500","Avg Loc F1: 0.5000","Total: 2  Fake: 1  Real: 1"):
            self.assertIn(marker,text)
        self.assertNotIn("fake videos only",text)


if __name__ == "__main__":
    unittest.main()
