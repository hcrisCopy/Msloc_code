import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from test_contract import ROOT, fixture
from common import read_json
from dataset import FeatureDataset
from af_backend import (minimum_padded_length, model_config, official_preset,
                        require_official_nms,official_modules,training_utilities)
from verify_official import verify_core, verify_nms


class OfficialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_upstream_training_step_and_ema_parity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = verify_core()
        self.assertTrue(report["model_and_one_step_parity"])

    def test_structure_defaults_and_data_specific_padding(self):
        source = official_preset()["model"]
        actual = model_config(input_dim=150528)
        for name in ("backbone_arch","regression_range","n_mha_win_size","fpn_dim","head_dim"):
            self.assertEqual(json.loads(json.dumps(actual[name])),json.loads(json.dumps(source[name])))
        self.assertEqual(actual,json.loads(json.dumps(actual)))  # Resume config must be JSON-stable.
        self.assertEqual(actual["input_dim"],[150528])
        self.assertEqual(actual["embd_dim"],[512])
        self.assertEqual(actual["num_classes"],1)
        self.assertEqual(minimum_padded_length(62),576)
        self.assertEqual(minimum_padded_length(248),576)
        self.assertEqual(minimum_padded_length(600),1152)

    def test_official_soft_nms_binary(self):
        try:
            require_official_nms()
        except RuntimeError as error:
            self.skipTest(str(error))
        self.assertTrue(verify_nms()["official_soft_nms"])

    def test_actual_cli_update_matches_original_train_one_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            splits = fixture(root)
            command = [sys.executable,str(ROOT/"train.py"),"--train-json",str(splits["train"]),
                "--features",str(root),"--variant","xclip","--output",str(root/"run"),
                "--batch-size","2","--epochs","1","--warmup-epochs","0",
                "--max-updates-this-run","1","--width","8","--max-seq-len","32",
                "--attention-window","5","--branches","2","--device","cpu","--no-progress"]
            result = subprocess.run(command,capture_output=True,text=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            actual = torch.load(root/"run/last.pt",map_location="cpu",weights_only=False)
            config = read_json(root/"run/run_config.json")
            utilities = training_utilities()
            utilities.fix_random_seed(42,include_cuda=torch.cuda.is_available())
            modeling,_ = official_modules()
            reference = modeling.make_meta_arch("LocPointTransformer",**config["model_config"])
            optimizer = utilities.make_optimizer(reference,config["optimizer_config"])
            scheduler = utilities.make_scheduler(optimizer,config["optimizer_config"],1)
            ema = utilities.ModelEma(reference)
            dataset = FeatureDataset(splits["train"],root,"xclip",is_training=True,max_seq_len=32)
            order = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(42)).tolist()
            batch = [dataset[i] for i in order]
            with contextlib.redirect_stdout(io.StringIO()):
                utilities.train_one_epoch([batch],reference,optimizer,scheduler,0,model_ema=ema,
                                          clip_grad_l2norm=1.0,print_freq=100)
            for current,expected in ((actual["model"],reference.state_dict()),(actual["ema"],ema.module.state_dict())):
                for name,value in expected.items():
                    torch.testing.assert_close(current[name],value,rtol=1e-6,atol=1e-7)
            self.assertEqual(actual["optimizer"]["param_groups"][0]["lr"],optimizer.param_groups[0]["lr"])


if __name__ == "__main__":
    unittest.main()
