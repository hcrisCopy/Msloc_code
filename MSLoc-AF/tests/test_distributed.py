"""Exercise real two-process DDP with the official model, on CPU/Gloo or CUDA/NCCL."""
import copy
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_contract import ROOT, fixture
# Import the entry point before NMS tests temporarily prepend vendor/utils to
# sys.path; production also imports the local metrics before loading NMS.
import train as training_entry
from af_backend import build_model, training_utilities
from common import read_json, sha256
from dataset import FeatureDataset
from engine import seed_everything
from parallel import assigned_device, epoch_video_order, parse_device_ids, validate_cuda_devices
from runtime_compat import LAUNCHER_SOURCE_TRANSITIONS, require_compatible_run


def launch_pair(root, splits, output, *extra):
    """Use a FileStore so Windows CPU tests need no TCP rendezvous service."""
    store = root / f"rendezvous_{time.time_ns()}"
    processes, logs = [], []
    command = [sys.executable, str(ROOT / "train.py"),
        "--train-json", str(splits["train"]), "--features", str(root),
        "--variant", "xclip", "--output", str(output), "--batch-size", "4",
        "--epochs", "2", "--warmup-epochs", "0", "--width", "8",
        "--max-seq-len", "32", "--attention-window", "5", "--branches", "2",
        "--device", "cpu", *map(str, extra)]
    try:
        for rank in range(2):
            stream = (root / f"{store.name}_rank{rank}.log").open("w", encoding="utf-8")
            logs.append(stream)
            env = dict(os.environ, WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                MSLOC_AF_DIST_INIT_METHOD=store.as_uri(), PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
            processes.append(subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 120
        for process in processes:
            process.wait(timeout=max(1, deadline-time.monotonic()))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for stream in logs:
            stream.close()
    results = [(p.returncode, Path(s.name).read_text(encoding="utf-8")) for p,s in zip(processes,logs)]
    if any(code for code,_ in results):
        raise AssertionError("\n".join(f"rank {i} exit {code}:\n{log}" for i,(code,log) in enumerate(results)))
    return [log for _,log in results]


def launch_spawn(root, splits, output, *extra):
    """Exercise the SAME internal spawn launcher as --device-ids, using CPU."""
    command = [sys.executable, str(Path(__file__).resolve()), "--spawn-cpu",
        "--train-json", str(splits["train"]), "--features", str(root),
        "--variant", "xclip", "--output", str(output), "--train-batch-size", "4",
        "--epochs", "2", "--warmup-epochs", "0", "--width", "8",
        "--max-seq-len", "32", "--attention-window", "5", "--branches", "2",
        "--device", "cpu", *map(str, extra)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MSLOC_AF_DIST_INIT_METHOD"):
        env.pop(key, None)
    result = subprocess.run(command, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    return result.stdout+result.stderr


class DistributedTests(unittest.TestCase):
    def test_only_exact_launcher_update_can_resume_previous_ddp_version(self):
        current = dict(world_size=2,batch_size=4,source_sha256={
            name:sha256(ROOT/name) for name in LAUNCHER_SOURCE_TRANSITIONS})
        saved = dict(current,source_sha256={name:pair[0] for name,pair in LAUNCHER_SOURCE_TRANSITIONS.items()})
        require_compatible_run(saved,current)
        with self.assertRaises(ValueError):
            require_compatible_run(saved,dict(current,world_size=4))
        with self.assertRaises(ValueError):
            require_compatible_run(saved,dict(current,source_sha256={name:"changed" for name in current["source_sha256"]}))

    def test_device_ids_validation_mapping_and_direct_python_entry(self):
        self.assertEqual(parse_device_ids(" 3,1 "),[3,1])
        self.assertEqual(parse_device_ids([3,1]),[3,1])
        for invalid in ("", "0,0", "-1,0", "0,,1", "0.5,1", [True,1], [1.5,2]):
            with self.assertRaises(ValueError):
                parse_device_ids(invalid)
        self.assertEqual(str(assigned_device("cuda:0",[3,1],0,True)),"cuda:3")
        self.assertEqual(str(assigned_device("cuda:0",[3,1],1,True)),"cuda:1")
        self.assertEqual(str(assigned_device("cuda:0",[3],0,False)),"cuda:3")
        with patch("torch.cuda.is_available",return_value=True), patch("torch.cuda.device_count",return_value=4):
            validate_cuda_devices([3,1])
            with self.assertRaises(ValueError):
                validate_cuda_devices([4])
        args = SimpleNamespace(device_ids=[3,1],batch_size=4)
        with patch("train.parse_args",return_value=args), patch("train.validate_cuda_devices") as validate, \
                patch("train.spawn_training") as spawn, patch.dict(os.environ,WORLD_SIZE="1"):
            training_entry.main()
            validate.assert_called_once_with([3,1])
            spawn.assert_called_once_with(args,2)

    def test_global_batch_shards_whole_videos_without_duplicates_and_resumes(self):
        order = torch.randperm(11,generator=torch.Generator().manual_seed(42)).tolist()
        shards = [epoch_video_order(11,4,2,rank,42)[0] for rank in range(2)]
        self.assertEqual(set(shards[0]) & set(shards[1]),set())
        self.assertEqual(set(shards[0]+shards[1]),set(order[:8]))
        for rank in range(2):
            self.assertEqual(epoch_video_order(11,4,2,rank,42,1)[0],shards[rank][2:])
        for batch in (1,3):
            with self.assertRaisesRegex(ValueError,"GLOBAL"):
                epoch_video_order(11,batch,2,0,42)

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "Requires two-process Gloo")
    def test_real_ddp_step_official_gradient_mean_and_mid_epoch_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="msloc_ddp_") as temporary:
            root = Path(temporary)
            splits = fixture(root,per_kind=4)
            output = root / "resumed"
            logs = launch_pair(root,splits,output,"--max-updates-this-run",1)
            self.assertIn("effective_batch=4; batch_per_rank=2",logs[0])
            self.assertIn("scaled_lr=0.0002",logs[0])
            self.assertIn("Train 1/2",logs[0])
            self.assertNotIn("Training epoch",logs[1])
            self.assertNotIn("Train 1/2",logs[1])
            paused = torch.load(output/"last.pt",map_location="cpu",weights_only=False)
            config = read_json(output/"run_config.json")
            self.assertEqual((paused["world_size"],paused["batch_in_epoch"]),(2,1))
            self.assertEqual(config["optimizer_config"]["learning_rate"],2e-4)
            self.assertNotIn("accumulation_steps",config)

            # Independent reference: upstream models/losses on each shard, then
            # average their gradients BEFORE the official clip/optimizer/EMA step.
            utilities = training_utilities()
            initial_state, models = None, []
            normalizers = []
            for rank in range(2):
                seed_everything(42+rank)
                model = build_model(8,config["model_config"])
                if rank == 0:
                    initial_state = copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(initial_state)
                dataset = FeatureDataset(splits["train"],root,"xclip",is_training=True,max_seq_len=32)
                order,_ = epoch_video_order(len(dataset),4,2,rank,42)
                model.train()
                model([dataset[index] for index in order[:2]])["final_loss"].backward()
                models.append(model)
                normalizers.append(model.loss_normalizer)
            optimizer = utilities.make_optimizer(models[0],config["optimizer_config"])
            scheduler = utilities.make_scheduler(optimizer,config["optimizer_config"],2)
            ema = utilities.ModelEma(models[0])
            for first,second in zip(models[0].parameters(),models[1].parameters()):
                self.assertEqual(first.grad is None,second.grad is None)
                if first.grad is not None:
                    first.grad.add_(second.grad).div_(2)
            torch.nn.utils.clip_grad_norm_(models[0].parameters(),1.)
            optimizer.step()
            scheduler.step()
            ema.update(models[0])
            for actual,expected in ((paused["model"],models[0].state_dict()),(paused["ema"],ema.module.state_dict())):
                for name,value in expected.items():
                    torch.testing.assert_close(actual[name],value,rtol=1e-5,atol=1e-7)
            self.assertEqual([s["loss_normalizer"] for s in paused["rank_states"]],normalizers)

            # Resume mid-epoch, cross an epoch boundary, and compare with an
            # uninterrupted DDP run, including each rank's RNG/normalizer state.
            logs = launch_spawn(root,splits,output,"--resume")
            self.assertIn("Restored:",logs)
            launch_pair(root,splits,root/"continuous","--no-progress")
            resumed = torch.load(output/"last.pt",map_location="cpu",weights_only=False)
            reference = torch.load(root/"continuous/last.pt",map_location="cpu",weights_only=False)
            self.assertEqual(resumed["global_updates"],reference["global_updates"])
            self.assertEqual(resumed["global_updates"],4)
            for name in ("model","ema"):
                for key,value in reference[name].items():
                    torch.testing.assert_close(resumed[name][key],value,rtol=1e-6,atol=1e-7)
            for actual,expected in zip(resumed["rank_states"],reference["rank_states"]):
                self.assertEqual(actual["loss_normalizer"],expected["loss_normalizer"])
                self.assertEqual(actual["python_rng"],expected["python_rng"])
                torch.testing.assert_close(actual["torch_rng"],expected["torch_rng"],rtol=0,atol=0)
            final = torch.load(output/"final.pt",map_location="cpu",weights_only=False)
            standalone = build_model(final["input_dim"],final["model_config"])
            standalone.load_state_dict(final["model"],strict=True)
            self.assertTrue(read_json(output/"training_progress.json")["complete"])
            old_mtime = (output/"last.pt").stat().st_mtime_ns
            self.assertIn("Training already complete",launch_spawn(root,splits,output,"--resume"))
            self.assertEqual((output/"last.pt").stat().st_mtime_ns,old_mtime)


if __name__ == "__main__":
    if "--spawn-cpu" in sys.argv:
        sys.argv.remove("--spawn-cpu")
        from train import parse_args, spawn_training
        spawn_training(parse_args(),2)
    else:
        unittest.main()
