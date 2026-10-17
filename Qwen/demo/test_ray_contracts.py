"""Server-side CPU tests for losses, unmerged weight export and synchronization contracts.

Run: python -m unittest discover -s Qwen/demo -p test_ray_contracts.py -v
No model downloads, GPU allocation or training dataset are needed.
"""

import copy
import unittest

import torch

from ray_protocol import WeightVersion, batch_indices, make_layout, tensor_checksum, verify_signal
from ray_models import effective_weights, grpo_loss, opsd_loss


class ContractTests(unittest.TestCase):
    def test_stale_rollout_and_partial_sync_are_rejected(self):
        version = WeightVersion()
        with self.assertRaises(ValueError):
            version.require(0)
        version.begin(0, [["a"], ["b"]])
        version.accept(0, 0, ["a"], "digest-a")
        with self.assertRaises(ValueError):
            version.commit(0, ["digest-a"])
        with self.assertRaises(ValueError):
            version.require(0)
        version.accept(0, 1, ["b"], "digest-b")
        version.commit(0, ["digest-a", "digest-b"])
        version.require(0)
        with self.assertRaises(ValueError):
            version.require(1)
        with self.assertRaises(ValueError):
            version.begin(0, [["a"]])

    def test_corrupt_payload_and_wrong_sample_are_detectable(self):
        tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        checksum = tensor_checksum([("weight", tensor)])
        altered = tensor.clone()
        altered[1, 2] += 1
        self.assertNotEqual(checksum, tensor_checksum([("weight", altered)]))
        job = {"index": 0, "version": 4, "tokens": [10, 11]}
        with self.assertRaises(ValueError):
            verify_signal(job, {**job, "tokens": [10, 12]})

    def test_device_count_does_not_change_batch_selection(self):
        expected = batch_indices(100, 7, 8)
        for gpus in (1, 2, 3, 4, 8):
            layout = make_layout(gpus, "grpo")
            self.assertEqual({r["slot"] for r in layout}, set(range(gpus)))
            for slot in range(gpus):
                self.assertAlmostEqual(sum(r["gpu"] for r in layout if r["slot"] == slot), 1.0)
            trainers = sum(r["role"] == "trainer" for r in layout)
            reconstructed = sorted(i for rank in range(trainers) for i in expected[rank::trainers])
            self.assertEqual(reconstructed, sorted(expected))

    def test_full_vocabulary_jsd_and_teacher_stop_gradient(self):
        student = torch.tensor([[0.8, 0.2]], dtype=torch.float32).log().requires_grad_()
        teacher = torch.tensor([[0.2, 0.8]], dtype=torch.float32).log().requires_grad_()
        loss = opsd_loss(student, teacher)
        self.assertAlmostEqual(float(loss), 0.19274476, places=6)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad[0, 0]), 0)
        self.assertLess(float(student.grad[0, 1]), 0)

    def test_grpo_ratio_one_still_has_policy_gradient(self):
        logps = torch.tensor([-0.5, -1.0], requires_grad=True)
        loss, kl = grpo_loss(logps, logps.detach().clone(), advantage=2.0)
        self.assertAlmostEqual(float(loss), -2.0, places=6)
        self.assertAlmostEqual(float(kl), 0.0, places=6)
        loss.backward()
        torch.testing.assert_close(logps.grad, torch.tensor([-1.0, -1.0]))

    def test_group_advantages_use_sample_standard_deviation(self):
        from swift.rl_core.advantage import compute_advantages
        rewards = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        advantage, _ = compute_advantages(rewards, torch.tensor([1.0]), 4)
        expected = torch.tensor([-0.5, -0.5, 0.5, 0.5]) / (torch.tensor([0.0, 0.0, 1.0, 1.0]).std()+1e-4)
        torch.testing.assert_close(advantage, expected)

    def test_effective_lora_export_does_not_mutate_training_weights(self):
        from peft import LoraConfig, get_peft_model
        from transformers import PretrainedConfig

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = PretrainedConfig(tie_word_embeddings=False)
                self.language_model = torch.nn.Linear(4, 4, bias=False)

            def forward(self, value):
                return self.language_model(value)

        torch.manual_seed(42)
        model = get_peft_model(TinyModel(), LoraConfig(r=2, lora_alpha=4, target_modules=["language_model"]))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.1)
        before = copy.deepcopy(model.state_dict())
        weights = dict(effective_weights(model))
        sample = torch.randn(3, 4)
        torch.testing.assert_close(model(sample), torch.nn.functional.linear(sample, weights["language_model.weight"]))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
