"""Server-side CPU integration check: real Ray objects, receivers and stale-version rejection."""

import unittest
import os

import ray
import torch

from ray_protocol import WeightVersion, tensor_checksum
from demo_utils import ROOT


@ray.remote(num_cpus=1)
class Receiver:
    def __init__(self):
        self.version = WeightVersion()

    def load(self, packet):
        version, tensor = packet["version"], packet["tensor"]
        checksum = tensor_checksum([("weight", tensor)])
        if checksum != packet["checksum"]:
            raise ValueError("Corrupt transport.")
        self.version.begin(version, [["weight"]])
        self.weight = tensor.clone()
        self.version.accept(version, 0, ["weight"], checksum)
        self.version.commit(version, [checksum])
        return checksum

    def predict(self, version, value):
        self.version.require(version)
        return (self.weight @ value).tolist()


class RayTransportTest(unittest.TestCase):
    def test_update_reaches_both_receivers_and_old_version_is_rejected(self):
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "Qwen/demo"), str(ROOT / "Qwen"), os.environ.get("PYTHONPATH", "")])
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False)
        try:
            workers = [Receiver.remote(), Receiver.remote()]
            for version in range(2):
                tensor = torch.eye(2) * (version+1)
                checksum = tensor_checksum([("weight", tensor)])
                packet = ray.put({"version": version, "tensor": tensor, "checksum": checksum})
                self.assertEqual(ray.get([w.load.remote(packet) for w in workers], timeout=30), [checksum, checksum])
                outputs = ray.get([w.predict.remote(version, torch.ones(2)) for w in workers], timeout=30)
                self.assertEqual(outputs, [[float(version+1)]*2]*2)
            with self.assertRaises(ray.exceptions.RayTaskError):
                ray.get(workers[0].predict.remote(0, torch.ones(2)), timeout=30)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
