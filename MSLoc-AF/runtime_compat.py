"""Compatibility for audited presentation/launcher updates, not model changes."""
LEGACY_SOURCE_HASHES = {
    "train.py": {"6a6fe7aa2c9608f238ea9f72b120fe315e6acaf65177e540d13b7b2a21d163b4", "87aa128a2e5b1b03de167eee1c97e2bf36943251fd458b5a86abd8d286b0e3aa"},
    "engine.py": {"b244e666524f8c26fb55f1e205cfb16b83da4de93507a0c39ef92d4746a9cc13", "a0996b12d4586260604cd3d60a829dfb91613f7a124cd80be1c52d14d023c72f"},
    "metrics.py": {"a331afe8e4c864c7a0bc44088e0ec8f59956f4ff554b58abd9fa9ce4c727a3f7", "9d18509a22d2d8f72a4f8662e5bc37e7678a87a622c424c18d1f48d3980a5039"},
    "dataset.py": {"902460530952713b09daee44e2898bbc18fccd139ac3c36b528861d84497db40", "cf0f1c2a77f3b0e863feabbf7ab15836a2ea47e12f6896ebea3821a0acf66620"},
    "predict.py": {"04b3900a302fb58b6c99356ef0e76028193752d28df87b4a413de15b83fd06df", "47add806b5583846b4366105927ac74dd1071e0519f6b843f912d4af3641efc1"},
}

# Populated with the exact reviewed source hashes after the display-only edits.
PRESENTATION_SOURCE_HASHES = {
    "train.py": {"195ca44ecea4f9490152188c4363e5dd74e61421975357ebf3aea4444dc85df4", "9dea2a712cb55b3c6197194650721d57a3aa3344a300f29b63d29b5f262f3df6"},
    "engine.py": {"f50ce15c8b7211f82466ccfcbd5de6e8830591e83adfe8c4ec8e5826da069680", "1b8f25eed4973c2e2280a2733d85c18794a2742e530cde31b6cd47c44ccf8855"},
    "metrics.py": {"f8a663b1e48db70da92ec87b0f53aae7f9ef55e0b82f474521cf16adc1b12ea5", "ec0052abbd007e3708897121dd706584fbc668e117dbda40e9bf8e8a262b11f5"},
    "dataset.py": {"07847d52a810a241e82e52c26e4d79c1bad9c1ff9f79f7ae60b1bf84ee732d76", "29efb1f003d7a7762f04f9559a0c8d4fb0150695b1190856c1b9c0b87f191336"},
    "predict.py": {"580629b692026c955acce5df4d2897c40c1e056495b854ab4711fb6cc44f5bc7", "1d14821795ba7e7352b206d77114034fcc694d0e8d87836c9035eb90bf8323c4"},
}

# Exact before/after pairs for training/inference launchers and NMS import-path
# cleanup. Model/loss, per-video decode/NMS and checkpoint formats are unchanged.
LAUNCHER_SOURCE_TRANSITIONS = {
    "train.py": ("da3a8e3e5253ffbb43097d769dc4bd169185e4e4c69d7e12a20ac00deef87851",
                 "d3446718d9aafa3759c68fd20c5fbe378bb0261d785b3ea5a985387ab21f6eb5"),
    "parallel.py": ("1df0319269060ffc9cfaa7fdc1f15f26e1689ec6e31a7e125e268219e24806ef",
                    "cec2da049c3fb73f8c56fe9cef4ab460b57d801851ae9543c98abf48e4fdc8e6"),
    "af_backend.py": ("3370f440d4f45fc4531b7e1e9cac3801db1b80be3135980f9794bb565b40c102",
                      "0ed0eee549c6feaf816247a369a8045314c2f435f631d9b2fdc7590ae5695279"),
    "predict.py": ("fbd99f5f4d56a68593bca378e7710313870085efc0b661be91f933d08285ebee",
                   "80539b6e9ae38c9d065a4a755ea584ae3a34eda20c0704cd54d0da9192bd90ba"),
}


def same_computation_source(name, saved, current):
    return saved == current or (saved, current) == LAUNCHER_SOURCE_TRANSITIONS.get(name) or (
        saved in LEGACY_SOURCE_HASHES.get(name,set())
        and current in PRESENTATION_SOURCE_HASHES.get(name,set()))


def require_compatible_run(saved, current):
    old,new = dict(saved),dict(current)
    old_sources,new_sources = old.pop("source_sha256"),new.pop("source_sha256")
    if old != new or old_sources.keys() != new_sources.keys():
        raise ValueError("Resume requires the same data and training configuration")
    for name,digest in old_sources.items():
        if not same_computation_source(name,digest,new_sources[name]):
            raise ValueError(f"Resume source differs beyond the supported presentation/launcher updates: {name}")


def require_compatible_inference(saved, current):
    old,new = dict(saved),dict(current)
    old_hash,new_hash = old.pop("predictor_sha256"),new.pop("predictor_sha256")
    if old != new or not same_computation_source("predict.py",old_hash,new_hash):
        raise ValueError("Prediction resume requires the same checkpoint, annotations, cache and computation")
