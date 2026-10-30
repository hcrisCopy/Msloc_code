"""Load pinned, unmodified ActionFormer implementation and its THUMOS preset.

There is no local attention/head/loss/NMS/optimizer implementation. The sole
model adaptation uses the upstream backbone's existing list-input projection.
"""
import copy
import importlib
import importlib.util
import sys
import types
from pathlib import Path

from common import read_json, sha256

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT/"vendor/actionformer"
PROTOCOL = "actionformer_tasle_official_modules_v1"
NAMESPACE = "_msloc_official_actionformer"
PRESET = VENDOR/"configs/thumos_i3d.yaml"
NMS_DIR = VENDOR/"libs/utils"


def verify_vendor():
    provenance = read_json(ROOT/"vendor/SOURCE.json")
    if provenance["commit"] != "61ea7eb9308a568b0cf45e3804830836e30061de":
        raise RuntimeError("Unexpected ActionFormer revision")
    for relative,digest in provenance["files"].items():
        path = VENDOR/relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Missing/modified official ActionFormer source: {path}")


def require_official_nms():
    """Require the C++ extension built in place from the pinned upstream source."""
    import torch  # Load PyTorch's shared libraries before loading its extension.
    added_path = str(NMS_DIR) not in sys.path
    if added_path:
        sys.path.insert(0,str(NMS_DIR))
    try:
        extension = importlib.import_module("nms_1d_cpu")
    except ImportError as error:
        raise RuntimeError("Official ActionFormer NMS is not built. From MSLoc-AF run: "
            "cd vendor/actionformer/libs/utils && python setup.py build_ext --inplace . "
            "Use Linux/GCC with the same PyTorch environment as training/inference. "
            "No substitute NMS is used.") from error
    finally:
        # Spawn inherits sys.path. Leaving vendor/utils first would shadow this
        # project's metrics.py when a new inference process imports predict.py.
        if added_path:
            sys.path.remove(str(NMS_DIR))
    if not Path(extension.__file__).resolve().is_relative_to(NMS_DIR.resolve()):
        raise RuntimeError("Build nms_1d_cpu --inplace in the pinned vendor/libs/utils directory; "
                           "a global extension has unverifiable source provenance")
    return extension


def batched_nms(*args,**kwargs):
    require_official_nms()
    official_modules()
    return importlib.import_module(NAMESPACE+".utils.nms").batched_nms(*args,**kwargs)


def official_modules():
    if NAMESPACE not in sys.modules:
        verify_vendor()
        package = types.ModuleType(NAMESPACE)
        package.__path__ = [str(VENDOR/"libs")]
        sys.modules[NAMESPACE] = package
        # Delay only the optional binary import so training/parity checks can run
        # without NMS. Actual postprocessing ALWAYS calls upstream nms.py + C++.
        utils = types.ModuleType(NAMESPACE+".utils")
        utils.__path__ = [str(NMS_DIR)]
        utils.batched_nms = batched_nms
        sys.modules[utils.__name__] = utils
    modeling = importlib.import_module(NAMESPACE+".modeling")
    name = NAMESPACE+".defaults"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,VENDOR/"libs/core/config.py")
        config = importlib.util.module_from_spec(spec)
        sys.modules[name] = config
        spec.loader.exec_module(config)
    return modeling,sys.modules[name]


def official_preset():
    _,defaults = official_modules()
    return copy.deepcopy(defaults.load_config(str(PRESET)))


def training_utilities():
    official_modules()
    return importlib.import_module(NAMESPACE+".utils.train_utils")


def data_utilities():
    official_modules()
    name = NAMESPACE+".datasets"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(VENDOR/"libs/datasets")]
        sys.modules[name] = package
    return importlib.import_module(NAMESPACE+".datasets.data_utils")


def minimum_padded_length(feature_length,attention_window=19,branches=5):
    """Round actual training T to the official attention/FPN divisibility rule."""
    divisor = 2**branches * (2*(attention_window//2) if attention_window > 1 else 1)
    return ((feature_length+divisor-1)//divisor)*divisor


def model_config(width=None,max_seq_len=None,attention_window=None,branches=None,input_dim=None):
    config = official_preset()["model"]
    # Reuse the official structure. The training entry point derives padded T
    # from TASLE; small width/depth overrides below are for interface tests.
    width = config["embd_dim"] if width is None else width
    max_seq_len = config["max_seq_len"] if max_seq_len is None else max_seq_len
    attention_window = config["n_mha_win_size"] if attention_window is None else attention_window
    branches = config["backbone_arch"][-1] if branches is None else branches
    if width <= 0 or width % config["n_head"] or branches < 0 or max_seq_len <= 0:
        raise ValueError("Invalid width, pyramid depth or sequence length")
    if 1 < attention_window < 5 or attention_window > 1 and attention_window % 2 == 0:
        raise ValueError("Upstream local attention requires an odd window >=5 (or <=1 for global)")
    divisor = 2**branches * (2*(attention_window//2) if attention_window > 1 else 1)
    if max_seq_len % divisor:
        raise ValueError(f"max_seq_len must be divisible by {divisor}")
    ranges = copy.deepcopy(config["regression_range"])
    if branches != config["backbone_arch"][-1]:
        ranges = [[0,10000]] if branches == 0 else [
            [0 if i == 0 else 2**(i+1),10000 if i == branches else 2**(i+2)]
            for i in range(branches+1)]
    config.update(input_dim=[input_dim] if input_dim is not None else None,
                  embd_dim=[width],fpn_dim=width,head_dim=width,num_classes=1,
                  max_seq_len=max_seq_len,n_mha_win_size=attention_window,
                  backbone_arch=[config["backbone_arch"][0],config["backbone_arch"][1],branches],
                  regression_range=[list(pair) for pair in ranges])
    return config


def build_model(input_dim,config):
    modeling,_ = official_modules()
    resolved = copy.deepcopy(config)
    if resolved["input_dim"] is None:
        resolved["input_dim"] = [input_dim]
    if (not isinstance(resolved["input_dim"],(list,tuple))
            or list(resolved["input_dim"]) != [input_dim]
            or not isinstance(resolved["embd_dim"],(list,tuple))):
        raise ValueError("This is not an official-native-projection model config; legacy custom "
                         "checkpoints require their old code and cannot be reused as this baseline")
    return modeling.make_meta_arch("LocPointTransformer",**resolved)
