"""Read stage-specific defaults from YAML, then let explicit CLI flags override them.

Like MSLoc-PG, relative data/output paths are relative to the working directory.
This module only parses arguments; it never starts another stage.
"""
import argparse
import sys
from pathlib import Path

import yaml


def parse_stage_args(parser, stage):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    known, _ = preliminary.parse_known_args()
    parser.add_argument("--config", help="YAML defaults; explicit CLI arguments take precedence")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", default=True)
    config = {}
    if known.config and not any(flag in sys.argv for flag in ("--help","-h")):
        with Path(known.config).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config,dict):
            parser.error("Config must contain a YAML mapping")
        if stage == "train" and any(key.replace("-", "_") == "accumulation_steps" for key in config):
            parser.error("accumulation_steps was removed: delete it from YAML; one batch is one optimizer update")
    aliases = dict(train_json_path="train_json", val_json_path="val_json",
                   features_path="features", max_epoch="epochs", train_batch_size="batch_size",
                   save_dir="output", model_path="checkpoint", output_dir="output")
    if stage == "extract":
        aliases.update(dataset_base_path="frames_root", xclip_model_path="xclip_weights",
                       features_path="output", extract_batch_size="batch_size",
                       window_length="window_seconds", window_stride="stride_seconds")
        if "annotations" not in config:
            config["annotations"] = [config[key] for key in ("train_json_path","test_json_path","val_json_path")
                                     if config.get(key)]
            if not config["annotations"]:
                config.pop("annotations")
    elif stage == "eval":
        aliases.update(test_json_path="annotations")
        if "output_dir" not in config and config.get("save_dir"):
            config["output_dir"] = str(Path(config["save_dir"])/"eval")
        if "model_path" not in config and config.get("save_dir"):
            config["model_path"] = str(Path(config["save_dir"])/(
                "best.pt" if config.get("val_json_path") else "final.pt"))
    actions = {action.dest:action for action in parser._actions}
    defaults = {}
    for key,value in config.items():
        dest = aliases.get(key,key.replace("-","_"))
        if dest in actions and value is not None:
            action = actions[dest]
            if action.choices and value not in action.choices:
                parser.error(f"Invalid config value for {key}: {value}")
            if action.type is not None:
                value = [action.type(item) for item in value] if isinstance(value,(list,tuple)) else action.type(value)
            defaults[dest] = value
    parser.set_defaults(**defaults)
    for dest in defaults:
        actions[dest].required = False
    return parser.parse_args()
