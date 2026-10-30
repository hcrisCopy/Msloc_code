"""Verify source provenance and numerical parity; production verification requires C++ NMS."""
import argparse
import copy
import json

import numpy as np
import torch

from af_backend import (NAMESPACE, PROTOCOL, batched_nms, build_model, data_utilities,
                        model_config, official_modules, official_preset,
                        require_official_nms, training_utilities, verify_vendor)
from common import grid_to_seconds, seconds_to_grid


def verify_core(device="cpu"):
    verify_vendor()
    preset = official_preset()
    tools = training_utilities()
    cfg = model_config(width=8,max_seq_len=32,attention_window=5,branches=2,input_dim=8)
    torch.manual_seed(7)
    adapter = build_model(8,cfg).to(device)
    modeling,_ = official_modules()
    reference = modeling.make_meta_arch("LocPointTransformer",**copy.deepcopy(cfg)).to(device)
    reference.load_state_dict(adapter.state_dict())
    assert type(adapter) is type(reference)
    assert type(adapter.backbone).__module__ == NAMESPACE+".modeling.backbones"
    assert type(adapter.backbone.proj[0]).__module__ == NAMESPACE+".modeling.blocks"
    opt = copy.deepcopy(preset["opt"])
    opt.update(epochs=1,warmup_epochs=0,warmup=False)
    first = tools.make_optimizer(adapter,opt)
    second = tools.make_optimizer(reference,opt)
    first_sched = tools.make_scheduler(first,opt,1)
    second_sched = tools.make_scheduler(second,opt,1)
    first_ema,second_ema = tools.ModelEma(adapter),tools.ModelEma(reference)
    sample = dict(video_id="parity",feats=torch.randn(8,4),segments=torch.tensor([[0.1,2.9]]),
                  labels=torch.tensor([0]),fps=8.,duration=8.,feat_stride=16.,feat_num_frames=16.)
    adapter.train()
    torch.manual_seed(123)
    losses = adapter([copy.deepcopy(sample)])
    first.zero_grad(set_to_none=True)
    losses["final_loss"].backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(),preset["train_cfg"]["clip_grad_l2norm"])
    first.step()
    first_sched.step()
    first_ema.update(adapter)
    torch.manual_seed(123)
    # This is the ORIGINAL official training loop, not a locally reproduced reference.
    tools.train_one_epoch([[copy.deepcopy(sample)]],reference,second,second_sched,0,
                          model_ema=second_ema,clip_grad_l2norm=preset["train_cfg"]["clip_grad_l2norm"],
                          print_freq=100)
    for left,right in ((adapter.state_dict(),reference.state_dict()),
                       (first_ema.module.state_dict(),second_ema.module.state_dict())):
        for name,value in left.items():
            torch.testing.assert_close(value,right[name],rtol=1e-6,atol=1e-7)
    assert first_sched.get_last_lr() == second_sched.get_last_lr()
    segments = [[1.25,3.75],[7.0,8.0]]
    actual = seconds_to_grid(segments,2.,2.)
    source_formula = np.asarray(segments,dtype=np.float32)*8./16.-0.5*16./16.
    np.testing.assert_allclose(actual,source_formula)
    np.testing.assert_allclose(grid_to_seconds(actual,2.,2.,8.),segments)
    assert data_utilities().truncate_feats.__module__ == NAMESPACE+".datasets.data_utils"
    return dict(protocol=PROTOCOL,official_commit="61ea7eb9308a568b0cf45e3804830836e30061de",
                model_and_one_step_parity=True,time_coordinates_match_upstream=True,
                reference="unmodified train_utils.train_one_epoch",test_model="tiny CPU/GPU interface test, not paper hyperparameters")


def verify_nms():
    extension = require_official_nms()
    config = official_preset()["test_cfg"]
    segments = torch.tensor([[0.,2.],[0.,2.],[3.,4.]])
    scores = torch.tensor([0.9,0.8,0.7])
    labels = torch.zeros(3,dtype=torch.long)
    output,score,_ = batched_nms(segments,scores,labels,config["iou_threshold"],config["min_score"],
        config["max_seg_num"],use_soft_nms=True,multiclass=config["multiclass_nms"],
        sigma=config["nms_sigma"],voting_thresh=config["voting_thresh"])
    assert len(output) == 3 and torch.isfinite(output).all() and torch.isfinite(score).all()
    assert (score[:-1] >= score[1:]).all() and float(score.min()) < 0.2
    return dict(official_soft_nms=True,extension=extension.__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--allow-missing-nms",action="store_true",
                        help="Local source/math checks only; does NOT certify production inference")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    result = verify_core(args.device)
    try:
        result.update(verify_nms())
        result["complete"] = True
    except RuntimeError as error:
        if not args.allow_missing_nms:
            raise
        result.update(complete=False,official_soft_nms=False,unverified=str(error))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
