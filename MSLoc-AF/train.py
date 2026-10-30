"""Train xclip/mamba/fusion ActionFormer experiments on a shared frozen cache."""
import argparse
import copy
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from af_backend import (PROTOCOL, build_model, model_config, official_preset, training_utilities,
                        data_utilities, minimum_padded_length)
from common import VARIANTS, read_json, sha256, write_json
from dataset import FeatureDataset, collate_videos
from engine import infer, save_torch, seed_everything
from metrics import choose_threshold, prediction_records
from cli_config import parse_stage_args
from display import tqdm
from runtime_compat import require_compatible_run
from parallel import (DistributedContext, epoch_video_order, parse_device_ids,
                      validate_cuda_devices)


def parse_args():
    preset = official_preset()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", help="Optional; omit to train fixed epochs without validation")
    parser.add_argument("--features", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", help="Single-process device when --device-ids is omitted")
    parser.add_argument("--device-ids", help="GPU IDs, e.g. 0,1,2,3; starts one DDP process per selected GPU")
    parser.add_argument("--epochs", type=int, default=preset["opt"]["epochs"],
                        help="Official meaning: epochs AFTER warmup; default 30+5=35 total")
    parser.add_argument("--batch-size", "--train-batch-size", type=int, default=preset["loader"]["batch_size"],
                        help="GLOBAL videos per update across all GPUs; must divide evenly across ranks")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers per rank; 0 avoids copying high-dimensional caches")
    parser.add_argument("--lr", type=float, default=preset["opt"]["learning_rate"],
                        help="Base LR; multiplied by GPU count as in official ActionFormer train.py")
    parser.add_argument("--weight-decay", type=float, default=preset["opt"]["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=preset["opt"]["warmup_epochs"])
    parser.add_argument("--clip-grad", type=float, default=preset["train_cfg"]["clip_grad_l2norm"])
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--width", type=int, default=preset["model"]["embd_dim"])
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Default: derive padding from training feature length and official divisibility")
    parser.add_argument("--attention-window", type=int, default=preset["model"]["n_mha_win_size"])
    parser.add_argument("--branches", type=int, default=preset["model"]["backbone_arch"][-1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="CUDA mixed precision; disabled by default")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="TASLE F1 decision threshold, separate from official NMS min_score=0.001")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every-updates", type=int, default=20,
                        help="Persist at optimizer-update boundaries; epoch end is always saved")
    parser.add_argument("--max-updates-this-run", type=int, default=0,
                        help="Gracefully stop after N updates for a resume check; 0=unlimited")
    args = parse_stage_args(parser,"train")
    if args.device_ids is not None:
        try:
            args.device_ids = parse_device_ids(args.device_ids)
        except ValueError as error:
            parser.error(str(error))
        if not args.device.startswith("cuda"):
            parser.error("--device-ids selects CUDA GPUs; omit it when using --device cpu")
        if args.batch_size < len(args.device_ids) or args.batch_size % len(args.device_ids):
            parser.error("--train-batch-size is GLOBAL and must be >= GPU count and divisible by GPU count")
    if (min(args.epochs, args.batch_size) <= 0 or args.num_workers < 0
            or args.lr <= 0 or args.weight_decay < 0 or args.warmup_epochs < 0
            or args.clip_grad <= 0 or not 0 <= args.ema_decay < 1
            or args.checkpoint_every_updates <= 0 or args.max_updates_this_run < 0
            or not 0 <= args.score_threshold <= 1
            or any(not 0 <= threshold <= 1 for threshold in args.thresholds)):
        parser.error("Invalid training hyperparameters")
    if args.amp and not args.device.startswith("cuda"):
        parser.error("--amp requires a CUDA device")
    return args


def train_model(args, context):
    preset = official_preset()
    log = context.print
    show_progress = args.show_progress and context.primary
    # Different stochastic augmentation/dropout streams, synchronized model weights
    # via DDP's initial broadcast. The global video permutation uses args.seed.
    seed_everything(args.seed + context.rank)
    log(f"MSLoc-AF: ActionFormer, features={args.variant}, seed={args.seed}, device={context.device}, "
        f"world_size={context.world_size}",flush=True)
    if args.config:
        log(f"Config: {args.config}",flush=True)
    log("******* Building dataloaders *******",flush=True)
    total_epochs = args.epochs+args.warmup_epochs
    train = FeatureDataset(args.train_json, args.features, args.variant,show_progress=show_progress,
                           is_training=True,max_seq_len=args.max_seq_len)
    val = FeatureDataset(args.val_json, args.features, args.variant,show_progress=show_progress) if args.val_json else None
    if val is not None:
        if train.signature != val.signature:
            raise ValueError("Training/validation feature specifications differ")
        overlap = {r["video_path"] for r in train.records} & {r["video_path"] for r in val.records}
        if overlap:
            raise ValueError(f"Training/validation video leakage: {sorted(overlap)[:5]}")
    if not any(r["type"] == "fake" for r in train.records):
        raise ValueError("Training data contains no fake videos")
    reports = {"train": train.supervision_report()}
    if val is not None:
        reports["val"] = val.supervision_report()
    if args.max_seq_len is None:
        args.max_seq_len = minimum_padded_length(reports["train"]["max_windows"],
                                                args.attention_window,args.branches)
    train.max_seq_len = args.max_seq_len
    log(f"Temporal grid: stride={train.spec['stride_seconds']}s; "
          f"max training T={reports['train']['max_windows']}; padded length={args.max_seq_len}",flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = model_config(args.width, args.max_seq_len, args.attention_window, args.branches,input_dim=train.input_dim)
    opt_config = copy.deepcopy(preset["opt"])
    opt_config.update(learning_rate=args.lr*context.world_size,weight_decay=args.weight_decay,epochs=args.epochs,
                      warmup_epochs=args.warmup_epochs,warmup=args.warmup_epochs>0)
    settings = vars(args).copy()
    settings.pop("resume")
    settings.pop("device")
    settings.pop("device_ids")
    settings.pop("checkpoint_every_updates")
    settings.pop("max_updates_this_run")
    settings.pop("config")
    settings.pop("show_progress")
    settings.update(train_sha256=sha256(args.train_json), val_sha256=sha256(args.val_json) if val is not None else None,
                    protocol=PROTOCOL,total_epochs=total_epochs,optimizer_config=opt_config,
                    world_size=context.world_size, batch_size_per_rank=args.batch_size//context.world_size,
                    parallel_backend="ddp" if context.distributed else "single",
                    loader_drop_last=True,official_crop=dict(crop_ratio=preset["dataset"]["crop_ratio"],
                                                           trunc_thresh=preset["dataset"]["trunc_thresh"]),
                    feature_signature=train.signature, model_config=config,
                    source_sha256={name: sha256(Path(__file__).parent / name) for name in (
                        "train.py", "parallel.py", "cli_config.py", "common.py", "dataset.py", "af_backend.py", "engine.py", "metrics.py")},
                    metric_source_sha256=sha256(Path(__file__).resolve().parents[1] / "scripts/evaluate_long.py"),
                    upstream_source_sha256=sha256(Path(__file__).parent / "vendor/SOURCE.json"))
    config_path = output / "run_config.json"
    last_path = output / "last.pt"
    export_path = output / ("best.pt" if val is not None else "final.pt")
    # Validate before writing run metadata or constructing the large model.
    _, total_batches = epoch_video_order(len(train), args.batch_size, context.world_size, context.rank, args.seed)
    batch_size_per_rank = args.batch_size // context.world_size
    train_count = total_batches * args.batch_size
    def initialize_output():
        if args.resume:
            if not config_path.is_file():
                raise ValueError("Resume requires the original run_config.json")
            require_compatible_run(read_json(config_path),settings)
        elif any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}; use a new path or --resume")
        if not args.resume:
            write_json(config_path, settings)
        write_json(output / "supervision_report.json", reports)
    context.primary_call(initialize_output)
    progress_path = output / "training_progress.json"
    if (args.resume and progress_path.exists() and read_json(progress_path).get("complete")
            and last_path.is_file() and export_path.is_file()):
        log(f"Training already complete: {export_path}", flush=True)
        return
    for split, report in reports.items():
        missed = len(report["segments_without_finest_grid_center"])
        log(f"{split}: videos={report['videos']}, real={report['real_videos']}, "
              f"GT segments={report['fake_segments']}, no interior grid center={missed}", flush=True)
        if missed:
            log("Short segments lacking an interior feature point receive no positive target; "
                  "see supervision_report.json. A denser stride requires a new cache for ALL variants.", flush=True)
    model = build_model(train.input_dim, config).to(context.device)
    training_model = context.wrap(model)
    utilities = training_utilities()
    optimizer = utilities.make_optimizer(model,opt_config)
    scheduler = utilities.make_scheduler(optimizer,opt_config,total_batches)
    model_ema = utilities.ModelEma(model,decay=args.ema_decay)
    ema = model_ema.module.requires_grad_(False)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    epoch_start, best_score, batch_start, global_updates = 0, float("-inf"), 0, 0
    history, restored_totals = [], {}
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        if state["world_size"] != context.world_size:
            raise ValueError("Resume requires the original GPU/process count")
        rank_state = state["rank_states"][context.rank]
        model.loss_normalizer = rank_state["loss_normalizer"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch_start, best_score = state["epoch"], state["best_score"]
        batch_start, global_updates = state["batch_in_epoch"], state["global_updates"]
        restored_totals = state["epoch_totals"]
        history = state["history"]
        torch.set_rng_state(rank_state["torch_rng"])
        np.random.set_state(rank_state["numpy_rng"])
        random.setstate(rank_state["python_rng"])
        if context.device.type == "cuda" and rank_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(rank_state["cuda_rng"], device=context.device)
        log(f"Restored: epoch_index={epoch_start}, completed_batches={batch_start}, "
              f"optimizer_updates={global_updates}", flush=True)
    elif args.resume:
        log("Matching run configuration found without a committed checkpoint; restarting epoch 1", flush=True)
    def checkpoint(epoch_index, completed_batches, totals):
        rank_states = context.gather_states(dict(
            loss_normalizer=model.loss_normalizer, torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state(context.device) if context.device.type == "cuda" else None))
        def persist():
            save_torch(last_path, dict(model=model.state_dict(), ema=ema.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch_index, batch_in_epoch=completed_batches, global_updates=global_updates,
                epoch_totals=totals, best_score=best_score, history=history,
                world_size=context.world_size, rank_states=rank_states,
                # Retain rank-0 diagnostic keys used by the single-device audit.
                **rank_states[0]))
            write_json(progress_path,dict(epoch=epoch_index,batch_in_epoch=completed_batches,
                        global_updates=global_updates,complete=epoch_index >= total_epochs and completed_batches == 0))
        context.primary_call(persist)
    updates_this_run = 0
    params = sum(p.numel() for p in model.parameters())
    if val is None:
        log(f"No validation: {args.epochs} epochs + {args.warmup_epochs} warmup = {total_epochs} total, final-epoch EMA export, "
              f"fixed score threshold={args.score_threshold}", flush=True)
    def export(epoch, threshold, selection, validation_metrics=None):
        save_torch(export_path, dict(model=ema.state_dict(), model_config=config,protocol=PROTOCOL,
            input_dim=train.input_dim, variant=args.variant, feature_signature=train.signature,
            epoch=epoch, score_threshold=threshold, selection=selection,
            validation_metrics=validation_metrics,
            parameters=params, upstream_source_sha256=settings["upstream_source_sha256"],
            metric_source_sha256=settings["metric_source_sha256"], source_sha256=settings["source_sha256"]))
    log(f"variant={args.variant}; input_dim={train.input_dim}; parameters={params:,}; "
        f"effective_batch={args.batch_size}; batch_per_rank={batch_size_per_rank}; GPUs/processes={context.world_size}; "
        f"base_lr={args.lr:g}; scaled_lr={opt_config['learning_rate']:g}; device={context.device}", flush=True)
    for epoch in range(epoch_start, total_epochs):
        log(f"******* Training epoch {epoch+1}/{total_epochs} *******",flush=True)
        started = time.monotonic()
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        cursor = batch_start if epoch == epoch_start else 0
        totals = dict(restored_totals) if cursor else {}
        # Reproducible per-epoch order, independent of dropout/global RNG. Resume
        # only reads unprocessed videos, rather than replaying earlier batches.
        order, _ = epoch_video_order(len(train), args.batch_size, context.world_size,
                                    context.rank, args.seed+epoch, cursor)
        remaining = Subset(train, order)
        loader = DataLoader(remaining, batch_size=batch_size_per_rank, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed+epoch+context.rank),
                            num_workers=args.num_workers, collate_fn=collate_videos, drop_last=True,
                            worker_init_fn=data_utilities().worker_init_reset_seed)
        with tqdm(loader,total=total_batches,initial=cursor,desc=f"Train {epoch+1}/{total_epochs}",
                  dynamic_ncols=True,disable=not show_progress) as pbar:
            for step, videos in enumerate(pbar, start=cursor):
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    losses = training_model(videos)
                    loss = losses["final_loss"]
                context.require_finite(loss, f"Nonfinite loss at epoch {epoch+1}, batch {step+1}")
                scaler.scale(loss).backward()
                for name, value in context.mean_losses(losses).items():
                    totals[name] = totals.get(name, 0.0) + value * args.batch_size
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                context.require_finite(norm, "Nonfinite gradients; rerun in full precision")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model_ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                global_updates += 1
                updates_this_run += 1
                stop = args.max_updates_this_run and updates_this_run >= args.max_updates_this_run
                if global_updates % args.checkpoint_every_updates == 0 or step+1 == total_batches or stop:
                    checkpoint(epoch, step+1, totals)
                if stop:
                    if context.primary:
                        tqdm.write(f"Saved after {updates_this_run} updates this run; continue with the same command "
                                   "and --resume (omit --max-updates-this-run to finish)")
                    return
                seen = min((step+1)*args.batch_size,train_count)
                pbar.set_postfix(loss=totals["final_loss"]/seen,
                    cls_loss=totals["cls_loss"]/seen,reg_loss=totals["reg_loss"]/seen,
                    lr=optimizer.param_groups[0]["lr"])
        summary = dict(epoch=epoch+1, losses={k:v/train_count for k,v in totals.items()},
                       score_threshold=args.score_threshold)
        if val is not None and context.primary:
            raw = infer(ema, val,show_progress=show_progress)
            best, trials = choose_threshold(val, raw, args.thresholds)
            score = best["metrics"]["Total"]["Loc_IoU"]
            summary.update(validation=best, score_threshold=best["threshold"])
            write_json(output / "val_thresholds_last.json", trials)
            if score > best_score:
                best_score = score
                export(epoch+1, best["threshold"], "validation_best_ema", best["metrics"])
                write_json(output / "val_predictions_best.json",
                           prediction_records(val.records, raw, best["threshold"]))
        elif val is None and epoch+1 == total_epochs and context.primary:
            export(epoch+1, args.score_threshold, "last_epoch_ema")
        summary["seconds"] = time.monotonic()-started
        summary, best_score = context.broadcast((summary, best_score))
        history.append(summary)
        if context.primary:
            write_json(output / "history.json", history)
        checkpoint(epoch+1, 0, {})
        if val is not None:
            log(f"epoch={epoch+1}: val F1_Loc={best_score:.6f}, threshold={summary['score_threshold']}; "
                  f"best={best_score:.6f}; elapsed={summary['seconds']:.1f}s", flush=True)
        else:
            log(f"*****Average Training loss {summary['losses']['final_loss']:.6f} "
                  f"*****Epoch {epoch+1} *****",flush=True)
            log(f"Epoch {epoch+1} completed in {summary['seconds']:.1f}s; "
                  f"checkpoint: {last_path}",flush=True)
    log(f"Inference EMA model: {export_path}")


def run_training(args):
    context = DistributedContext(args.device, args.device_ids)
    try:
        train_model(args, context)
    finally:
        context.close()


def _spawn_worker(rank, args, world_size, init_method):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
                      MSLOC_AF_DIST_INIT_METHOD=init_method)
    run_training(args)


def spawn_training(args, world_size):
    """Single-node DDP launcher; FileStore avoids a fixed or raced TCP port."""
    with tempfile.TemporaryDirectory(prefix="msloc_af_ddp_") as folder:
        init_method = (Path(folder) / "rendezvous").as_uri()
        torch.multiprocessing.spawn(_spawn_worker, args=(args, world_size, init_method),
                                    nprocs=world_size, join=True)


def main():
    args = parse_args()
    if args.device_ids:
        validate_cuda_devices(args.device_ids)
        if len(args.device_ids) > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
            print(f"Launching DDP: devices={args.device_ids}; global batch={args.batch_size}", flush=True)
            spawn_training(args, len(args.device_ids))
            return
    run_training(args)


if __name__ == "__main__":
    main()
