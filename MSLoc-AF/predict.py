"""Predict with an exported fixed/validation threshold and run the released evaluator."""
import argparse
from pathlib import Path
from queue import Empty

import torch

from af_backend import PROTOCOL, build_model, require_official_nms
from common import VARIANTS, feature_id, read_json, sha256, write_json
from dataset import FeatureDataset
from engine import infer
from metrics import paper_metrics, prediction_records
from cli_config import parse_stage_args
from display import print_evaluation_metrics, tqdm, memory_postfix
from parallel import assigned_device, parse_device_ids, validate_cuda_devices
from runtime_compat import require_compatible_inference, same_computation_source


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", "--model_path", "--model-path", required=True,
                        help="final.pt (no validation) or best.pt; not last.pt")
    parser.add_argument("--output", "--output_dir", "--output-dir", required=True)
    parser.add_argument("--variant",choices=VARIANTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-ids", help="GPU IDs, e.g. 0,1,2,3; distribute complete videos across GPUs")
    parser.add_argument("--resume", action="store_true", help="Reuse committed per-video predictions")
    args = parse_stage_args(parser,"eval")
    if args.device_ids is not None:
        try:
            args.device_ids = parse_device_ids(args.device_ids)
        except ValueError as error:
            parser.error(str(error))
        if not args.device.startswith("cuda"):
            parser.error("--device-ids selects CUDA GPUs; omit it when using --device cpu")
    return args


def committed_predictions(records, progress):
    raw = {}
    for record in records:
        video = record["video_path"]
        path = progress / (feature_id(video)+".json")
        if path.exists():
            saved = read_json(path)
            if saved["video_id"] != video:
                raise ValueError(f"Invalid saved prediction: {path}")
            raw[video] = saved["prediction"]
    return raw


def _inference_worker(rank, args, shards, progress_queue):
    # Inference replicas are independent: there are no gradients or DDP
    # collectives, so uneven video counts never require padding or duplication.
    device = assigned_device(args.device, args.device_ids, rank, distributed=True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = FeatureDataset(args.annotations, args.features, state["variant"])
    model = build_model(state["input_dim"], state["model_config"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    progress = Path(args.output) / "video_progress"
    with torch.no_grad():
        for index in shards[rank]:
            video = dataset[index]
            prediction = model([video])[0]  # Unmodified official decode + Soft-NMS.
            raw = dict(segments=prediction["segments"].tolist(), scores=prediction["scores"].tolist())
            write_json(progress/(feature_id(video["video_id"])+".json"),
                       dict(video_id=video["video_id"], prediction=raw))
            progress_queue.put(video["video_id"])  # Report only after atomic commit.


def inference_shards(pending, processes):
    if processes < 1:
        raise ValueError("Inference process count must be positive")
    return [pending[rank::processes] for rank in range(min(processes,len(pending)))]


def infer_parallel(args, pending, total, completed, processes):
    shards = inference_shards(pending, processes)
    queue = torch.multiprocessing.get_context("spawn").Queue()
    context = torch.multiprocessing.spawn(_inference_worker, args=(args,shards,queue),
                                          nprocs=len(shards), join=False)
    try:
        print("******** Start Testing. ********",flush=True)
        print(f"Parallel inference: {len(shards)} workers, {len(pending)} remaining videos",flush=True)
        with tqdm(total=total,initial=completed,desc="Testing",unit="video",
                  dynamic_ncols=True,disable=not args.show_progress) as pbar:
            while True:
                try:
                    queue.get(timeout=0.1)
                    pbar.update(1)
                    pbar.set_postfix(memory_postfix())
                except Empty:
                    pass
                if context.join(timeout=0):
                    break
            while True:
                try:
                    queue.get_nowait()
                    pbar.update(1)
                except Empty:
                    break
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join()
        queue.close()
        queue.join_thread()


def run_prediction(args, processes=None):
    processes = processes or (len(args.device_ids) if args.device_ids else 1)
    print("Temporal-segmentation model evaluation")
    print("="*50)
    print(f"Config:     {args.config or '(command-line arguments)'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output}")
    print(f"Devices:    {args.device_ids if args.device_ids else args.device}")
    print("="*50)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("protocol") != PROTOCOL:
        raise ValueError("This checkpoint belongs to the earlier modified ActionFormer experiment, "
                         "not the audited official baseline. Reuse the feature cache, but train "
                         "a new model into a new output directory using the official configs.")
    required = {"model_config", "variant", "feature_signature", "score_threshold"}
    if not required.issubset(state):
        raise ValueError("Expected an exported final.pt or best.pt checkpoint")
    if args.variant and args.variant != state["variant"]:
        raise ValueError("Config variant does not match the checkpoint")
    print("******* Building dataloaders *******",flush=True)
    dataset = FeatureDataset(args.annotations, args.features, state["variant"],show_progress=args.show_progress)
    if dataset.signature != state["feature_signature"]:
        raise ValueError("Feature timing/checkpoint/extraction differs from the training cache")
    if sha256(Path(__file__).parent / "vendor/SOURCE.json") != state["upstream_source_sha256"]:
        raise ValueError("ActionFormer vendor source differs from training")
    metric_path = Path(__file__).resolve().parents[1] / "scripts/evaluate_long.py"
    if sha256(metric_path) != state["metric_source_sha256"]:
        raise ValueError("Released evaluation code differs from training")
    for name in ("af_backend.py", "common.py", "dataset.py", "engine.py", "metrics.py"):
        if not same_computation_source(name,state["source_sha256"][name],sha256(Path(__file__).parent / name)):
            raise ValueError(f"Experiment source differs from training: {name}")
    output = Path(args.output)
    spec = dict(checkpoint_sha256=sha256(args.checkpoint), annotation_sha256=sha256(args.annotations),
                feature_signature=dataset.signature, predictor_sha256=sha256(__file__))
    config_path = output / "inference_config.json"
    if args.resume and config_path.exists():
        previous = read_json(config_path)
        require_compatible_inference(previous,spec)
        spec = previous
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(config_path, spec)
    progress = output / "video_progress"
    progress.mkdir(exist_ok=True)
    raw = committed_predictions(dataset.records, progress)
    print(f"Prediction progress: {len(raw)}/{len(dataset)} videos already committed", flush=True)
    if len(raw) == len(dataset) and (output/"metrics.json").is_file() and (output/"predictions.json").is_file():
        print(f"Evaluation already complete: {output / 'metrics.json'}", flush=True)
        print_evaluation_metrics(read_json(output/"metrics.json")["metrics"],dataset.records)
        print("[ok] Evaluation pipeline finished.")
        return
    print("******* Running evaluation *******",flush=True)
    if len(raw) < len(dataset):
        require_official_nms()
        if processes > 1:
            pending = [i for i,r in enumerate(dataset.records) if r["video_path"] not in raw]
            infer_parallel(args,pending,len(dataset),len(raw),processes)
            raw = committed_predictions(dataset.records, progress)
        else:
            device = assigned_device(args.device,args.device_ids,0,distributed=False)
            model = build_model(state["input_dim"], state["model_config"]).to(device)
            model.load_state_dict(state["model"], strict=True)
            print(f"[ok] Loaded model: {args.checkpoint}",flush=True)
            def commit(video_id, prediction):
                write_json(progress/(feature_id(video_id)+".json"),dict(video_id=video_id,prediction=prediction))
            raw = infer(model, dataset, cached=raw, on_video=commit,
                        show_progress=args.show_progress,description="Testing")
    if len(raw) != len(dataset):
        raise RuntimeError(f"Incomplete inference: {len(raw)}/{len(dataset)} videos committed; rerun with --resume")
    predictions = prediction_records(dataset.records, raw, state["score_threshold"])
    write_json(output / "raw_predictions.json", raw)
    write_json(output / "predictions.json", predictions)
    metrics = paper_metrics(args.annotations, predictions,verbose=True,
                            prediction_file=output/"predictions.json")
    write_json(output / "metrics.json", dict(metrics=metrics, variant=state["variant"],
        epoch=state["epoch"], score_threshold=state["score_threshold"],
        parameters=state["parameters"], annotation_sha256=sha256(args.annotations),
        selection=state.get("selection", "validation_best_ema"),
        checkpoint_sha256=sha256(args.checkpoint), feature_signature=dataset.signature))
    print(f"variant={state['variant']}; selection={state.get('selection', 'validation_best_ema')}; "
          f"score threshold={state['score_threshold']}")
    print(f"[ok] Evaluation done. Results saved to: {output}")
    print(f"   - predictions JSON: {output / 'predictions.json'}")
    print(f"   - raw predictions:  {output / 'raw_predictions.json'}")
    print(f"   - summary JSON:     {output / 'metrics.json'}")
    print_evaluation_metrics(metrics,dataset.records)
    print("[ok] Evaluation pipeline finished.")


def main():
    args = parse_args()
    if args.device_ids:
        validate_cuda_devices(args.device_ids)
    run_prediction(args)


if __name__ == "__main__":
    main()
