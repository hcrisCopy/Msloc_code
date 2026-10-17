"""Run resumable student/teacher generation or an actual vLLM input-alignment check."""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

from demo_utils import (check_versions, data_identity, device_ids, environment, model_identity,
                        prepare_output, read_rows, require_root, source_identity, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["student", "teacher", "alignment"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default="none")
    p.add_argument("--data", required=True)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--vllm-memory", type=float, default=0.5)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", choices=["none", "auto"], default="none")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    require_root()
    if args.worker:
        worker(args)
        return
    if args.max_new_tokens <= 0 or args.max_length <= args.max_new_tokens or not 0 < args.vllm_memory < 1:
        raise ValueError("Invalid generation length or vLLM memory fraction.")
    if args.mode == "alignment" and args.adapter != "none":
        raise ValueError("Alignment checks the base processor; pass --adapter none. Training still supports LoRA.")
    config = {"mode": args.mode, "weights": model_identity(args.model, args.adapter),
              "data": data_identity(args.data), "devices": device_ids(args.devices),
              "max_new_tokens": args.max_new_tokens, "max_length": args.max_length,
              "vllm_memory": args.vllm_memory, "source": source_identity(),
              "versions": check_versions(args.mode == "alignment")}
    output = prepare_output(args.output, config, args.resume, args.clean,
                            [args.model, args.data] + ([args.adapter] if args.adapter != "none" else []))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(device_ids(args.devices))}", __file__]
    for key in ("mode", "model", "adapter", "data", "devices", "max_new_tokens", "max_length", "vllm_memory", "output"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    command += ["--worker"]
    subprocess.run(command, env=environment(args.devices), check=True)
    rows = []
    for rank in range(len(device_ids(args.devices))):
        rows += read_rows(output / f"rank_{rank}.jsonl")
    expected = {r["sample_id"] for r in read_rows(Path(args.data) / "opsd.jsonl")}
    if len(rows) != len(expected) or {r["sample_id"] for r in rows} != expected:
        raise ValueError("Missing or duplicate inference results.")
    from demo_utils import write_rows
    write_rows(output / "predictions.jsonl", rows)
    if args.mode == "alignment":
        write_json(output / "alignment.json", {"passed": all(r["aligned"] for r in rows),
                   "config": config, "samples": len(rows)})
        if not all(r["aligned"] for r in rows):
            raise RuntimeError(f"HF/vLLM input mismatch. Inspect {output}/predictions.jsonl; training is blocked.")
    print(f"{args.mode}: {len(rows)} samples; results in {output}")


def worker(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank]
    # These are independent inference replicas, not a distributed model.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    import torch
    from tqdm import tqdm
    import video_plugin  # noqa: F401
    from swift import get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine, VllmEngine
    from common import parse_answer

    torch.cuda.set_device(0)
    torch.manual_seed(42)
    shard = Path(args.output) / f"rank_{rank}.jsonl"
    finished = read_rows(shard) if shard.exists() else []
    done = {r["sample_id"] for r in finished}
    if len(done) != len(finished):
        raise ValueError("Duplicate IDs in inference resume file.")
    samples = read_rows(Path(args.data) / "opsd.jsonl")
    targets = {r["id"]: r for r in read_rows(Path(args.data) / "targets.jsonl")}
    tasks = [s for i, s in enumerate(samples) if i % world == rank and s["sample_id"] not in done]
    if not tasks:
        shard.touch(exist_ok=True)
        return
    capture = {}
    if args.mode == "alignment":
        # This hook observes real vLLM preprocessing, not a reconstructed approximation.
        from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
        original = Qwen3VLMultiModalProcessor._call_hf_processor

        def capture_pixels(self, *values, **kwargs):
            result = original(self, *values, **kwargs)
            if "pixel_values_videos" in result:
                capture["pixels"] = result["pixel_values_videos"].detach().cpu()
                capture["grid"] = result["video_grid_thw"].detach().cpu()
            return result

        Qwen3VLMultiModalProcessor._call_hf_processor = capture_pixels
        engine = VllmEngine(args.model, torch_dtype=torch.bfloat16, use_async_engine=False,
                            gpu_memory_utilization=args.vllm_memory, max_model_len=args.max_length,
                            tensor_parallel_size=1, seed=42, mm_processor_cache_gb=0,
                            enable_prefix_caching=False, max_num_seqs=1)
    else:
        adapters = [] if args.adapter == "none" else [args.adapter]
        engine = TransformersEngine(args.model, adapters=adapters,
                                    adapter_names=["demo"] if adapters else None,
                                    torch_dtype=torch.bfloat16, attn_impl="flash_attn",
                                    device_map={"": "cuda:0"})
        if adapters:
            engine.model.set_adapter("demo")
    engine.template = get_template(engine.processor, enable_thinking=False)
    engine.template.set_mode("vllm" if args.mode == "alignment" else "transformers")
    request_config = RequestConfig(max_tokens=1 if args.mode == "alignment" else args.max_new_tokens,
                                   temperature=0, seed=42, return_details=True)
    import json
    with shard.open("a", encoding="utf-8") as stream:
        for sample in tqdm(tasks, desc=f"{args.mode} rank {rank}", unit="proposal", disable=rank != 0):
            torch.manual_seed(42)
            target = targets[sample["sample_id"]]
            messages = copy.deepcopy(sample["messages"])
            if args.mode == "teacher":
                messages = [{"role": "user", "content": sample["teacher_prompt"]}]
            request = InferRequest(messages=messages, videos=sample["videos"], chat_template_kwargs={"nframes": 40})
            record = {"sample_id": sample["sample_id"], "target_kind": target["target_kind"],
                      "target_relative": target["target_relative"]}
            if args.mode == "alignment":
                hf_template = copy.copy(engine.template)
                hf_template.set_mode("transformers")
                hf = hf_template.encode(copy.deepcopy(request))
                capture.clear()
                response = engine.infer([request], request_config=request_config)[0]
                hf_ids = hf["input_ids"]
                ids_match = list(hf_ids) == list(response.prompt_token_ids)
                if "pixels" not in capture:
                    raise RuntimeError("vLLM pixel observation did not fire; cannot certify alignment.")
                expected_pixels = hf["pixel_values_videos"].cpu()
                pixels_match = expected_pixels.shape == capture["pixels"].shape and torch.equal(expected_pixels, capture["pixels"])
                grid_match = torch.equal(hf["video_grid_thw"].cpu(), capture["grid"])
                record.update(aligned=bool(ids_match and pixels_match and grid_match),
                              token_ids_equal=ids_match, pixels_equal=pixels_match, grid_equal=grid_match,
                              hf_tokens=len(hf_ids), vllm_tokens=len(response.prompt_token_ids),
                              pixel_shape=list(expected_pixels.shape))
            else:
                encoded = engine.template.encode(copy.deepcopy(request))
                if len(encoded["input_ids"]) + args.max_new_tokens > args.max_length:
                    raise ValueError(f"{sample['sample_id']}: prompt plus generation exceeds --max-length.")
                response = engine.infer([request], request_config=request_config)[0]
                raw = response.choices[0].message.content
                if not isinstance(raw, str):
                    raise TypeError("Inference returned no text.")
                record.update(raw=raw, **parse_answer(raw, target["proposal"][1] - target["proposal"][0]))
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
