"""Single-GPU vLLM judge, based on vLLM 0.19.1 structured_outputs.py.

Reference:
https://github.com/vllm-project/vllm/blob/v0.19.1/examples/offline_inference/structured_outputs.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from text_rq_data import judge_messages, required_stages, score_schema, validate_stage
from text_rq_storage import (
    assemble_result, atomic_json, checked_request, checked_result, fingerprint,
    read_json, request_path, result_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    manifest = read_json(output / "run_config.json")
    config, run_id = manifest["config"], manifest["run_id"]
    tasks = read_json(output / "tasks.json")
    tasks = [task for task in tasks if task["sample_index"] % args.world_size == args.rank]
    pending = []
    for task in tasks:
        path = result_path(output, task)
        if path.exists():
            checked_result(path, task, run_id)
        else:
            pending.append(task)
    progress = output / f"progress_rank_{args.rank}.json"
    atomic_json(progress, {"new_completed": 0})
    completed = 0
    jobs = []
    for task in pending:
        missing = []
        for stage in required_stages(task):
            path = request_path(output, task, stage)
            if path.exists():
                checked_request(path, task, stage, run_id)
            else:
                missing.append((task, stage))
        if missing:
            jobs.extend(missing)
        else:
            atomic_json(result_path(output, task), assemble_result(output, task, run_id))
            completed += 1
    atomic_json(progress, {"new_completed": completed})
    if not jobs:
        return

    # CUDA visibility and offline mode are set by the launcher before imports.
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    settings = config["inference"]
    llm = LLM(
        model=config["model"]["path"], tokenizer=config["model"]["path"],
        dtype="bfloat16", tensor_parallel_size=1, seed=42,
        trust_remote_code=False, language_model_only=True,
        max_model_len=settings["max_model_len"],
        gpu_memory_utilization=settings["gpu_memory_utilization"],
        max_num_seqs=settings["batch_size"],
        max_num_batched_tokens=settings["max_num_batched_tokens"],
        enable_prefix_caching=True, enable_chunked_prefill=True,
        generation_config="vllm",
        structured_outputs_config={"backend": "xgrammar"},
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()
    parameters = {
        stage: SamplingParams(
            temperature=settings["temperature"], top_p=settings["top_p"],
            top_k=settings["top_k"], seed=42,
            max_tokens=settings["max_new_tokens"],
            structured_outputs=StructuredOutputsParams(json=score_schema(stage)),
        ) for stage in ("fluency", "consistency", "factual")
    }
    for offset in range(0, len(jobs), settings["batch_size"]):
        batch = jobs[offset:offset + settings["batch_size"]]
        prompts = []
        for task, stage in batch:
            messages = judge_messages(task, config["prompts"], stage)
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            if len(tokens) + settings["max_new_tokens"] > settings["max_model_len"]:
                raise ValueError(
                    f"{task['id']}: input {len(tokens)} + output budget "
                    f"{settings['max_new_tokens']} exceeds --max-model-len; no truncation"
                )
            prompts.append({"prompt_token_ids": tokens})
        started = time.monotonic()
        responses = llm.generate(
            prompts, sampling_params=[parameters[stage] for _, stage in batch],
            use_tqdm=False,
        )
        if len(responses) != len(batch):
            raise RuntimeError("vLLM returned a different number of responses")
        elapsed = time.monotonic() - started
        for (task, stage), response in zip(batch, responses):
            if len(response.outputs) != 1:
                raise RuntimeError(f"{task['id']}: expected one judge output")
            completion = response.outputs[0]
            raw = completion.text
            record = {
                "id": task["id"], "stage": stage,
                "run_id": run_id, "task_sha256": fingerprint(task),
                "judge_output": raw, "finish_reason": completion.finish_reason,
                "input_tokens": len(response.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "batch_size": len(batch), "batch_seconds": elapsed,
            }
            try:
                if completion.finish_reason != "stop":
                    raise ValueError(f"Judge output did not finish: {completion.finish_reason}")
                scores = json.loads(raw)
                validate_stage(scores, stage)
            except (ValueError, TypeError) as exc:
                # Preserve the exact failed response; never replace it with zero.
                atomic_json(output / "failures" / f"sample_{task['sample_index']:08d}_{stage}.json",
                            {**record, "error": str(exc)})
                raise RuntimeError(f"{task['id']}: invalid judge output; see failures/") from exc
            atomic_json(request_path(output, task, stage), {**record, "scores": scores})
            if all(request_path(output, task, name).exists() for name in required_stages(task)):
                atomic_json(result_path(output, task), assemble_result(output, task, run_id))
                completed += 1
                atomic_json(progress, {"new_completed": completed})


if __name__ == "__main__":
    main()
