"""Reject oversized inputs before a trainer can drop or resample any record."""

import argparse
import copy

from demo_utils import read_rows, require_root, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage", choices=["sft", "opsd", "grpo"], required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    require_root()
    import video_plugin  # noqa: F401
    from swift.model import get_processor
    from swift import get_template
    from tqdm import tqdm
    processor = get_processor(args.model, model_type="qwen3_5", download_model=False)
    template = get_template(processor, enable_thinking=False)
    template.set_mode("train" if args.stage == "sft" else "transformers")
    records = []
    for row in tqdm(read_rows(args.dataset), desc="Validate all input lengths", unit="proposal"):
        views = [("student", row)]
        if args.stage == "opsd":
            teacher = copy.deepcopy(row)
            teacher["messages"] = [{"role": "user", "content": row["teacher_prompt"]}]
            views.append(("teacher", teacher))
        for role, view in views:
            encoded = template.encode(copy.deepcopy(view))
            length = len(encoded["input_ids"])
            reserved = 0 if args.stage == "sft" else args.max_new_tokens
            if length + reserved > args.max_length:
                raise ValueError(f"{row['sample_id']} {role}: {length}+{reserved} > {args.max_length}; increase --max-length explicitly.")
            records.append({"sample_id": row["sample_id"], "role": role, "tokens": length})
    write_json(args.output, {"passed": True, "max_length": args.max_length, "records": records})


if __name__ == "__main__":
    main()
