"""比较同批 proposal 的 SFT 学生与特权教师完整生成评测，给 OPSD 准入结论。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-eval", required=True)
    parser.add_argument("--teacher-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    student_dir, teacher_dir = Path(args.student_eval), Path(args.teacher_eval)
    output = Path(args.output)
    if not output.resolve().is_relative_to(Path("../MSLoc_data/Qwen").resolve()):
        raise ValueError("教师准入结果必须写入 ../MSLoc_data/Qwen/ 下")
    if output.exists() and not args.clean:
        raise FileExistsError(f"结果已存在：{output}；重算时指定 --clean")
    student_config, teacher_config = load(student_dir / "eval_config.json"), load(teacher_dir / "eval_config.json")
    keys = ("model", "adapter", "adapter_config_sha256", "adapter_weights_sha256",
            "proposals", "proposals_sha256", "annotation", "annotation_sha256",
            "video_root", "prompt_text", "frames", "max_new_tokens",
            "temperature", "top_p", "top_k", "repetition_penalty")
    differences = [key for key in keys if student_config[key] != teacher_config[key]]
    if differences or student_config["teacher_prompt_file"] is not None or teacher_config["teacher_prompt_file"] is None:
        raise ValueError(f"学生与教师评测配置不匹配：{differences}")
    student_metrics, teacher_metrics = load(student_dir / "metrics.json"), load(teacher_dir / "metrics.json")
    student_parse, teacher_parse = load(student_dir / "parse_summary.json"), load(teacher_dir / "parse_summary.json")
    student_total, teacher_total = student_metrics["Total"], teacher_metrics["Total"]
    student_invalid = len(student_parse["invalid_proposal_ids"])
    teacher_invalid = len(teacher_parse["invalid_proposal_ids"])
    # 完整生成必须整体更强：定位两项上升，检测准确率与格式错误不退化。
    passed = (
        teacher_total["Loc_F1"] > student_total["Loc_F1"]
        and teacher_total["Loc_IoU"] > student_total["Loc_IoU"]
        and teacher_total["Det_Acc"] >= student_total["Det_Acc"]
        and teacher_invalid <= student_invalid
    )
    result = {
        "passed": passed,
        "criterion": "teacher Loc_F1 and Loc_IoU > student; Det_Acc >= student; invalid count <= student",
        "student": {"Total": student_total, "invalid_proposals": student_invalid},
        "teacher": {"Total": teacher_total, "invalid_proposals": teacher_invalid},
        "student_eval": str(student_dir),
        "teacher_eval": str(teacher_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("教师完整生成评测未通过准入；不得启动 OPSD")


if __name__ == "__main__":
    main()
