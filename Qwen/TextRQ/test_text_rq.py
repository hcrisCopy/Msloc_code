"""CPU-only protocol tests; run on the server, without loading judge weights."""

from __future__ import annotations

import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_rq_data import (
    ALL_DIMENSIONS, STAGE_FIELDS, judge_messages, prepare_video,
    required_stages, score_schema, select_tasks, validate_scores, validate_stage,
)
from text_rq_metrics import aggregate, association_analysis, correlation, iou_pairs, summarize
from text_rq_report import build_report, format_report, report_csv
from text_rq_storage import (
    assemble_result, atomic_json, checked_request, checked_result, fingerprint,
    request_path, result_path,
)


def annotation(segment=None, domain="generator"):
    def entry(field, subject, description):
        return {"obj": subject, "bnd_class": "Appearance anomaly",
                "bnd_sub_class": "Disappearance", field: description}

    return {
        "segment": segment or [10, 20], "combine_dir": "Round3/test", "model": domain,
        "obj_cot": [
            entry("obj_caption", "ball", "The ball disappears."),
            entry("obj_caption", "shadow", "Its shadow remains."),
        ],
        "bnd_cot_st": [entry("bnd_caption", "ball", "The ball suddenly vanishes.")],
        "bnd_cot_ed": [entry("bnd_caption", "ball", "The ball reappears.")],
    }


def proposal(index=0, window=None, answer=None):
    return {
        "id": f"video.mp4::{index}", "video_path": "video.mp4",
        "proposal_index": index, "proposal": window or [8, 22],
        "raw_response": answer if answer is not None else (
            "<think>\n\n</think>\n\n"
            "Explanation: The ball disappears. Its shadow remains. The ball reappears.\n"
            "Interval: [2, 12]"
        ),
    }


def prepare(rows, annotations=None, source_type="fake", **metadata):
    return prepare_video(
        {"video_path": "video.mp4", "proposal_results": rows},
        {"video_path": "video.mp4", "type": source_type,
         "annotations": [annotation()] if annotations is None else annotations, **metadata},
        0.7,
    )


def scored(task, value=4):
    scores = dict.fromkeys(ALL_DIMENSIONS)
    for stage in required_stages(task):
        scores.update({key: {"score": value, "reason": "Specific reason."}
                       for key in STAGE_FIELDS[stage]})
    return {**task, "scores": scores, "score": validate_scores(scores, task)}


def prompts():
    root = Path(__file__).resolve().parents[1] / "prompts"
    return {name: {"text": (root / f"text_rq_{name}.txt").read_text(encoding="utf-8").strip()}
            for name in ("common", "fluency", "consistency", "factual")}


class ProtocolTests(unittest.TestCase):
    def test_repeated_gt_all_objects_and_no_one_to_one_deduplication(self):
        data = prepare([proposal(0), proposal(1)])
        self.assertEqual(len(data["tasks"]), 2)
        self.assertEqual([row["matched_gt_index"] for row in data["tasks"]], [0, 0])
        self.assertTrue(all(row["high_iou_fake"] for row in data["tasks"]))
        self.assertEqual(len(data["tasks"][0]["reference"]["main"]), 2)
        self.assertTrue(data["tasks"][0]["reference"]["boundary_reference_available"])

    def test_wrong_class_low_iou_and_exact_threshold_are_kept(self):
        data = prepare([
            proposal(0, answer="Explanation: The ball disappears.\nInterval: [2, 9]"),
            proposal(1, answer="Explanation: A ball moves normally.\nReal"),
            proposal(2, answer="Explanation: The ball disappears.\nInterval: [0, 1]"),
        ])
        self.assertEqual(len(data["tasks"]), 3)
        self.assertEqual([row["high_iou_fake"] for row in data["tasks"]], [False] * 3)
        self.assertEqual([row["iou"] for row in data["tasks"]], [0.7, None, 0.0])
        self.assertEqual(data["tasks"][1]["outcome"], "FN")
        self.assertIn("factual", required_stages(data["tasks"][1]))
        self.assertEqual(data["tasks"][2]["reference_policy"], "single_gt_in_proposal")

    def test_multi_gt_match_and_ambiguous_reference(self):
        annotations = [annotation([10, 20], "first"), annotation([40, 50], "second")]
        rows = [
            proposal(0, [0, 60], "Explanation: The ball disappears.\nInterval: [41, 49]"),
            proposal(1, [0, 60], "Explanation: Everything moves naturally.\nReal"),
            proposal(2, [0, 60], "Explanation: The ball disappears.\nInterval: [25, 30]"),
        ]
        tasks = prepare(rows, annotations)["tasks"]
        self.assertEqual(tasks[0]["matched_gt_index"], 1)
        self.assertEqual(tasks[0]["tool_domain"], "second")
        self.assertEqual(tasks[0]["iou"], 0.8)
        self.assertIsNone(tasks[1]["reference"])
        self.assertIsNone(tasks[2]["reference"])
        self.assertEqual(tasks[1]["tool_domain"], "mixed")
        self.assertEqual(tasks[1]["video_domains"], ["first", "second"])
        tied = prepare([proposal(0, [0, 30])], [annotation(), annotation()])["tasks"][0]
        self.assertIsNone(tied["reference"])
        self.assertEqual(tied["reference_policy"], "ambiguous_multiple_gt")

    def test_real_gt_false_positive_and_background_have_no_factual_reference(self):
        fake_prediction = prepare([proposal()], [], "real", tool_domain="camera")["tasks"][0]
        self.assertEqual(fake_prediction["outcome"], "FP")
        self.assertEqual(fake_prediction["tool_domain"], "camera")
        self.assertIsNone(fake_prediction["reference"])
        self.assertIsNone(fake_prediction["iou"])
        self.assertEqual(required_stages(fake_prediction), ["fluency", "consistency"])
        normal = proposal(window=[30, 35], answer="Explanation: A person walks naturally.\nReal")
        background = prepare([normal])["tasks"][0]
        self.assertTrue(background["correct_real"])
        self.assertEqual(background["tool_domain"], "generator")
        self.assertIsNone(background["reference"])
        real = prepare([normal], [], "real")["tasks"][0]
        self.assertEqual(real["tool_domain"], "real")

    def test_format_failure_keeps_exact_explanation_without_guessing_conclusion(self):
        tasks = prepare([
            proposal(0, answer="Explanation: A person walks.\nMaybe fake"),
            proposal(1, answer="Explanation: A person walks.\nInterval: [0, 99]"),
            proposal(2, answer="Unstructured prose with no Explanation field."),
        ])
        self.assertEqual(len(tasks["tasks"]), 2)
        malformed, out_of_range = tasks["tasks"]
        self.assertIsNone(malformed["prediction_kind"])
        self.assertEqual(required_stages(malformed), ["fluency", "factual"])
        self.assertEqual(out_of_range["prediction_kind"], "fake")
        self.assertIsNone(out_of_range["iou"])
        self.assertEqual(out_of_range["outcome"], "invalid")
        self.assertFalse(out_of_range["high_iou_fake"])
        self.assertEqual(tasks["audit"][2]["selection"], "missing_explanation")
        self.assertTrue(prepare([])["no_proposal"])

    def test_partial_and_round4_reference_boundary_policy(self):
        partial = proposal(window=[11, 20],
                           answer="Explanation: The ball is absent.\nInterval: [0, 9]")
        reference = prepare([partial])["tasks"][0]["reference"]
        self.assertEqual(reference["start"], [])
        self.assertEqual(reference["end"], [])
        round4 = annotation()
        round4["combine_dir"] = "Round4/test"
        self.assertEqual(prepare([proposal()], [round4])["tasks"][0]["reference"]["start"], [])

    def test_tool_domain_precedence_and_invalid_data_are_explicit(self):
        ann = annotation()
        ann["tool_domain"] = "official_domain"
        self.assertEqual(prepare([proposal()], [ann])["tasks"][0]["tool_domain"], "official_domain")
        del ann["obj_cot"][0]["obj"]
        with self.assertRaises(ValueError):
            prepare([proposal()], [ann])
        with self.assertRaises(ValueError):
            prepare([proposal(), proposal()])
        ann = annotation()
        del ann["model"]
        with self.assertRaises(KeyError):
            prepare([proposal()], [ann])

    def test_common_prompts_are_identical_and_information_is_isolated(self):
        templates = prompts()
        fake = prepare([proposal()])["tasks"][0]
        real = prepare([proposal(answer="Explanation: A person walks.\nReal")], [], "real")["tasks"][0]
        fake_fluency = judge_messages(fake, templates, "fluency")
        real_fluency = judge_messages(real, templates, "fluency")
        self.assertEqual(fake_fluency[0], real_fluency[0])
        self.assertEqual(fake_fluency[1]["content"], "Text A (Model Explanation):\n" + fake["explanation"])
        for stage in required_stages(fake):
            messages = judge_messages(fake, templates, stage)
            self.assertTrue(messages[0]["content"].startswith(templates["common"]["text"] + "\n\n"))
            self.assertEqual(set(score_schema(stage)["properties"]), set(STAGE_FIELDS[stage]))
        consistency = judge_messages(fake, templates, "consistency")[1]["content"]
        self.assertEqual(consistency, "Text A (Model Explanation):\n" + fake["explanation"]
                         + "\n\nPrediction conclusion:\nFake (Interval)")
        self.assertEqual(judge_messages(real, templates, "consistency")[1]["content"],
                         "Text A (Model Explanation):\n" + real["explanation"] + "\n\nPrediction conclusion:\nReal")
        factual = judge_messages(fake, templates, "factual")[1]["content"]
        self.assertNotIn("Prediction conclusion:", factual)
        self.assertNotIn("generator", factual)
        self.assertNotIn("[2, 12]", factual)
        self.assertIn("Reference scope: Start, Main, End.", factual)
        for phase in ("start", "main", "end"):
            for entry in fake["reference"][phase]:
                for value in entry.values():
                    self.assertIn(value, factual)
        self.assertEqual(judge_messages(fake, templates, "consistency")[0],
                         judge_messages(real, templates, "consistency")[0])
        with self.assertRaises(ValueError):
            judge_messages(real, templates, "factual")

    def test_main_only_prompt_keeps_all_objects_without_boundary_captions(self):
        templates = prompts()
        ann = annotation()
        ann["combine_dir"] = "Round4/test"
        for explanation in ("The ball disappears.",
                            "The ball disappears. Its shadow remains. The ball reappears."):
            task = prepare([proposal(answer=f"Explanation: {explanation}\nInterval: [2, 12]")],
                           [ann])["tasks"][0]
            message = judge_messages(task, templates, "factual")[1]["content"]
            self.assertTrue(message.startswith("Text A (Model Explanation):\n" + explanation
                                               + "\n\nText B (GT Reference Descriptions):"))
            self.assertIn("Reference scope: Main only.", message)
            self.assertIn("1. Subject: ball", message)
            self.assertIn("2. Subject: shadow", message)
            self.assertIn("Description: Its shadow remains.", message)
            self.assertNotIn("Start:", message)
            self.assertNotIn("End:", message)
            self.assertNotIn("The ball suddenly vanishes.", message)

    def test_consistency_is_not_in_rq_and_missing_facts_never_become_zero(self):
        fake = prepare([proposal()])["tasks"][0]
        result = scored(fake, 4)
        result["scores"]["conclusion_consistency"]["score"] = 1
        self.assertEqual(validate_scores(result["scores"], fake), 4)
        real = prepare([proposal(answer="Explanation: A person walks.\nReal")], [], "real")["tasks"][0]
        real_result = scored(real, 5)
        self.assertIsNone(real_result["score"])
        self.assertIsNone(real_result["scores"]["object_accuracy"])
        invalid = copy.deepcopy(result["scores"])
        invalid["fluency"]["score"] = True
        with self.assertRaises(ValueError):
            validate_scores(invalid, fake)
        with self.assertRaises(ValueError):
            validate_stage({"fluency": {"score": 4, "reason": ""}}, "fluency")
        with self.assertRaises(ValueError):
            validate_scores(result["scores"], real)

    def test_proposal_weighting_and_domain_views(self):
        data = prepare([proposal(0), proposal(1)])
        rows = [scored(data["tasks"][0], 2), scored(data["tasks"][1], 4)]
        third = copy.deepcopy(rows[1])
        third["video_path"] = "another.mp4"
        third["scores"] = {name: {"score": 5, "reason": "Reason."} for name in ALL_DIMENSIONS}
        third["score"] = 5
        rows.append(third)
        self.assertAlmostEqual(aggregate(rows)["rq_mean"], 11 / 3)
        summary = summarize(rows, data["audit"], 1)
        self.assertEqual(summary["all"]["video_count"], 2)
        self.assertEqual(summary["by_tool_domain"]["generator"]["high_iou_fake"]["count"], 3)
        self.assertEqual(summary["videos_without_proposals"], 1)
        real = prepare([proposal(answer="Explanation: A person walks.\nReal")], [], "real")
        mixed = summarize(rows + [scored(real["tasks"][0], 5)],
                          data["audit"] + real["audit"], 0)
        self.assertEqual(mixed["all"]["count"], 4)
        self.assertEqual(mixed["all"]["rq_count"], 3)
        self.assertEqual(mixed["by_tool_domain"]["real"]["all"]["rq_count"], 0)
        self.assertEqual(mixed["by_tool_domain"]["real"]["all"]["dimensions"]["fluency"]["mean"], 5)

    def test_report_macro_missing_scores_and_threshold_without_mutation(self):
        fake = prepare([proposal(0), proposal(1), proposal(2)])
        real = prepare([proposal(3, answer="Explanation: A person walks.\nReal")], [], "real")
        tasks = fake["tasks"] + real["tasks"]
        tasks[2]["tool_domain"] = "second_tool"
        tasks[0]["iou"] = 0.8
        rows = [scored(task, value) for task, value in zip(tasks, (2, 4, 5, 5))]
        audit = [{key: value for key, value in task.items()
                  if key not in {"explanation", "reference", "raw_response"}} for task in tasks]
        original = copy.deepcopy((rows, audit))
        report = build_report(rows, audit, 1, 0.9)
        self.assertAlmostEqual(report["overall"]["all"]["rq_mean"], 11 / 3)
        self.assertEqual(report["macro_group"]["RQ"]["mean"], 4)
        self.assertEqual(report["macro_group"]["RQ"]["group_count"], 2)
        self.assertEqual(report["macro_group"]["RQ"]["excluded_groups"], ["real"])
        self.assertEqual(report["macro_group"]["fluency"]["group_count"], 3)
        self.assertEqual(report["overall"]["high_iou_fake"]["count"], 2)
        self.assertEqual((rows, audit), original)
        grouped = build_report(rows, audit, 1, 0.7, "outcome")
        self.assertEqual(set(grouped["groups"]), {"TP", "TN"})
        self.assertEqual(grouped["overall"]["all"], report["overall"]["all"])

    def test_report_print_csv_and_duplicate_protection(self):
        data = prepare([proposal()])
        rows = [scored(data["tasks"][0])]
        report = build_report(rows, data["audit"], 0, 0.7)
        text = format_report(report)
        for label in ("--- Domain: generator ---", "Macro_domain", "Overall_proposal",
                      "object_accuracy", "anomaly_type_accuracy", "detail_consistency", "Valid N"):
            self.assertIn(label, text)
        csv_rows = list(csv.DictReader(io.StringIO(report_csv(report))))
        overall = [row for row in csv_rows if row["scope"] == "overall"
                   and row["view"] == "all" and row["metric"] == "RQ"]
        self.assertEqual(len(overall), 1)
        self.assertEqual(float(overall[0]["mean"]), 4)
        self.assertEqual(overall[0]["valid_n"], "1")
        with self.assertRaises(ValueError):
            build_report(rows * 2, data["audit"], 0, 0.7)

    def test_fixed_sampling_does_not_depend_on_input_order(self):
        items = [{"id": str(i), "kind": "real" if i < 2 else "fake"} for i in range(20)]
        self.assertEqual(select_tasks(items, 5), select_tasks(list(reversed(items)), 5))
        self.assertEqual({row["kind"] for row in select_tasks(items, 5)}, {"fake", "real"})
        with self.assertRaises(ValueError):
            select_tasks(items, 21)

    def test_iou_association_nulls_bins_and_cluster_bootstrap(self):
        rows = []
        for index, end in enumerate((4, 7, 9, 12)):
            task = prepare([proposal(answer=f"Explanation: The ball disappears.\nInterval: [2, {end}]")])["tasks"][0]
            task["video_path"] = f"video-{index // 2}.mp4"
            rows.append(scored(task, index + 1))
        pairs = iou_pairs(rows)
        stats = correlation(pairs, "rq", 40)
        self.assertAlmostEqual(stats["spearman"], 1)
        self.assertEqual(stats["video_count"], 2)
        self.assertEqual(stats, correlation(pairs, "rq", 40))
        report = association_analysis(rows, 0)
        self.assertEqual(report["all"]["iou_bins"]["(0.7,1]"]["count"], 1)
        self.assertEqual(report["all"]["iou_bins"]["(0.5,0.7]"]["count"], 1)
        self.assertIn("generator", report["by_tool_domain"])
        constant = [{**row, "rq": 4} for row in pairs]
        self.assertIsNone(correlation(constant, "rq", 0)["spearman"])
        real = prepare([proposal()], [], "real")["tasks"][0]
        self.assertEqual(iou_pairs([scored(real)]), [])

    def test_request_level_resume_identity_and_assembled_result(self):
        task = {**prepare([proposal()])["tasks"][0], "sample_index": 0}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for stage in required_stages(task):
                scores = {name: {"score": 4, "reason": "Reason."} for name in STAGE_FIELDS[stage]}
                record = {"id": task["id"], "run_id": "run-a", "task_sha256": fingerprint(task),
                          "stage": stage, "finish_reason": "stop", "scores": scores,
                          "judge_output": json.dumps(scores)}
                path = request_path(output, task, stage)
                atomic_json(path, record)
                checked_request(path, task, stage, "run-a")
                with self.assertRaises(ValueError):
                    checked_request(path, task, stage, "run-b")
            result = assemble_result(output, task, "run-a")
            path = result_path(output, task)
            atomic_json(path, result)
            checked_result(path, task, "run-a")
            with self.assertRaises(ValueError):
                checked_result(path, {**task, "explanation": "Changed."}, "run-a")
            result["scores"]["conclusion_consistency"]["score"] = 1
            atomic_json(path, result)
            with self.assertRaises(ValueError):
                checked_result(path, task, "run-a")


if __name__ == "__main__":
    unittest.main()
