"""Use the released paper evaluator, rather than introducing different F1 semantics."""
import contextlib
import importlib.util
import io
import tempfile
from pathlib import Path

from common import write_json


def prediction_records(records, raw_predictions, threshold):
    result = []
    for record in records:
        raw = raw_predictions[record["video_path"]]
        selected = [(segment, float(score)) for segment, score in zip(raw["segments"], raw["scores"])
                    if score >= threshold and segment[1] > segment[0]]
        item = dict(record)
        item["model_inference"] = dict(type="fake" if selected else "real",
                                       segment=[list(s) for s, _ in selected],
                                       scores=[v for _, v in selected])
        result.append(item)
    return result


def paper_metrics(annotations, predictions, verbose=False, prediction_file=None):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_long.py"
    spec = importlib.util.spec_from_file_location("_msloc_released_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if prediction_file is not None:
        with (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())):
            return module.evaluate(str(annotations),str(prediction_file))
    with tempfile.TemporaryDirectory(prefix="msloc_af_metrics_") as folder:
        output = Path(folder) / "predictions.json"
        write_json(output, predictions)
        with (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())):
            result = module.evaluate(str(annotations), str(output))
    return result


def choose_threshold(dataset, raw_predictions, thresholds):
    trials = []
    for threshold in thresholds:
        predictions = prediction_records(dataset.records, raw_predictions, threshold)
        metrics = paper_metrics(dataset.annotations, predictions)
        trials.append(dict(threshold=float(threshold), metrics=metrics))
    # Loc_IoU is the released name of paper F1_Loc. Deterministic tie break:
    # F1_Det then larger threshold (fewer false positives).
    best = max(trials, key=lambda x: (x["metrics"]["Total"]["Loc_IoU"],
                                     x["metrics"]["Total"]["Loc_F1"], x["threshold"]))
    return best, trials
