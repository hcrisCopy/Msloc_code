"""Terminal presentation following the original MSLoc-PG training/eval style."""
from pathlib import Path

from tqdm import tqdm


def memory_postfix():
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return dict(ram_gib=f"{int(line.split()[1])/1024**2:.2f}")
    return {}


def print_evaluation_metrics(metrics, records):
    """Same report structure, with explicit labels for the released paper metrics.

    PG's diagnostic F1 uses fake videos only. Do not label the paper evaluator's
    all-video/domain-macro F1 as that different statistic.
    """
    total = metrics["Total"]
    print("\n" + "="*60)
    print("Evaluation summary")
    print("="*60)
    print("Temporal segmentation evaluation")
    print("="*50)
    print("Frame-level metrics (0.01-second grid; released domain average):")
    print(f"  F1: {total['Loc_F1']:.4f}")
    print("\nVideo-level metrics:")
    print(f"  Classification accuracy: {total['Det_Acc']:.4f}")
    print("\nLocalization metrics (released domain average):")
    print(f"  Avg Loc F1: {total['Loc_IoU']:.4f}")
    print("\nDataset stats:")
    fake = sum(r["type"] == "fake" for r in records)
    print(f"  Total: {len(records)}  Fake: {fake}  Real: {len(records)-fake}")
    print("="*50)
