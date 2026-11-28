"""Small helpers shared by every experiment in experiments/<name>/run.py.

Folder rule: outputs/<dataset>/<experiment>/<part>/   (one folder per part)
             outputs/<dataset>/<experiment>/summary_<rate>.csv   (all parts together)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from .config import Config


def part_args(description: str, default_step: int) -> argparse.Namespace:
    """Standard command line: --part (one or more) and --step."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--part", nargs="+", required=True, help="one or more scene ids (part-...)")
    parser.add_argument("--step", type=int, default=default_step,
                        help="use every Nth moment (1 = every moment = 10 Hz)")
    return parser.parse_args()


def rate_tag(step: int) -> str:
    """Moments per second of driving that were evaluated, used in every output name: 10hz, 1hz, ..."""
    return f"{round(10 / step, 2):g}hz".replace(".", "p")


def experiment_dir(cfg: Config, name: str) -> Path:
    return cfg.output_dir / name


def ade3(pred_xy: np.ndarray, gt_xy: np.ndarray, valid: np.ndarray) -> float:
    """Average distance (m) between prediction and human over the first 3 s of valid steps."""
    n3 = int(valid[:30].sum())
    return float(np.linalg.norm(pred_xy - gt_xy, axis=1)[:n3].mean()) if n3 else float("nan")


def _value(text: str):
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


def read_results(path: Path) -> list[dict]:
    """Read a results_<rate>.csv back with numbers as floats and True/False as bools."""
    with path.open() as f:
        return [{k: _value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_summary(exp_dir: Path, rate: str, summarize) -> Path:
    """Combine every <part>/results_<rate>.csv of an experiment into summary_<rate>.csv.

    summarize(rows) -> dict of columns, called once per part and once for all parts pooled.
    Every part that has results at this rate is included, also parts from earlier runs.
    """
    table = []
    pooled = []
    for csv_path in sorted(exp_dir.glob(f"part-*/results_{rate}.csv")):
        rows = read_results(csv_path)
        pooled += rows
        table.append({"part": csv_path.parent.name, **summarize(rows)})
    if not table:
        raise SystemExit(f"No results_{rate}.csv found under {exp_dir}")
    table.append({"part": f"all ({len(table)} parts)", **summarize(pooled)})
    out = exp_dir / f"summary_{rate}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        for row in table:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\nSummary of all parts at {rate} ({out}):")
    for row in table:
        print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return out
