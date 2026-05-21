import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.paths import OUTPUT_ROOT

FIGURE_DIR = os.path.join(str(OUTPUT_ROOT), "report_figures")


def read_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required input file not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_float(value: str):
    try:
        if value is None or str(value).strip().upper() == "N/A" or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def save_best_distance_by_run(raw_rows: List[Dict[str, str]]) -> None:
    rows = [row for row in raw_rows if parse_float(row.get("best_distance")) is not None]
    if not rows:
        return

    labels = [row.get("run_id") or row.get("experiment_name") or str(index + 1) for index, row in enumerate(rows)]
    values = [float(row["best_distance"]) for row in rows]

    width = max(8, min(24, len(values) * 0.45))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(values)), values, color="tab:blue")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Best distance")
    ax.set_title("Best distance by run")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "best_distance_by_run.png"), dpi=150)
    plt.close(fig)


def save_summary_mean(summary_rows: List[Dict[str, str]]) -> None:
    rows = [row for row in summary_rows if parse_float(row.get("mean")) is not None]
    if not rows:
        return

    labels = []
    means = []
    yerr_low = []
    yerr_high = []
    for row in rows:
        label_parts = [
            row.get("experiment_name", ""),
            row.get("algorithm", ""),
            row.get("mutation_rate", ""),
            row.get("crossover_type", ""),
            row.get("selection_type", ""),
        ]
        labels.append(" | ".join(part for part in label_parts if part))
        mean = float(row["mean"])
        means.append(mean)
        low = parse_float(row.get("ci95_low"))
        high = parse_float(row.get("ci95_high"))
        yerr_low.append(mean - low if low is not None else 0.0)
        yerr_high.append(high - mean if high is not None else 0.0)

    width = max(8, min(24, len(means) * 0.55))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(means)), means, yerr=[yerr_low, yerr_high], color="tab:green", capsize=3)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Mean best distance")
    ax.set_title("Summary statistics with 95% CI")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "summary_statistics_ci.png"), dpi=150)
    plt.close(fig)


def save_dataset_metadata(metadata_rows: List[Dict[str, str]]) -> None:
    rows = [row for row in metadata_rows if parse_float(row.get("n_cities")) is not None]
    if not rows:
        return

    labels = [row.get("dataset_name") or str(index + 1) for index, row in enumerate(rows)]
    values = [float(row["n_cities"]) for row in rows]

    width = max(6, min(18, len(values) * 0.6))
    fig, ax = plt.subplots(figsize=(width, 4))
    ax.bar(range(len(values)), values, color="tab:orange")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("City count")
    ax.set_title("Dataset metadata")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "dataset_city_counts.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    raw_rows = read_rows(os.path.join(OUTPUT_ROOT, "raw_results.csv"))
    summary_rows = read_rows(os.path.join(OUTPUT_ROOT, "summary_statistics.csv"))
    metadata_rows = read_rows(os.path.join(OUTPUT_ROOT, "dataset_metadata.csv"))

    save_best_distance_by_run(raw_rows)
    save_summary_mean(summary_rows)
    save_dataset_metadata(metadata_rows)


if __name__ == "__main__":
    main()
