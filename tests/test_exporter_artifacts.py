import csv
import json

from app.reporting.exporter import Exporter


def test_metrics_export_preserves_extended_metric_columns(tmp_path):
    exporter = Exporter(output_root=tmp_path)
    folder = exporter.create_experiment_folder("metrics_export")

    exporter.save_metrics(
        folder,
        [
            {
                "generation": 1,
                "best_fitness": 0.5,
                "avg_fitness": 0.25,
                "worst_fitness": 0.1,
                "std_fitness": 0.02,
                "diversity": 9,
            }
        ],
    )

    with open(f"{folder}/metrics.csv", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["avg_fitness"] == "0.25"
    assert row["worst_fitness"] == "0.1"
    assert row["std_fitness"] == "0.02"
    assert row["diversity"] == "9"


def test_best_route_history_has_accurate_artifact_name(tmp_path):
    exporter = Exporter(output_root=tmp_path)
    folder = exporter.create_experiment_folder("route_history")

    exporter.save_best_route_history(folder, [[0, 1, 2], [0, 2, 1]])

    with open(f"{folder}/best_route_history.json", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved == [[0, 1, 2], [0, 2, 1]]
