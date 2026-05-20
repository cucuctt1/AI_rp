import csv
import json
import math
import os
import statistics
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RAW_RESULT_REQUIRED_FIELDS = [
    "experiment_name",
    "algorithm",
    "run_id",
    "seed",
    "dataset_name",
    "n_cities",
    "pop_size",
    "generations",
    "crossover_type",
    "mutation_type",
    "selection_type",
    "mutation_rate",
    "elitism_k",
    "best_distance",
    "generation_found",
    "runtime_seconds",
    "fitness_evaluations",
]

RAW_RESULT_OPTIONAL_FIELDS = [
    "base_seed",
    "git_commit",
    "coordinate_source_or_seed",
    "distance_metric",
    "known_optimum",
    "known_optimum_note",
    "optimality_gap",
    "optimality_gap_reason",
    "nearest_neighbor_distance",
    "baseline_relative_improvement_percent",
]

RAW_RESULT_FIELDS = RAW_RESULT_REQUIRED_FIELDS + RAW_RESULT_OPTIONAL_FIELDS

SUMMARY_STATISTICS_FIELDS = [
    "experiment_name",
    "algorithm",
    "dataset_name",
    "n_runs",
    "mean",
    "std_dev",
    "min",
    "max",
    "confidence_interval_95",
    "ci95_low",
    "ci95_high",
    "statistical_test",
    "p_value",
    "effect_size",
    "comparison_target",
    "pop_size",
    "generations",
    "crossover_type",
    "mutation_type",
    "selection_type",
    "mutation_rate",
    "elitism_k",
]

DATASET_METADATA_FIELDS = [
    "dataset_name",
    "n_cities",
    "coordinate_source_or_seed",
    "distance_metric",
    "known_optimum",
    "known_optimum_note",
]


def to_json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    if callable(value):
        return "<callable>"
    return value


def _cell(value: Any) -> Any:
    value = to_json_ready(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return value


def _read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not os.path.isfile(path):
        return [], []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _cell(row.get(field, "")) for field in fieldnames})


def _write_json(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([to_json_ready(row) for row in rows], handle, indent=2)


def _merge_fields(preferred: Sequence[str], existing: Sequence[str], incoming: Sequence[str]) -> List[str]:
    fields: List[str] = []
    for field in list(preferred) + list(existing) + list(incoming):
        if field not in fields:
            fields.append(field)
    return fields


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "" or str(value).strip().upper() == "N/A":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def get_git_commit_hash(repo_root: Optional[str] = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root or os.getcwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = completed.stdout.strip()
        return commit or "N/A"
    except Exception:
        return "N/A"


def nearest_neighbor_route(dist_matrix: np.ndarray, start: int = 0) -> List[int]:
    matrix = np.asarray(dist_matrix, dtype=float)
    city_count = int(matrix.shape[0])
    if city_count == 0:
        return []

    start = int(start) % city_count
    route = [start]
    unvisited = set(range(city_count))
    unvisited.remove(start)

    while unvisited:
        current = route[-1]
        next_city = min(unvisited, key=lambda city: matrix[current, city])
        route.append(int(next_city))
        unvisited.remove(next_city)

    return route


def route_distance_from_matrix(route: Sequence[int], dist_matrix: np.ndarray) -> float:
    matrix = np.asarray(dist_matrix, dtype=float)
    if not route:
        return 0.0
    total = 0.0
    for index, city in enumerate(route):
        total += matrix[int(city), int(route[(index + 1) % len(route)])]
    return float(total)


def nearest_neighbor_distance(dist_matrix: np.ndarray, start: int = 0) -> float:
    route = nearest_neighbor_route(dist_matrix, start=start)
    return route_distance_from_matrix(route, dist_matrix)


def build_optimality_fields(
    best_distance: float,
    known_optimum: Any = "N/A",
    dist_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    optimum = _parse_float(known_optimum)
    if optimum is not None and optimum > 0:
        gap = ((float(best_distance) - optimum) / optimum) * 100.0
        return {
            "known_optimum": float(optimum),
            "known_optimum_note": "",
            "optimality_gap": float(gap),
            "optimality_gap_reason": "",
            "nearest_neighbor_distance": "",
            "baseline_relative_improvement_percent": "",
        }

    fields: Dict[str, Any] = {
        "known_optimum": "N/A",
        "known_optimum_note": "Known optimum is not available for this generated dataset.",
        "optimality_gap": "N/A",
        "optimality_gap_reason": (
            "Known optimum unavailable; baseline_relative_improvement_percent uses nearest-neighbor distance."
        ),
        "nearest_neighbor_distance": "N/A",
        "baseline_relative_improvement_percent": "N/A",
    }
    if dist_matrix is not None:
        baseline_distance = nearest_neighbor_distance(dist_matrix)
        fields["nearest_neighbor_distance"] = float(baseline_distance)
        if baseline_distance > 0:
            fields["baseline_relative_improvement_percent"] = (
                (float(baseline_distance) - float(best_distance)) / float(baseline_distance)
            ) * 100.0
    return fields


def upsert_dataset_metadata(
    dataset_name: str,
    n_cities: int,
    coordinate_source_or_seed: str,
    distance_metric: str = "euclidean",
    known_optimum: Any = "N/A",
    known_optimum_note: str = "Known optimum is not available for this generated dataset.",
    output_root: str = "outputs",
) -> None:
    path = os.path.join(output_root, "dataset_metadata.csv")
    json_path = os.path.join(output_root, "dataset_metadata.json")
    existing_fields, rows = _read_csv(path)
    fields = _merge_fields(DATASET_METADATA_FIELDS, existing_fields, [])

    new_row = {
        "dataset_name": dataset_name,
        "n_cities": int(n_cities),
        "coordinate_source_or_seed": coordinate_source_or_seed,
        "distance_metric": distance_metric,
        "known_optimum": known_optimum,
        "known_optimum_note": known_optimum_note,
    }

    replaced = False
    for index, row in enumerate(rows):
        if row.get("dataset_name") == str(dataset_name):
            rows[index] = {**row, **new_row}
            replaced = True
            break
    if not replaced:
        rows.append(new_row)

    _write_csv(path, fields, rows)
    _write_json(json_path, rows)


def append_raw_result(row: Dict[str, Any], output_root: str = "outputs") -> None:
    path = os.path.join(output_root, "raw_results.csv")
    json_path = os.path.join(output_root, "raw_results.json")
    existing_fields, rows = _read_csv(path)
    fields = _merge_fields(RAW_RESULT_FIELDS, existing_fields, row.keys())
    rows.append({field: row.get(field, "") for field in fields})
    _write_csv(path, fields, rows)
    _write_json(json_path, rows)


def update_summary_statistics(output_root: str = "outputs") -> None:
    raw_path = os.path.join(output_root, "raw_results.csv")
    _, raw_rows = _read_csv(raw_path)
    groups: Dict[Tuple[str, ...], List[float]] = {}
    group_fields = [
        "experiment_name",
        "algorithm",
        "dataset_name",
        "pop_size",
        "generations",
        "crossover_type",
        "mutation_type",
        "selection_type",
        "mutation_rate",
        "elitism_k",
    ]

    for row in raw_rows:
        best_distance = _parse_float(row.get("best_distance"))
        if best_distance is None:
            continue
        key = tuple(str(row.get(field, "")) for field in group_fields)
        groups.setdefault(key, []).append(best_distance)

    summary_rows: List[Dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        value_count = len(values)
        mean_value = statistics.fmean(values)
        std_dev = statistics.stdev(values) if value_count > 1 else 0.0
        margin = 1.96 * std_dev / math.sqrt(value_count) if value_count > 1 else 0.0
        low = mean_value - margin
        high = mean_value + margin
        group_values = dict(zip(group_fields, key))
        summary_rows.append(
            {
                **group_values,
                "n_runs": value_count,
                "mean": mean_value,
                "std_dev": std_dev,
                "min": min(values),
                "max": max(values),
                "confidence_interval_95": f"[{low:.6f}, {high:.6f}]",
                "ci95_low": low,
                "ci95_high": high,
                "statistical_test": "N/A",
                "p_value": "N/A",
                "effect_size": "N/A",
                "comparison_target": "N/A",
            }
        )

    csv_path = os.path.join(output_root, "summary_statistics.csv")
    json_path = os.path.join(output_root, "summary_statistics.json")
    _write_csv(csv_path, SUMMARY_STATISTICS_FIELDS, summary_rows)
    _write_json(json_path, summary_rows)
