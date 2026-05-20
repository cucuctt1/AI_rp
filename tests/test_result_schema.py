import csv

from utils.results_reporting import RAW_RESULT_REQUIRED_FIELDS, append_raw_result


def test_raw_result_rows_contain_required_fields(tmp_path):
    row = {field: "value" for field in RAW_RESULT_REQUIRED_FIELDS}
    row["best_distance"] = 1.0
    row["runtime_seconds"] = 0.01
    row["fitness_evaluations"] = 10

    append_raw_result(row, output_root=str(tmp_path))

    with open(tmp_path / "raw_results.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        saved = next(reader)

    for field in RAW_RESULT_REQUIRED_FIELDS:
        assert field in saved
        assert saved[field] != ""
