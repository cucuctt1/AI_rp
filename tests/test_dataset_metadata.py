import csv

from utils.results_reporting import DATASET_METADATA_FIELDS, upsert_dataset_metadata


def test_dataset_metadata_contains_required_fields(tmp_path):
    upsert_dataset_metadata(
        dataset_name="unit_dataset",
        n_cities=4,
        coordinate_source_or_seed="unit-test",
        distance_metric="euclidean",
        known_optimum="N/A",
        output_root=str(tmp_path),
    )

    with open(tmp_path / "dataset_metadata.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        saved = next(reader)

    for field in DATASET_METADATA_FIELDS:
        assert field in saved
        assert saved[field] != ""
