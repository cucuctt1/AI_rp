from utils.results_reporting import build_optimality_fields


def test_optimality_gap_formula():
    fields = build_optimality_fields(best_distance=110.0, known_optimum=100.0)

    assert fields["optimality_gap"] == 10.0


def test_unknown_optimum_is_labeled_na():
    fields = build_optimality_fields(best_distance=110.0, known_optimum="N/A")

    assert fields["optimality_gap"] == "N/A"
    assert fields["optimality_gap_reason"]
