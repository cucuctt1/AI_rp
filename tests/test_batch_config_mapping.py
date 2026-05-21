from app.experiments.config_mapping import normalize_core_ga_config


def test_gui_batch_keys_are_mirrored_to_core_ga_keys():
    config = normalize_core_ga_config(
        {
            "pop_size": 77,
            "elite_size": 5,
            "tournament_size": 4,
        }
    )

    assert config["population_size"] == 77
    assert config["pop_size"] == 77
    assert config["elitism_k"] == 5
    assert config["elite_size"] == 5
    assert config["tournament_size"] == 4


def test_core_ga_keys_are_mirrored_to_gui_result_fields():
    config = normalize_core_ga_config(
        {
            "population_size": 88,
            "elitism_k": 6,
        }
    )

    assert config["pop_size"] == 88
    assert config["elite_size"] == 6
