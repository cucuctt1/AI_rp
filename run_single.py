import numpy as np

from app.algorithms.problem import compute_distance_matrix
from app.config.settings import DEFAULT_CONFIG
from app.experiments.runner import run_single_experiment


if __name__ == "__main__":
    np.random.seed(42)
    cities = np.random.uniform(0.0, 100.0, size=(50, 2))
    dist_matrix = compute_distance_matrix(cities)

    config = DEFAULT_CONFIG.copy()
    config["generations"] = 200
    config["population_size"] = 100
    config["local_search_freq"] = 10

    run_single_experiment(
        "single_test_ox_swap",
        config,
        dist_matrix,
        cities=cities,
        seed=99,
        dataset_name="generated_random_50_seed_42",
        coordinate_source_or_seed="numpy.random.seed(42); uniform(0,100)",
        known_optimum="N/A",
    )
