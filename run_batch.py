import numpy as np

from app.algorithms.problem import compute_distance_matrix
from app.config.settings import DEFAULT_CONFIG
from app.experiments.batch_runner import run_grid_search


if __name__ == "__main__":
    np.random.seed(42)
    cities = np.random.uniform(0.0, 100.0, size=(30, 2))
    dist_matrix = compute_distance_matrix(cities)

    base_config = DEFAULT_CONFIG.copy()
    base_config["generations"] = 150
    base_config["population_size"] = 50

    param_grid = {
        "mutation_rate": [0.01, 0.05, 0.1],
        "crossover_type": ["pmx", "order"],
        "selection_type": ["tournament", "roulette"],
    }

    run_grid_search(
        base_config,
        param_grid,
        dist_matrix,
        cities=cities,
        base_experiment_name="grid_search",
        num_trials=2,
        dataset_name="generated_random_30_seed_42",
        coordinate_source_or_seed="numpy.random.seed(42); uniform(0,100)",
        known_optimum="N/A",
    )
