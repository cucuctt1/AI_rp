import random

import numpy as np

from core.ga_engine import GAEngine
from tsp_ga_app.problem import compute_distance_matrix


def run_small_ga(seed):
    cities = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    config = {
        "population_size": 12,
        "generations": 20,
        "mutation_rate": 0.05,
        "crossover_type": "order",
        "mutation_type": "swap",
        "selection_type": "tournament",
        "elitism_k": 2,
        "adaptive_mutation": False,
        "local_search_freq": 0,
    }
    np.random.seed(seed)
    random.seed(seed)
    return GAEngine(config, compute_distance_matrix(cities)).run()


def test_same_seed_produces_same_best_distance():
    first = run_small_ga(123)
    second = run_small_ga(123)

    assert first["best_distance"] == second["best_distance"]
    assert first["best_route"] == second["best_route"]
