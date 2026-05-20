import numpy as np

from ga.elitism import get_elites
from tsp_ga_app.operators import evolve_population


def test_core_elitism_preserves_best_individual():
    population = [[0, 1, 2], [2, 1, 0], [1, 0, 2]]
    fitnesses = np.array([0.1, 0.9, 0.2])

    elites = get_elites(population, fitnesses, 1)

    assert elites == [[2, 1, 0]]


def test_app_evolution_preserves_lowest_distance_elite():
    population = [[0, 1, 2], [0, 2, 1], [1, 0, 2]]
    dist_matrix = np.array(
        [
            [0.0, 1.0, 10.0],
            [1.0, 0.0, 1.0],
            [10.0, 1.0, 0.0],
        ]
    )
    distances = [12.0, 12.0, 21.0]

    next_population = evolve_population(
        population,
        dist_matrix,
        distances=distances,
        elite_size=1,
        crossover_rate=0.0,
        mutation_rate=0.0,
    )

    assert next_population[0] == [0, 1, 2]
