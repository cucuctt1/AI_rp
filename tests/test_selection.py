import random

import numpy as np

from ga.selection import roulette_selection, tournament_selection
from tsp_ga_app.operators import tournament_selection as app_tournament_selection


def assert_valid_route(route, size):
    assert len(route) == size
    assert sorted(route) == list(range(size))


def test_selection_returns_valid_individuals():
    population = [list(np.random.permutation(6)) for _ in range(8)]
    fitnesses = np.linspace(1.0, 8.0, num=8)
    distances = list(reversed(fitnesses))

    np.random.seed(5)
    assert_valid_route(tournament_selection(population, fitnesses, k=3), 6)

    np.random.seed(5)
    assert_valid_route(roulette_selection(population, fitnesses), 6)

    random.seed(5)
    assert_valid_route(app_tournament_selection(population, distances, tournament_size=3), 6)
