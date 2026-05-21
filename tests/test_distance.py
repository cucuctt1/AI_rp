import numpy as np
import pytest

from tsp_ga_app.problem import compute_distance_matrix, fitness, route_distance


def test_distance_matrix_and_route_distance_are_euclidean_cycle():
    cities = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 0.0]])
    dist_matrix = compute_distance_matrix(cities)

    assert dist_matrix[0, 1] == 5.0
    assert dist_matrix[1, 2] == 5.0
    assert dist_matrix[0, 2] == 6.0
    assert route_distance([0, 1, 2], dist_matrix) == 16.0


def test_fitness_zero_distance_returns_zero():
    cities = np.array([[0.0, 0.0], [0.0, 0.0]])
    dist_matrix = compute_distance_matrix(cities)

    assert fitness([0, 1], dist_matrix) == 0.0


def test_route_distance_rejects_invalid_route():
    dist_matrix = np.zeros((3, 3))

    with pytest.raises(ValueError):
        route_distance([0, 0, 1], dist_matrix)
