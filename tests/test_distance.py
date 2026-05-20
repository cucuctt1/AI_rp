import numpy as np

from tsp_ga_app.problem import compute_distance_matrix, route_distance


def test_distance_matrix_and_route_distance_are_euclidean_cycle():
    cities = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 0.0]])
    dist_matrix = compute_distance_matrix(cities)

    assert dist_matrix[0, 1] == 5.0
    assert dist_matrix[1, 2] == 5.0
    assert dist_matrix[0, 2] == 6.0
    assert route_distance([0, 1, 2], dist_matrix) == 16.0
