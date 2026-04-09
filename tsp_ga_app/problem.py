from typing import Optional, Sequence

import numpy as np

# Used by fitness(route) when no matrix is explicitly passed.
CURRENT_DISTANCE_MATRIX: Optional[np.ndarray] = None


def set_distance_matrix(dist_matrix: np.ndarray) -> None:
    """Set the active distance matrix used by fitness(route)."""
    global CURRENT_DISTANCE_MATRIX
    CURRENT_DISTANCE_MATRIX = dist_matrix


def generate_cities(n: int) -> np.ndarray:
    """Generate n random 2D city coordinates."""
    return np.random.uniform(0.0, 100.0, size=(n, 2))


def compute_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Compute a symmetric Euclidean distance matrix for all cities."""
    deltas = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def route_distance(route: Sequence[int], dist_matrix: np.ndarray) -> float:
    """Compute cycle length, including the return edge to the start city."""
    total = 0.0
    route_len = len(route)
    for i in range(route_len):
        total += dist_matrix[route[i], route[(i + 1) % route_len]]
    return float(total)


def fitness(route: Sequence[int], dist_matrix: Optional[np.ndarray] = None) -> float:
    """Fitness is inverse distance; larger is better."""
    matrix = dist_matrix if dist_matrix is not None else CURRENT_DISTANCE_MATRIX
    if matrix is None:
        raise ValueError("Distance matrix is not set for fitness evaluation.")
    return 1.0 / route_distance(route, matrix)
