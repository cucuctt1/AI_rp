from typing import List, Optional, Tuple

import numpy as np

from .config import (
    CROSSOVER_RATE,
    ELITE_SIZE,
    GENERATIONS,
    MUTATION_RATE,
    POP_SIZE,
    TOURNAMENT_SIZE,
)
from .operators import create_population, evolve_population
from .problem import route_distance, set_distance_matrix


def genetic_algorithm(
    cities: np.ndarray,
    dist_matrix: np.ndarray,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    crossover_rate: float = CROSSOVER_RATE,
    elite_size: int = ELITE_SIZE,
    tournament_size: int = TOURNAMENT_SIZE,
) -> Tuple[List[int], float, List[float], List[List[int]], float]:
    """Run GA and return best solution, convergence history, and frame routes."""
    set_distance_matrix(dist_matrix)

    population = create_population(pop_size=pop_size, num_cities=len(cities))
    best_route: Optional[List[int]] = None
    best_distance = float("inf")

    best_distance_history: List[float] = []
    best_route_history: List[List[int]] = []

    initial_distances = [route_distance(route, dist_matrix) for route in population]
    initial_best_distance = float(min(initial_distances))

    for _ in range(generations):
        distances = [route_distance(route, dist_matrix) for route in population]
        generation_best_idx = int(np.argmin(distances))
        generation_best_route = list(population[generation_best_idx])
        generation_best_distance = float(distances[generation_best_idx])

        if generation_best_distance < best_distance:
            best_distance = generation_best_distance
            best_route = generation_best_route

        if best_route is None:
            raise RuntimeError("Best route tracking failed.")

        best_distance_history.append(best_distance)
        best_route_history.append(list(best_route))

        population = evolve_population(
            population=population,
            dist_matrix=dist_matrix,
            distances=distances,
            elite_size=elite_size,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        )

    return best_route, best_distance, best_distance_history, best_route_history, initial_best_distance
