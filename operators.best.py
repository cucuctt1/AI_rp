import random
from typing import List, Optional, Sequence

import numpy as np

from .config import (
    CROSSOVER_RATE,
    ELITE_SIZE,
    MUTATION_RATE,
    NUM_CITIES,
    POP_SIZE,
    TOURNAMENT_SIZE,
)
from .problem import route_distance


def create_population(pop_size: int = POP_SIZE, num_cities: int = NUM_CITIES) -> List[List[int]]:
    """Create an initial population of random valid city permutations."""
    population: List[List[int]] = []
    base_route = list(range(num_cities))
    for _ in range(pop_size):
        route = base_route.copy()
        random.shuffle(route)
        population.append(route)
    return population


def tournament_selection(
    population: Sequence[Sequence[int]],
    distances: Sequence[float],
    tournament_size: int = TOURNAMENT_SIZE,
) -> List[int]:
    """Pick one parent via tournament selection (min distance wins)."""
    if not population:
        raise ValueError("Population cannot be empty.")

    k = min(tournament_size, len(population))
    candidate_indices = random.sample(range(len(population)), k)
    winner_index = min(candidate_indices, key=lambda idx: distances[idx])
    return list(population[winner_index])


def crossover_OX1(
    parent_a: Sequence[int],
    parent_b: Sequence[int],
    crossover_rate: float = CROSSOVER_RATE,
) -> List[int]:
    """Order Crossover (OX1) for permutation chromosomes."""
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same length.")

    size = len(parent_a)
    if size < 2 or random.random() > crossover_rate:
        return list(parent_a)

    left, right = sorted(random.sample(range(size), 2))

    child = [-1] * size
    child[left : right + 1] = parent_a[left : right + 1]

    taken = set(child[left : right + 1])
    insert_pos = (right + 1) % size

    # Fill remaining slots in the order parent_b appears, skipping duplicates.
    for gene in parent_b:
        if gene in taken:
            continue
        while child[insert_pos] != -1:
            insert_pos = (insert_pos + 1) % size
        child[insert_pos] = gene
        insert_pos = (insert_pos + 1) % size

    if -1 in child or len(set(child)) != size:
        raise ValueError("OX1 produced an invalid child permutation.")

    return child


def mutation_inversion(route: Sequence[int], mutation_rate: float = MUTATION_RATE) -> List[int]:
    """Inversion mutation: reverse one random subsegment."""
    mutated = list(route)
    if len(mutated) >= 2 and random.random() < mutation_rate:
        left, right = sorted(random.sample(range(len(mutated)), 2))
        mutated[left : right + 1] = reversed(mutated[left : right + 1])
    return mutated


def evolve_population(
    population: Sequence[Sequence[int]],
    dist_matrix: np.ndarray,
    distances: Optional[Sequence[float]] = None,
    elite_size: int = ELITE_SIZE,
    tournament_size: int = TOURNAMENT_SIZE,
    crossover_rate: float = CROSSOVER_RATE,
    mutation_rate: float = MUTATION_RATE,
) -> List[List[int]]:
    """Create the next generation with elitism, selection, crossover, and mutation."""
    pop_size = len(population)
    if pop_size == 0:
        raise ValueError("Population cannot be empty.")

    if distances is None:
        distances = [route_distance(route, dist_matrix) for route in population]

    city_count = len(population[0])
    elite_count = max(0, min(elite_size, pop_size))
    ranked_indices = sorted(range(pop_size), key=lambda idx: distances[idx])

    next_population: List[List[int]] = [
        list(population[idx]) for idx in ranked_indices[:elite_count]
    ]

    while len(next_population) < pop_size:
        parent_a = tournament_selection(population, distances, tournament_size)
        parent_b = tournament_selection(population, distances, tournament_size)
        child = crossover_OX1(parent_a, parent_b, crossover_rate)
        child = mutation_inversion(child, mutation_rate)

        if len(child) != city_count or len(set(child)) != city_count:
            raise ValueError("Evolution produced an invalid route permutation.")

        next_population.append(child)

    return next_population
