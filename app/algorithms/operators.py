import random
from typing import List, Optional, Sequence

import numpy as np

from app.config.settings import (
    CROSSOVER_RATE,
    ELITE_SIZE,
    MUTATION_RATE,
    NUM_CITIES,
    POP_SIZE,
    TOURNAMENT_SIZE,
)
from app.algorithms.problem import route_distance


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


def tournament_selection_fitness(population, fitnesses, k=3):
    """Pick one parent by tournament selection where larger fitness wins."""
    indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmax(fitnesses[indices])]
    return population[best_idx]


def roulette_selection(population, fitnesses):
    """Pick one parent by roulette-wheel sampling over fitness values."""
    total_fit = np.sum(fitnesses)
    if total_fit == 0:
        return population[np.random.randint(len(population))]
    probs = fitnesses / total_fit
    chosen_idx = np.random.choice(len(population), p=probs)
    return population[chosen_idx]


def pmx_crossover(parent1, parent2):
    size = len(parent1)

    cx1, cx2 = sorted(random.sample(range(size), 2))

    child = [-1] * size
    child[cx1:cx2] = parent1[cx1:cx2]

    for i in range(cx1, cx2):
        val = parent2[i]

        if val not in child:
            pos = i

            while True:
                mapped_val = parent1[pos]
                pos = parent2.index(mapped_val)

                if child[pos] == -1:
                    child[pos] = val
                    break

    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child


def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]

    p2_idx = (end + 1) % size
    c_idx = (end + 1) % size
    count = 0
    while -1 in child:
        if parent2[p2_idx] not in child:
            child[c_idx] = parent2[p2_idx]
            c_idx = (c_idx + 1) % size
        p2_idx = (p2_idx + 1) % size

        count += 1
        if count > 10000:
            raise RuntimeError("Order Crossover loop detected. Check implementation.")

    return child


def swap_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


def reverse_mutation(route, mutation_rate):
    """Inversion mutation for TSP routes."""
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        route[idx1:idx2 + 1] = reversed(route[idx1:idx2 + 1])
    return route


def scramble_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        subset = route[idx1:idx2 + 1]
        random.shuffle(subset)
        route[idx1:idx2 + 1] = subset
    return route


def get_elites(population, fitnesses, k):
    """Return the k best individuals where larger fitness is better."""
    if k <= 0:
        return []
    elite_indices = np.argsort(fitnesses)[-k:]
    return [population[i].copy() for i in elite_indices]


def two_opt_search(route, dist_matrix, max_improvements=50):
    """Apply 2-opt local search to refine a TSP route."""
    best_route = route.copy()
    n = len(best_route)
    improved = True
    iterations = 0

    while improved and iterations < max_improvements:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue

                a = best_route[i - 1]
                b = best_route[i]
                c = best_route[j - 1]
                d = best_route[j % n]

                dist_before = dist_matrix[a][b] + dist_matrix[c][d]
                dist_after = dist_matrix[a][c] + dist_matrix[b][d]

                if dist_after < dist_before:
                    best_route[i:j] = list(reversed(best_route[i:j]))
                    improved = True
        iterations += 1
    return best_route
