import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    GENERATIONS,
    MUTATION_RATE,
    POP_SIZE,
)
from .problem import route_distance, set_distance_matrix

ProgressCallback = Optional[Callable[[Dict[str, object]], None]]

BAT_FREQUENCY_MIN = 0.0
BAT_FREQUENCY_MAX = 2.0
BAT_INITIAL_LOUDNESS = 0.9
BAT_INITIAL_PULSE_RATE = 0.15
BAT_ALPHA = 0.96
BAT_GAMMA = 0.9
BAT_MAX_GUIDED_MOVES = 8
BAT_LOCAL_WALK_SEGMENT = 5


def _emit_progress(
    progress_callback: ProgressCallback,
    generation: int,
    total_generations: int,
    best_route: Sequence[int],
    best_distance: float,
) -> None:
    if progress_callback is None:
        return

    payload: Dict[str, object] = {
        "backend": "bat",
        "phase": "bat",
        "generation": int(generation),
        "total_generations": int(total_generations),
        "restart_index": 1,
        "restart_count": 1,
        "best_route": list(best_route),
        "best_distance": float(best_distance),
    }
    try:
        progress_callback(payload)
    except RuntimeError:
        raise
    except Exception:
        return


def _validate_route(route: Sequence[int], city_count: int) -> None:
    if len(route) != city_count:
        raise ValueError("Route length does not match city count.")
    if set(route) != set(range(city_count)):
        raise ValueError("Route is not a valid city permutation.")


def _random_route(city_count: int) -> List[int]:
    route = list(range(city_count))
    random.shuffle(route)
    return route


def _hamming_distance(route_a: Sequence[int], route_b: Sequence[int]) -> int:
    return sum(1 for city_a, city_b in zip(route_a, route_b) if city_a != city_b)


def _random_inversion(route: Sequence[int]) -> List[int]:
    candidate = list(route)
    if len(candidate) < 2:
        return candidate

    left, right = sorted(random.sample(range(len(candidate)), 2))
    candidate[left : right + 1] = reversed(candidate[left : right + 1])
    return candidate


def _guided_move_towards_best(
    route: Sequence[int],
    global_best: Sequence[int],
    move_count: int,
) -> List[int]:
    candidate = list(route)
    city_count = len(candidate)

    if city_count < 2:
        return candidate

    iterations = max(1, min(int(move_count), city_count * 2))
    for _ in range(iterations):
        position = random.randrange(city_count)
        target_city = global_best[position]
        current_index = candidate.index(target_city)
        if current_index != position:
            candidate[position], candidate[current_index] = candidate[current_index], candidate[position]

        # Keep exploration alive.
        if random.random() < 0.25:
            left, right = random.sample(range(city_count), 2)
            candidate[left], candidate[right] = candidate[right], candidate[left]

    return candidate


def _local_walk_near_best(global_best: Sequence[int], segment_max: int) -> List[int]:
    candidate = list(global_best)
    city_count = len(candidate)

    if city_count < 2:
        return candidate

    segment_size = max(2, min(segment_max, city_count))
    left = random.randint(0, city_count - segment_size)
    right = left + segment_size
    candidate[left:right] = reversed(candidate[left:right])

    if city_count >= 4 and random.random() < 0.5:
        idx_a, idx_b = random.sample(range(city_count), 2)
        candidate[idx_a], candidate[idx_b] = candidate[idx_b], candidate[idx_a]

    return candidate


def bat_algorithm_tsp(
    cities: np.ndarray,
    dist_matrix: np.ndarray,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    crossover_rate: float = 0.0,
    elite_size: int = 0,
    tournament_size: int = 0,
    progress_callback: ProgressCallback = None,
) -> Tuple[List[int], float, List[float], List[List[int]], float]:
    """Run a bat-inspired metaheuristic for TSP (permutation-safe variant)."""
    _ = cities
    _ = crossover_rate
    _ = elite_size
    _ = tournament_size

    matrix = np.asarray(dist_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be a square matrix.")

    city_count = int(matrix.shape[0])
    if city_count < 2:
        raise ValueError("At least 2 cities are required.")

    set_distance_matrix(matrix)

    pop_size = max(4, int(pop_size))
    generations = max(1, int(generations))
    mutation_rate = max(0.0, min(1.0, float(mutation_rate)))

    population: List[List[int]] = [_random_route(city_count) for _ in range(pop_size)]
    distances = [route_distance(route, matrix) for route in population]

    initial_best_distance = float(min(distances))
    best_index = int(np.argmin(distances))
    best_route = list(population[best_index])
    best_distance = float(distances[best_index])

    velocities = [0.0 for _ in range(pop_size)]
    loudness = [BAT_INITIAL_LOUDNESS for _ in range(pop_size)]
    pulse_rates = [BAT_INITIAL_PULSE_RATE for _ in range(pop_size)]

    best_distance_history: List[float] = []
    best_route_history: List[List[int]] = []

    for generation_idx in range(generations):
        for bat_idx in range(pop_size):
            frequency = BAT_FREQUENCY_MIN + (BAT_FREQUENCY_MAX - BAT_FREQUENCY_MIN) * random.random()
            difference = _hamming_distance(population[bat_idx], best_route)
            velocities[bat_idx] += frequency * difference

            move_count = int(round(min(max(1.0, velocities[bat_idx]), float(BAT_MAX_GUIDED_MOVES))))
            candidate = _guided_move_towards_best(population[bat_idx], best_route, move_count)

            if random.random() > pulse_rates[bat_idx]:
                candidate = _local_walk_near_best(best_route, BAT_LOCAL_WALK_SEGMENT)

            if random.random() < mutation_rate:
                candidate = _random_inversion(candidate)

            _validate_route(candidate, city_count)
            candidate_distance = route_distance(candidate, matrix)

            current_distance = distances[bat_idx]
            improved = candidate_distance < current_distance
            accepted = improved and random.random() <= loudness[bat_idx]

            if accepted or candidate_distance < best_distance:
                population[bat_idx] = candidate
                distances[bat_idx] = candidate_distance

            if candidate_distance < best_distance:
                best_distance = float(candidate_distance)
                best_route = list(candidate)

            if accepted:
                loudness[bat_idx] = max(0.05, loudness[bat_idx] * BAT_ALPHA)
                pulse_update = 1.0 - math.exp(-BAT_GAMMA * (generation_idx + 1))
                pulse_rates[bat_idx] = min(1.0, BAT_INITIAL_PULSE_RATE + pulse_update * 0.5)

        best_distance_history.append(best_distance)
        best_route_history.append(list(best_route))

        _emit_progress(
            progress_callback=progress_callback,
            generation=generation_idx + 1,
            total_generations=generations,
            best_route=best_route,
            best_distance=best_distance,
        )

    return (
        list(best_route),
        float(best_distance),
        best_distance_history,
        best_route_history,
        float(initial_best_distance),
    )
