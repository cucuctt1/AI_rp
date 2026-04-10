import inspect
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from simpleai.search import local as simpleai_local
from simpleai.search.local import genetic
from simpleai.search.models import SearchNodeValueOrdered, SearchProblem
from simpleai.search.utils import InverseTransformSampler
from simpleai.search.viewers import BaseViewer

from .config import (
    CROSSOVER_RATE,
    ELITE_SIZE,
    GENERATIONS,
    MUTATION_RATE,
    POP_SIZE,
    SIMPLEAI_2OPT_MAX_PASSES,
    SIMPLEAI_DIVERSITY_RATE,
    SIMPLEAI_ENABLE_2OPT,
    SIMPLEAI_ENABLE_ELITISM,
    SIMPLEAI_EPSILON,
    SIMPLEAI_FITNESS_POWER,
    SIMPLEAI_RESTARTS,
    SIMPLEAI_USE_NATIVE_GENETIC,
    TOURNAMENT_SIZE,
)
from .operators import crossover_OX1
from .problem import route_distance, set_distance_matrix


ProgressCallback = Optional[Callable[[Dict[str, object]], None]]


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded_power(value: float, lower: float = 0.5, upper: float = 2.0) -> float:
    return max(lower, min(upper, float(value)))


def _sanitize_distance_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(dist_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Distance matrix contains non-finite values.")

    # Enforce symmetry and exact zeros on the diagonal.
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)

    if np.any(matrix < -1e-12):
        raise ValueError("Distance matrix contains negative entries.")

    matrix[matrix < 0.0] = 0.0
    return matrix


def _validate_route(route: Sequence[int], city_count: int) -> None:
    if len(route) != city_count:
        raise ValueError("Route length does not match city count.")

    route_set = set(route)
    if route_set != set(range(city_count)):
        raise ValueError("Route is not a valid permutation of city indices.")


def _emit_progress(
    progress_callback: ProgressCallback,
    backend: str,
    phase: str,
    generation: int,
    total_generations: int,
    best_route: Sequence[int],
    best_distance: float,
    restart_index: int,
    restart_count: int,
) -> None:
    if progress_callback is None:
        return

    payload: Dict[str, object] = {
        "backend": backend,
        "phase": phase,
        "generation": int(generation),
        "total_generations": int(total_generations),
        "restart_index": int(restart_index),
        "restart_count": int(restart_count),
        "best_route": list(best_route),
        "best_distance": float(best_distance),
    }
    try:
        progress_callback(payload)
    except RuntimeError:
        raise
    except Exception:
        # Progress notifications must not interrupt optimization.
        return


def _deterministic_inversion_mutation(route: Sequence[int]) -> List[int]:
    mutated = list(route)
    if len(mutated) < 2:
        return mutated

    left, right = sorted(random.sample(range(len(mutated)), 2))
    mutated[left : right + 1] = reversed(mutated[left : right + 1])
    return mutated


def _native_simpleai_supports_problem_ops() -> bool:
    try:
        source = inspect.getsource(simpleai_local._create_genetic_expander)
    except (OSError, TypeError, AttributeError):
        return False

    return "problem.crossover" in source and "problem.mutate" in source


class _SimpleAITSPProblem(SearchProblem):
    def __init__(
        self,
        dist_matrix: np.ndarray,
        crossover_rate: float,
        fitness_power: float,
        epsilon: float,
    ) -> None:
        super().__init__(initial_state=None)
        self.dist_matrix = dist_matrix
        self.city_count = int(dist_matrix.shape[0])
        self.crossover_rate = _clamp_probability(crossover_rate)
        self.fitness_power = _bounded_power(fitness_power)
        self.epsilon = max(float(epsilon), 1e-12)

    def generate_random_state(self) -> Tuple[int, ...]:
        route = list(range(self.city_count))
        random.shuffle(route)
        return tuple(route)

    def value(self, state: Sequence[int]) -> float:
        # Keep positive, stable fitness for weighted sampling.
        distance = route_distance(state, self.dist_matrix)
        base_fitness = 1.0 / (distance + self.epsilon)
        if abs(self.fitness_power - 1.0) < 1e-12:
            return base_fitness
        return base_fitness ** self.fitness_power

    def crossover(self, state1: Sequence[int], state2: Sequence[int]) -> Tuple[int, ...]:
        child = crossover_OX1(state1, state2, crossover_rate=self.crossover_rate)
        _validate_route(child, self.city_count)
        return tuple(child)

    def mutate(self, state: Sequence[int]) -> Tuple[int, ...]:
        # simpleAI already applies mutation probability externally.
        mutated = _deterministic_inversion_mutation(state)
        _validate_route(mutated, self.city_count)
        return tuple(mutated)


class _HistoryViewer(BaseViewer):
    def __init__(
        self,
        dist_matrix: np.ndarray,
        progress_callback: ProgressCallback = None,
        total_generations: int = 1,
        restart_index: int = 1,
        restart_count: int = 1,
    ) -> None:
        super().__init__()
        self.dist_matrix = dist_matrix
        self.progress_callback = progress_callback
        self.total_generations = max(1, int(total_generations))
        self.restart_index = max(1, int(restart_index))
        self.restart_count = max(1, int(restart_count))
        self.initial_best_distance: Optional[float] = None
        self.best_distance = float("inf")
        self.best_route: Optional[List[int]] = None
        self.best_distance_history: List[float] = []
        self.best_route_history: List[List[int]] = []

    def _best_node_by_distance(self, nodes) -> Optional[SearchNodeValueOrdered]:
        if not nodes:
            return None
        return min(nodes, key=lambda node: route_distance(node.state, self.dist_matrix))

    def handle_new_iteration(self, fringe):
        super().handle_new_iteration(fringe)
        best_node = self._best_node_by_distance(fringe)
        if best_node is None:
            return

        current_best_route = list(best_node.state)
        current_best_distance = route_distance(current_best_route, self.dist_matrix)

        if self.initial_best_distance is None:
            self.initial_best_distance = current_best_distance

        if current_best_distance < self.best_distance:
            self.best_distance = current_best_distance
            self.best_route = current_best_route

        if self.best_route is None:
            self.best_route = current_best_route
            self.best_distance = current_best_distance

        self.best_distance_history.append(self.best_distance)
        self.best_route_history.append(list(self.best_route))
        _emit_progress(
            progress_callback=self.progress_callback,
            backend="simpleai",
            phase="ga",
            generation=len(self.best_distance_history),
            total_generations=self.total_generations,
            best_route=self.best_route,
            best_distance=self.best_distance,
            restart_index=self.restart_index,
            restart_count=self.restart_count,
        )

    def handle_finished(self, fringe, node, solution_type):
        super().handle_finished(fringe, node, solution_type)

        candidates = []
        if fringe:
            candidates.extend(fringe)
        if node is not None:
            candidates.append(node)

        best_node = self._best_node_by_distance(candidates)
        if best_node is None:
            return

        final_route = list(best_node.state)
        final_distance = route_distance(final_route, self.dist_matrix)

        if final_distance < self.best_distance:
            self.best_distance = final_distance
            self.best_route = final_route

        if self.best_route is None:
            self.best_route = final_route
            self.best_distance = final_distance

        if self.best_distance_history:
            self.best_distance_history[-1] = self.best_distance
            self.best_route_history[-1] = list(self.best_route)
        else:
            self.best_distance_history.append(self.best_distance)
            self.best_route_history.append(list(self.best_route))
            _emit_progress(
                progress_callback=self.progress_callback,
                backend="simpleai",
                phase="ga",
                generation=1,
                total_generations=self.total_generations,
                best_route=self.best_route,
                best_distance=self.best_distance,
                restart_index=self.restart_index,
                restart_count=self.restart_count,
            )


def _two_opt_refine(
    route: Sequence[int],
    dist_matrix: np.ndarray,
    max_passes: int,
) -> Tuple[List[int], float, List[List[int]], List[float]]:
    """Apply 2-opt local search and return refinement history (best-so-far)."""
    refined = list(route)
    city_count = len(refined)

    if city_count < 4:
        best_distance = route_distance(refined, dist_matrix)
        return refined, best_distance, [], []

    best_distance = route_distance(refined, dist_matrix)
    refinement_routes: List[List[int]] = []
    refinement_distances: List[float] = []

    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        for left in range(1, city_count - 2):
            for right in range(left + 2, city_count + 1):
                candidate = refined.copy()
                candidate[left:right] = reversed(candidate[left:right])
                candidate_distance = route_distance(candidate, dist_matrix)

                if candidate_distance + 1e-12 < best_distance:
                    refined = candidate
                    best_distance = candidate_distance
                    refinement_routes.append(refined.copy())
                    refinement_distances.append(best_distance)
                    improved = True
                    break
            if improved:
                break
        passes += 1

    return refined, best_distance, refinement_routes, refinement_distances


def _run_manual_genetic_loop(
    problem: _SimpleAITSPProblem,
    pop_size: int,
    generations: int,
    mutation_rate: float,
    elite_size: int,
    diversity_rate: float,
    viewer: _HistoryViewer,
):
    """Manual GA wrapper using simpleAI building blocks, with optional elitism/diversity."""
    if viewer is not None:
        viewer.event("started")

    population = [problem.generate_random_state() for _ in range(pop_size)]

    best_node: Optional[SearchNodeValueOrdered] = None
    for _ in range(generations):
        nodes = [SearchNodeValueOrdered(state=state, problem=problem) for state in population]

        if viewer is not None:
            viewer.event("new_iteration", nodes)

        if nodes:
            generation_best = max(nodes, key=lambda node: node.value)
            if best_node is None or generation_best.value > best_node.value:
                best_node = generation_best

        sorted_nodes = sorted(nodes, key=lambda node: node.value, reverse=True)
        elite_count = max(0, min(elite_size, pop_size))
        next_population: List[Tuple[int, ...]] = [tuple(node.state) for node in sorted_nodes[:elite_count]]

        weights = [max(node.value, 0.0) for node in nodes]
        if sum(weights) <= 0:
            weights = [1.0] * len(nodes)

        sampler = InverseTransformSampler(weights, nodes)

        diversity_count = min(pop_size - len(next_population), int(round(pop_size * diversity_rate)))
        target_children = pop_size - diversity_count

        while len(next_population) < target_children:
            parent_a = sampler.sample().state
            parent_b = sampler.sample().state
            child = problem.crossover(parent_a, parent_b)
            if random.random() < mutation_rate:
                child = problem.mutate(child)
            _validate_route(child, problem.city_count)
            next_population.append(tuple(child))

        while len(next_population) < pop_size:
            next_population.append(problem.generate_random_state())

        population = next_population

    final_nodes = [SearchNodeValueOrdered(state=state, problem=problem) for state in population]
    final_best = max(final_nodes, key=lambda node: node.value) if final_nodes else None

    if best_node is None or (final_best is not None and final_best.value > best_node.value):
        best_node = final_best

    if viewer is not None:
        viewer.event("finished", final_nodes, best_node, "returned after reaching iteration limit")

    return best_node


def _run_single_simpleai(
    dist_matrix: np.ndarray,
    pop_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    fitness_power: float,
    epsilon: float,
    enable_2opt: bool,
    two_opt_max_passes: int,
    enable_elitism: bool,
    elite_size: int,
    diversity_rate: float,
    use_native_genetic: bool,
    progress_callback: ProgressCallback = None,
    restart_index: int = 1,
    restart_count: int = 1,
) -> Tuple[List[int], float, List[float], List[List[int]], float]:
    problem = _SimpleAITSPProblem(
        dist_matrix=dist_matrix,
        crossover_rate=crossover_rate,
        fitness_power=fitness_power,
        epsilon=epsilon,
    )
    viewer = _HistoryViewer(
        dist_matrix=dist_matrix,
        progress_callback=progress_callback,
        total_generations=generations,
        restart_index=restart_index,
        restart_count=restart_count,
    )

    native_supported = _native_simpleai_supports_problem_ops()
    use_native = use_native_genetic and native_supported and not enable_elitism

    if use_native:
        best_node = genetic(
            problem=problem,
            population_size=pop_size,
            mutation_chance=mutation_rate,
            iterations_limit=generations,
            viewer=viewer,
        )
    else:
        best_node = _run_manual_genetic_loop(
            problem=problem,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            elite_size=elite_size if enable_elitism else 0,
            diversity_rate=diversity_rate,
            viewer=viewer,
        )

    if best_node is None:
        raise RuntimeError("simpleAI genetic did not return a solution node.")

    if viewer.best_route is not None:
        best_route = list(viewer.best_route)
    else:
        best_route = list(best_node.state)

    _validate_route(best_route, int(dist_matrix.shape[0]))
    best_distance = route_distance(best_route, dist_matrix)

    initial_best_distance = (
        float(viewer.initial_best_distance) if viewer.initial_best_distance is not None else best_distance
    )
    best_distance_history = list(viewer.best_distance_history)
    best_route_history = [list(route) for route in viewer.best_route_history]

    if enable_2opt:
        refined_route, refined_distance, refine_routes, refine_distances = _two_opt_refine(
            route=best_route,
            dist_matrix=dist_matrix,
            max_passes=two_opt_max_passes,
        )

        if refine_distances:
            best_route = refined_route
            best_distance = refined_distance
            best_route_history.extend(refine_routes)
            best_distance_history.extend(refine_distances)

    if not best_distance_history:
        best_distance_history = [best_distance]
        best_route_history = [list(best_route)]
    elif best_distance < best_distance_history[-1]:
        best_distance_history[-1] = best_distance
        best_route_history[-1] = list(best_route)

    if len(best_route_history) != len(best_distance_history):
        shared_len = min(len(best_route_history), len(best_distance_history))
        best_route_history = best_route_history[:shared_len]
        best_distance_history = best_distance_history[:shared_len]

    return best_route, best_distance, best_distance_history, best_route_history, initial_best_distance


def genetic_algorithm_simpleai(
    cities: np.ndarray,
    dist_matrix: np.ndarray,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    crossover_rate: float = CROSSOVER_RATE,
    elite_size: int = ELITE_SIZE,
    tournament_size: int = TOURNAMENT_SIZE,
    progress_callback: ProgressCallback = None,
) -> Tuple[List[int], float, List[float], List[List[int]], float]:
    """Run a robust simpleAI-based TSP genetic solver with multi-restart support."""
    _ = cities
    _ = tournament_size

    safe_matrix = _sanitize_distance_matrix(dist_matrix)
    set_distance_matrix(safe_matrix)

    pop_size = max(2, int(pop_size))
    generations = max(1, int(generations))
    mutation_rate = _clamp_probability(mutation_rate)
    crossover_rate = _clamp_probability(crossover_rate)
    elite_size = max(0, int(elite_size))

    restarts = max(1, int(SIMPLEAI_RESTARTS))

    best_result: Optional[Tuple[List[int], float, List[float], List[List[int]], float]] = None
    global_initial_best_distance = float("inf")
    global_best_distance = float("inf")
    global_best_route: Optional[List[int]] = None
    global_distance_history: List[float] = []
    global_route_history: List[List[int]] = []

    for restart_idx in range(restarts):
        candidate = _run_single_simpleai(
            dist_matrix=safe_matrix,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            fitness_power=float(SIMPLEAI_FITNESS_POWER),
            epsilon=float(SIMPLEAI_EPSILON),
            enable_2opt=bool(SIMPLEAI_ENABLE_2OPT),
            two_opt_max_passes=max(1, int(SIMPLEAI_2OPT_MAX_PASSES)),
            enable_elitism=bool(SIMPLEAI_ENABLE_ELITISM),
            elite_size=elite_size,
            diversity_rate=_clamp_probability(float(SIMPLEAI_DIVERSITY_RATE)),
            use_native_genetic=bool(SIMPLEAI_USE_NATIVE_GENETIC),
            progress_callback=progress_callback,
            restart_index=restart_idx + 1,
            restart_count=restarts,
        )

        candidate_route, candidate_distance, candidate_dist_history, candidate_route_history, candidate_initial = (
            candidate
        )

        if candidate_initial < global_initial_best_distance:
            global_initial_best_distance = candidate_initial

        if best_result is None or candidate_distance < best_result[1]:
            best_result = candidate

        for route_step, distance_step in zip(candidate_route_history, candidate_dist_history):
            if distance_step < global_best_distance:
                global_best_distance = distance_step
                global_best_route = list(route_step)

            if global_best_route is None:
                global_best_route = list(route_step)
                global_best_distance = distance_step

            global_distance_history.append(global_best_distance)
            global_route_history.append(list(global_best_route))

        if candidate_distance < global_best_distance:
            global_best_distance = candidate_distance
            global_best_route = list(candidate_route)

    if best_result is None:
        raise RuntimeError("simpleAI multi-restart solver did not produce a result.")

    if global_best_route is None:
        global_best_route = list(best_result[0])
        global_best_distance = float(best_result[1])

    if not global_distance_history:
        global_distance_history = [global_best_distance]
        global_route_history = [list(global_best_route)]

    if global_initial_best_distance == float("inf"):
        global_initial_best_distance = float(best_result[4])

    return (
        list(global_best_route),
        float(global_best_distance),
        global_distance_history,
        global_route_history,
        float(global_initial_best_distance),
    )
