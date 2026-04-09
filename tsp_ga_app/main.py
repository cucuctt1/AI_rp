import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .config import (
    ANIMATION_INTERVAL_MS,
    CROSSOVER_RATE,
    ELITE_SIZE,
    GENERATIONS,
    GIF_PATH,
    MUTATION_RATE,
    NUM_CITIES,
    POP_SIZE,
    RANDOM_SEED,
    SAVE_GIF,
    TOURNAMENT_SIZE,
)
from .problem import compute_distance_matrix, generate_cities
from .solver import genetic_algorithm
from .visualization import animate_evolution, plot_convergence, plot_route


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    cities = generate_cities(NUM_CITIES)
    dist_matrix = compute_distance_matrix(cities)

    (
        best_route,
        best_distance,
        best_distance_history,
        best_route_history,
        initial_best_distance,
    ) = genetic_algorithm(
        cities=cities,
        dist_matrix=dist_matrix,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elite_size=ELITE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
    )

    improvement = initial_best_distance - best_distance
    improvement_pct = (improvement / initial_best_distance * 100.0) if initial_best_distance > 0 else 0.0

    print("Best route:", best_route)
    print(f"Best distance: {best_distance:.4f}")
    print(f"Initial best distance: {initial_best_distance:.4f}")
    print(f"Distance improvement: {improvement:.4f} ({improvement_pct:.2f}%)")

    backend_name = plt.get_backend().lower()
    can_show_animation = "agg" not in backend_name or SAVE_GIF
    animation: Optional[FuncAnimation] = None

    # Keep a reference to animation to prevent garbage collection before show().
    if can_show_animation:
        _, animation = animate_evolution(
            cities=cities,
            route_history=best_route_history,
            distance_history=best_distance_history,
            interval=ANIMATION_INTERVAL_MS,
            repeat=False,
            save_gif=SAVE_GIF,
            gif_path=GIF_PATH,
        )
    else:
        print("Animation display skipped on non-interactive backend.")

    plot_route(cities, best_route, best_distance, title="Final Best Route")
    plot_convergence(best_distance_history)

    _ = animation
    if "agg" in backend_name:
        plt.close("all")
        return

    plt.show()
