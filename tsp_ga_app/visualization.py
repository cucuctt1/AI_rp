from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .config import ANIMATION_INTERVAL_MS, GIF_PATH, SAVE_GIF


def plot_route(
    cities: np.ndarray,
    route: Sequence[int],
    distance: Optional[float] = None,
    title: str = "Final TSP Route",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a route with city markers and city index labels."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(cities[:, 0], cities[:, 1], c="tab:red", s=50, zorder=3)
    for city_idx, (x_coord, y_coord) in enumerate(cities):
        ax.text(x_coord + 0.8, y_coord + 0.8, str(city_idx), fontsize=8)

    closed_route = list(route) + [route[0]]
    ordered = cities[closed_route]
    ax.plot(
        ordered[:, 0],
        ordered[:, 1],
        color="tab:blue",
        linewidth=2,
        marker="o",
        markersize=5,
        zorder=2,
    )

    if distance is not None:
        ax.set_title(f"{title} | Distance: {distance:.3f}")
    else:
        ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, ax


def plot_convergence(best_distance_history: Sequence[float]) -> Tuple[plt.Figure, plt.Axes]:
    """Plot convergence curve: generation vs best distance."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    generations = np.arange(1, len(best_distance_history) + 1)
    ax.plot(generations, best_distance_history, color="tab:green", linewidth=2)
    ax.set_title("GA Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Distance")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, ax


def animate_evolution(
    cities: np.ndarray,
    route_history: Sequence[Sequence[int]],
    distance_history: Sequence[float],
    interval: int = ANIMATION_INTERVAL_MS,
    repeat: bool = False,
    save_gif: bool = SAVE_GIF,
    gif_path: str = GIF_PATH,
) -> Tuple[plt.Figure, FuncAnimation]:
    """Animate route improvement across generations."""
    if not route_history:
        raise ValueError("Route history must not be empty for animation.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cities[:, 0], cities[:, 1], c="tab:red", s=50, zorder=3)
    for city_idx, (x_coord, y_coord) in enumerate(cities):
        ax.text(x_coord + 0.8, y_coord + 0.8, str(city_idx), fontsize=8)

    route_line, = ax.plot([], [], color="tab:orange", linewidth=2, marker="o", markersize=5)
    title_text = ax.set_title("")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    def update(frame_idx: int):
        route = route_history[frame_idx]
        closed_route = list(route) + [route[0]]
        ordered = cities[closed_route]
        route_line.set_data(ordered[:, 0], ordered[:, 1])
        title_text.set_text(
            f"Generation {frame_idx + 1}/{len(route_history)} | Best Distance: {distance_history[frame_idx]:.3f}"
        )
        return route_line, title_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(route_history),
        interval=interval,
        blit=False,
        repeat=repeat,
    )

    if save_gif:
        try:
            fps = max(1, int(1000 / max(interval, 1)))
            animation.save(gif_path, writer="pillow", fps=fps)
            print(f"Animation GIF saved to: {gif_path}")
        except Exception as err:  # pylint: disable=broad-except
            print(f"Animation GIF save skipped: {err}")

    fig.tight_layout()
    return fig, animation
