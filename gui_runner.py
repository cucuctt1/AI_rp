from app.ui.legacy_window import (
    SolverWorker,
    TSPControlPanel,
    compute_distance_matrix,
    generate_cities,
    main,
)

__all__ = [
    "compute_distance_matrix",
    "generate_cities",
    "SolverWorker",
    "TSPControlPanel",
    "main",
]


if __name__ == "__main__":
    main()
