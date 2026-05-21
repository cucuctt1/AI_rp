from typing import Optional

from app.paths import OUTPUT_ROOT


# Studio/custom solver defaults.
POP_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
ELITE_SIZE = 2
NUM_CITIES = 20
TOURNAMENT_SIZE = 3

# Visualization configuration.
ANIMATION_INTERVAL_MS = 80

# Solver backend: "custom" or "simpleai".
SOLVER_BACKEND = "custom"
ENABLE_BAT_COMPARISON = False

# simpleAI quality tuning.
SIMPLEAI_RESTARTS = 8
SIMPLEAI_ENABLE_2OPT = True
SIMPLEAI_2OPT_MAX_PASSES = 25
SIMPLEAI_FITNESS_POWER = 2.0
SIMPLEAI_USE_NATIVE_GENETIC = True
SIMPLEAI_ENABLE_ELITISM = False
SIMPLEAI_DIVERSITY_RATE = 0.05
SIMPLEAI_EPSILON = 1e-3

# Bonus options.
RANDOM_SEED: Optional[int] = 34230
SAVE_GIF = True
GIF_PATH = str(OUTPUT_ROOT / "tsp_ga_evolution.gif")

# Legacy/core GA defaults.
DEFAULT_CONFIG = {
    "population_size": 100,
    "mutation_rate": 0.05,
    "crossover_type": "pmx",
    "mutation_type": "swap",
    "selection_type": "tournament",
    "generations": 500,
    "elitism_k": 2,
    "adaptive_mutation": False,
    "local_search_freq": 0,
}
