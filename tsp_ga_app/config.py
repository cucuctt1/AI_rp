from typing import Optional

# Default GA parameters requested in the task.
POP_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
ELITE_SIZE = 2
NUM_CITIES = 20
TOURNAMENT_SIZE = 3

# Visualization configuration.
ANIMATION_INTERVAL_MS = 80

# Solver backend: "custom" (from-scratch GA) or "simpleai".
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
GIF_PATH = "tsp_ga_evolution.gif"
