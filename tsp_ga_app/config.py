from typing import Optional

# Default GA parameters requested in the task.
POP_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITE_SIZE = 2
NUM_CITIES = 20
TOURNAMENT_SIZE = 3

# Visualization configuration.
ANIMATION_INTERVAL_MS = 80

# Bonus options.
RANDOM_SEED: Optional[int] = 34230
SAVE_GIF = True
GIF_PATH = "tsp_ga_evolution.gif"
