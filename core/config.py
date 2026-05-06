DEFAULT_CONFIG = {
    "population_size": 100,
    "mutation_rate": 0.05,
    "crossover_type": "pmx",     # other options: "order", "cycle"
    "mutation_type": "swap",     # other options: "reverse", "scramble"
    "selection_type": "tournament", # other options: "roulette"
    "generations": 500,
    "elitism_k": 2,
    "adaptive_mutation": False,
    "local_search_freq": 0       # 0 means disabled
}
