import numpy as np

def get_elites(population, fitnesses, k):
    """Return the k best individuals."""
    if k <= 0:
        return []
    elite_indices = np.argsort(fitnesses)[-k:]
    return [population[i].copy() for i in elite_indices]
