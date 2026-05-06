import numpy as np

def tournament_selection(population, fitnesses, k=3):
    indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmax(fitnesses[indices])]
    return population[best_idx]

def roulette_selection(population, fitnesses):
    total_fit = np.sum(fitnesses)
    if total_fit == 0:
        return population[np.random.randint(len(population))]
    probs = fitnesses / total_fit
    chosen_idx = np.random.choice(len(population), p=probs)
    return population[chosen_idx]
