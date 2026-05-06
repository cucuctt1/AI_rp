import random

def swap_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def reverse_mutation(route, mutation_rate):
    """Inversion mutation (often better for TSP). Reverses a subsection."""
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        route[idx1:idx2+1] = reversed(route[idx1:idx2+1])
    return route

def scramble_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        subset = route[idx1:idx2+1]
        random.shuffle(subset)
        route[idx1:idx2+1] = subset
    return route
