import numpy as np

# A small utility to generate cities internally so we don't rely on legacy code
def compute_distance_matrix(cities: np.ndarray) -> np.ndarray:
    deltas = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))

if __name__ == "__main__":
    from core.config import DEFAULT_CONFIG
    from experiments.runner import run_single_experiment
    
    np.random.seed(42)
    cities = np.random.uniform(0.0, 100.0, size=(50, 2))
    dist_matrix = compute_distance_matrix(cities)
    
    # Overwrite default config as needed
    config = DEFAULT_CONFIG.copy()
    config['generations'] = 200
    config['population_size'] = 100
    config['local_search_freq'] = 10  # Apply 2-opt every 10 generations
    
    run_single_experiment("single_test_ox_swap", config, dist_matrix, cities=cities, seed=99)
