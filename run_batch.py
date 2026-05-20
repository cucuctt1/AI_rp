import numpy as np
from tsp_ga_app.problem import compute_distance_matrix

if __name__ == "__main__":
    from core.config import DEFAULT_CONFIG
    from experiments.batch_runner import run_grid_search
    
    np.random.seed(42)
    # Using a 30-city problem for quick batch tests
    cities = np.random.uniform(0.0, 100.0, size=(30, 2))
    dist_matrix = compute_distance_matrix(cities)
    
    base_config = DEFAULT_CONFIG.copy()
    base_config['generations'] = 150
    base_config['population_size'] = 50
    
    param_grid = {
        'mutation_rate': [0.01, 0.05, 0.1],
        'crossover_type': ['pmx', 'order'],
        'selection_type': ['tournament', 'roulette']
    }
    
    # This will run 3 * 2 * 2 = 12 configurations, each num_trials times
    run_grid_search(
        base_config,
        param_grid,
        dist_matrix,
        cities=cities,
        base_experiment_name="grid_search",
        num_trials=2,
        dataset_name="generated_random_30_seed_42",
        coordinate_source_or_seed="numpy.random.seed(42); uniform(0,100)",
        known_optimum="N/A",
    )
