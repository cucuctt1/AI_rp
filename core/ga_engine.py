import time
import numpy as np

from core.config import DEFAULT_CONFIG
from ga.selection import tournament_selection, roulette_selection
from ga.crossover import pmx_crossover, order_crossover
from ga.mutation import swap_mutation, reverse_mutation, scramble_mutation
from ga.elitism import get_elites
from ga.local_search import two_opt_search
from utils.logger import get_logger

class GAEngine:
    def __init__(self, config, dist_matrix):
        self.config = {**DEFAULT_CONFIG, **dict(config or {})}
        self.dist_matrix = dist_matrix
        self.num_cities = dist_matrix.shape[0]
        self.logger = get_logger()
        
    def _route_distance(self, route):
        total = 0.0
        n = len(route)
        for i in range(n):
            total += self.dist_matrix[route[i], route[(i + 1) % n]]
        return float(total)
        
    def _evaluate_population(self, population):
        distances = np.array([self._route_distance(ind) for ind in population])
        # Fitness is inverse distance (handle division by zero if any)
        fitnesses = np.zeros_like(distances)
        valid = distances > 0
        fitnesses[valid] = 1.0 / distances[valid]
        return distances, fitnesses

    def run(self, callback=None):
        start_time = time.time()
        
        # Setup operators
        selection_fn = roulette_selection if self.config.get('selection_type', DEFAULT_CONFIG['selection_type']) == 'roulette' else tournament_selection
        crossover_fn = order_crossover if self.config.get('crossover_type', DEFAULT_CONFIG['crossover_type']) == 'order' else pmx_crossover
        
        mut_type = self.config.get('mutation_type', DEFAULT_CONFIG['mutation_type'])
        if mut_type == 'reverse':
            mutation_fn = reverse_mutation
        elif mut_type == 'scramble':
            mutation_fn = scramble_mutation
        else:
            mutation_fn = swap_mutation
            
        pop_size = int(self.config.get('population_size', DEFAULT_CONFIG['population_size']))
        generations = int(self.config.get('generations', DEFAULT_CONFIG['generations']))
        elitism_k = self.config.get('elitism_k', 2)
        local_search_freq = self.config.get('local_search_freq', 0)
        
        # Current mutation rate will be adaptable
        current_mutation_rate = float(self.config.get('mutation_rate', DEFAULT_CONFIG['mutation_rate']))

        # Initialize Population
        population = [list(np.random.permutation(self.num_cities)) for _ in range(pop_size)]
        
        metrics = []
        global_best_route = None
        global_best_dist = float('inf')
        convergence_gen = 0
        best_route_history = []
        best_distance_history = []
        
        stagnation_counter = 0

        self.logger.info(f"Starting GA with population {pop_size}, generations {generations}")

        for gen in range(1, generations + 1):
            distances, fitnesses = self._evaluate_population(population)
            
            gen_best_idx = np.argmin(distances)
            gen_best_dist = distances[gen_best_idx]
            gen_best_route = population[gen_best_idx]
            
            if gen_best_dist < global_best_dist:
                global_best_dist = gen_best_dist
                global_best_route = gen_best_route.copy()
                convergence_gen = gen
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # Adaptive Mutation Logic
            if self.config.get('adaptive_mutation', DEFAULT_CONFIG['adaptive_mutation']):
                if stagnation_counter > 20:
                    current_mutation_rate = min(0.5, current_mutation_rate * 1.5)
                elif stagnation_counter == 0:
                    current_mutation_rate = float(self.config.get('mutation_rate', DEFAULT_CONFIG['mutation_rate']))

            metrics.append({
                'generation': gen,
                'best_fitness': 1.0 / gen_best_dist,
                'avg_fitness': np.mean(fitnesses),
                'worst_fitness': np.min(fitnesses)
            })

            # Record best-so-far history for visualization/export
            if global_best_route is None:
                best_route_history.append(list(gen_best_route))
            else:
                best_route_history.append(list(global_best_route))
            best_distance_history.append(float(global_best_dist))

            new_population = []
            
            # Elitism
            elites = get_elites(population, fitnesses, elitism_k)
            new_population.extend(elites)
            
            # Create rest of population
            while len(new_population) < pop_size:
                p1 = selection_fn(population, fitnesses)
                p2 = selection_fn(population, fitnesses)
                
                child = crossover_fn(p1, p2)
                child = mutation_fn(child, current_mutation_rate)
                
                # Optional Local Search (2-opt)
                if local_search_freq > 0 and gen % local_search_freq == 0:
                    child = two_opt_search(child, self.dist_matrix)
                    
                new_population.append(child)
                
            population = new_population
            
            if gen % 50 == 0:
                self.logger.info(f"Gen {gen:4d} | Best Dist: {global_best_dist:.2f} | Adaptive Mut: {current_mutation_rate:.3f}")
                
            if callback:
                callback({
                    'generation': gen,
                    'global_best_distance': global_best_dist,
                    'global_best_route': global_best_route,
                    'metrics': metrics[-1]
                })

        runtime = time.time() - start_time
        
        return {
            'best_route': global_best_route,
            'best_distance': global_best_dist,
            'convergence_gen': convergence_gen,
            'metrics': metrics,
            'best_route_history': best_route_history,
            'best_distance_history': best_distance_history,
            'final_population': population,
            'runtime': runtime
        }
