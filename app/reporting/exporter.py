import os
import json
import csv
from datetime import datetime

from app.paths import OUTPUT_ROOT
from app.reporting.results import to_json_ready

class Exporter:
    def __init__(self, output_root=None):
        self.output_root = str(output_root) if output_root is not None else str(OUTPUT_ROOT)
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
            
    def create_experiment_folder(self, experiment_name: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{experiment_name}"
        folder_path = os.path.join(self.output_root, folder_name)
        os.makedirs(folder_path)
        return folder_path

    def save_config(self, folder_path: str, config: dict):
        file_path = os.path.join(folder_path, "config.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(to_json_ready(config), f, indent=4)

    def save_metrics(self, folder_path: str, metrics: list):
        """
        metrics: list of dicts with keys: 
        ['generation', 'best_fitness', 'avg_fitness', 'worst_fitness']
        """
        file_path = os.path.join(folder_path, "metrics.csv")
        metrics_keys = ['generation', 'best_fitness', 'avg_fitness', 'worst_fitness']
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_keys)
            writer.writeheader()
            writer.writerows(metrics)

    def save_best_solution(self, folder_path: str, route: list, distance: float, generation_found: int):

        file_path = os.path.join(folder_path, "best_solution.json")

        data = {
            "best_route": [int(x) for x in route],  # 👈 FIX
            "total_distance": float(distance),      # 👈 FIX
            "generation_found": int(generation_found)  # 👈 FIX (an toàn)
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save_summary(self, folder_path: str, best_distance: float, total_generations: int, convergence_generation: int, runtime: float):
        file_path = os.path.join(folder_path, "summary.json")
        data = {
            "best_distance": best_distance,
            "total_generations": total_generations,
            "convergence_generation": convergence_generation,
            "runtime": runtime
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
    def save_population_snapshot(self, folder_path: str, population: list, generation: int):
        file_path = os.path.join(folder_path, f"population_snapshot_gen_{generation}.json")

        # convert numpy → python
        clean_population = [[int(city) for city in route] for route in population]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(clean_population, f, indent=4)

    def save_figure(self, folder_path: str, fig, filename: str):
        """Save a Matplotlib Figure object to the experiment folder as PNG."""
        try:
            file_path = os.path.join(folder_path, filename)
            # Ensure folder exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fig.savefig(file_path)
        except Exception:
            # Swallow exceptions to avoid breaking experiment flow
            pass
    def append_batch_results(self, result: dict, dest_folder: str = None):
        """
        result: dict containing single run details. 
        Will act as grid search aggregate log.
        """
        target_root = dest_folder if dest_folder is not None else self.output_root
        file_path = os.path.join(target_root, "raw_runs.csv")
        file_exists = os.path.isfile(file_path)
        
        fieldnames = [
            "experiment_id", "crossover_type", "mutation_type", 
            "mutation_rate", "selection_type", "population_size", 
            "best_distance", "convergence_gen", "runtime",
            "experiment_name", "algorithm", "run_id", "seed", "dataset_name",
            "n_cities", "pop_size", "generations", "elitism_k",
            "generation_found", "runtime_seconds", "fitness_evaluations",
            "base_seed", "git_commit", "coordinate_source_or_seed",
            "distance_metric", "known_optimum", "known_optimum_note",
            "optimality_gap", "optimality_gap_reason",
            "nearest_neighbor_distance", "baseline_relative_improvement_percent"
        ]
        
        # Ensure destination folder exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            # Filter result keys to ensure they match fieldnames
            row = {key: result.get(key, None) for key in fieldnames}
            writer.writerow(row)
