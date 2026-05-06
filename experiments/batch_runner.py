import copy
import itertools
import os
import shutil
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from experiments.runner import run_single_experiment
from utils.exporter import Exporter
from utils.logger import setup_logger

def run_grid_search(
    base_config: dict,
    param_grid: Dict[str, List[Any]],
    dist_matrix: np.ndarray,
    cities: np.ndarray = None,
    base_experiment_name: str = "batch",
    num_trials: int = 1,
    seed_offset: int = 42,
    per_trial_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    param_grid: dictionary where keys are config keys, and values are lists of options to try.
    Example: {'mutation_rate': [0.01, 0.05, 0.1], 'crossover_type': ['pmx', 'order']}
    """
    exporter = Exporter()
    logger = setup_logger() # Root logger for batch runner

    # Create a batch parent folder to contain all trial subfolders and the batch-level raw_runs.csv
    batch_parent = exporter.create_experiment_folder(base_experiment_name)
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    total_runs = len(combinations) * num_trials
    logger.info(f"Starting Batch Experiment: {base_experiment_name}")
    logger.info(f"Total configurations: {len(combinations)}, Total runs: {total_runs}")
    
    run_counter = 0
    
    for combo in combinations:
        # Create a specific config for this combination
        current_config = copy.deepcopy(base_config)
        for k, v in zip(keys, combo):
            current_config[k] = v
            
        for trial in range(num_trials):
            run_counter += 1
            experiment_id = f"{base_experiment_name}_cfg{''.join(str(c) for c in combo)}_trial{trial}"
            current_seed = seed_offset + run_counter
            
            logger.info(f"--- Running {run_counter}/{total_runs} : {experiment_id} ---")
            
            # Use the single runner
            results, folder_path = run_single_experiment(experiment_id, current_config, dist_matrix, cities=cities, seed=current_seed)

            if per_trial_callback is not None:
                try:
                    per_trial_callback(
                        {
                            "experiment_id": experiment_id,
                            "best_distance": results.get("best_distance"),
                            "best_route": results.get("best_route"),
                            "folder_path": folder_path,
                        }
                    )
                except Exception:
                    pass

            # Move the created per-trial folder into the batch parent folder for containment
            try:
                dest_subfolder = os.path.join(batch_parent, os.path.basename(folder_path))
                # If destination exists, overwrite by removing
                if os.path.exists(dest_subfolder):
                    shutil.rmtree(dest_subfolder)
                shutil.move(folder_path, dest_subfolder)
                folder_path = dest_subfolder
            except Exception:
                # Non-fatal; continue and still append results to batch CSV
                pass

            # Reattach the logger correctly since run_single_experiment may set up its own logger
            logger = setup_logger()

            # Aggregate stats to raw_runs.csv (written inside batch parent)
            batch_result_row = {
                "experiment_id": experiment_id,
                "crossover_type": current_config.get('crossover_type', ''),
                "mutation_type": current_config.get('mutation_type', ''),
                "mutation_rate": current_config.get('mutation_rate', ''),
                "selection_type": current_config.get('selection_type', ''),
                "population_size": current_config.get('population_size', ''),
                "best_distance": results['best_distance'],
                "convergence_gen": results['convergence_gen'],
                "runtime": round(results['runtime'], 4)
            }
            exporter.append_batch_results(batch_result_row, dest_folder=batch_parent)
            
    logger.info("Batch Experiment Complete!")
    # Produce a simple batch-level summary plot (best distance per experiment)
    try:
        import csv
        import matplotlib.pyplot as plt

        csv_path = os.path.join(batch_parent, "raw_runs.csv")
        if os.path.isfile(csv_path):
            exp_ids = []
            bests = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    exp_ids.append(row.get('experiment_id', ''))
                    try:
                        bests.append(float(row.get('best_distance', float('nan'))))
                    except Exception:
                        bests.append(float('nan'))

            if exp_ids and bests:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(bests)), bests, color='tab:blue')
                ax.set_xticks(range(len(bests)))
                ax.set_xticklabels(exp_ids, rotation=90, fontsize=6)
                ax.set_ylabel('Best Distance')
                ax.set_title('Batch: Best Distance per Run')
                fig.tight_layout()
                exporter.save_figure(batch_parent, fig, 'batch_best_distances.png')
                plt.close(fig)
    except Exception:
        pass
