import numpy as np
import os
import random
from typing import Any
from app.algorithms.core_engine import GAEngine
from app.experiments.config_mapping import normalize_core_ga_config
from app.reporting.exporter import Exporter
from app.reporting.logger import setup_logger
from app.reporting.results import (
    append_raw_result,
    build_optimality_fields,
    get_git_commit_hash,
    update_summary_statistics,
    upsert_dataset_metadata,
)

def run_single_experiment(
    experiment_name: str,
    config: dict,
    dist_matrix: np.ndarray,
    cities: np.ndarray = None,
    seed: int = None,
    dataset_name: str = None,
    coordinate_source_or_seed: str = None,
    known_optimum: Any = "N/A",
    algorithm: str = "core_ga",
    run_id: str = None,
    summary_experiment_name: str = None,
):
    config = normalize_core_ga_config(config)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    exporter = Exporter()
    folder_path = exporter.create_experiment_folder(experiment_name)
    
    logger = setup_logger(folder_path)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output folder: {folder_path}")

    n_cities = int(dist_matrix.shape[0])
    dataset_name = dataset_name or f"generated_{n_cities}_cities"
    coordinate_source_or_seed = coordinate_source_or_seed or (
        "provided coordinates" if cities is not None else "distance matrix only"
    )
    git_commit = get_git_commit_hash()

    config_snapshot = dict(config)
    config_snapshot.update(
        {
            "algorithm": algorithm,
            "seed": seed if seed is not None else "N/A",
            "base_seed": config.get("base_seed", seed if seed is not None else "N/A"),
            "dataset_name": dataset_name,
            "coordinate_source_or_seed": coordinate_source_or_seed,
            "distance_metric": "euclidean",
            "known_optimum": known_optimum,
            "git_commit": git_commit,
        }
    )
    
    exporter.save_config(folder_path, config_snapshot)
    
    engine = GAEngine(config, dist_matrix)
    results = engine.run(callback=config.get('gui_callback', None))
    
    # Export all artifacts
    exporter.save_metrics(folder_path, results['metrics'])
    exporter.save_best_solution(
        folder_path, 
        results['best_route'], 
        results['best_distance'], 
        results['convergence_gen']
    )
    exporter.save_summary(
        folder_path, 
        results['best_distance'], 
        config['generations'], 
        results['convergence_gen'], 
        results['runtime']
    )
    exporter.save_population_snapshot(folder_path, results['final_population'], config['generations'])

    optimality_fields = build_optimality_fields(
        best_distance=results["best_distance"],
        known_optimum=known_optimum,
        dist_matrix=dist_matrix,
    )
    upsert_dataset_metadata(
        dataset_name=dataset_name,
        n_cities=n_cities,
        coordinate_source_or_seed=coordinate_source_or_seed,
        distance_metric="euclidean",
        known_optimum=optimality_fields.get("known_optimum", "N/A"),
        known_optimum_note=optimality_fields.get("known_optimum_note", ""),
    )

    raw_experiment_name = summary_experiment_name or experiment_name
    pop_size = config.get("population_size", config.get("pop_size", ""))
    elitism_k = config.get("elitism_k", config.get("elite_size", ""))
    raw_row = {
        "experiment_name": raw_experiment_name,
        "algorithm": algorithm,
        "run_id": run_id or experiment_name,
        "seed": seed if seed is not None else "N/A",
        "dataset_name": dataset_name,
        "n_cities": n_cities,
        "pop_size": pop_size,
        "generations": config.get("generations", ""),
        "crossover_type": config.get("crossover_type", ""),
        "mutation_type": config.get("mutation_type", ""),
        "selection_type": config.get("selection_type", ""),
        "mutation_rate": config.get("mutation_rate", ""),
        "elitism_k": elitism_k,
        "best_distance": results["best_distance"],
        "generation_found": results["convergence_gen"],
        "runtime_seconds": results["runtime"],
        "fitness_evaluations": results.get("fitness_evaluations", "N/A"),
        "base_seed": config.get("base_seed", seed if seed is not None else "N/A"),
        "git_commit": git_commit,
        "coordinate_source_or_seed": coordinate_source_or_seed,
        "distance_metric": "euclidean",
        **optimality_fields,
    }
    append_raw_result(raw_row)
    update_summary_statistics()

    # Generate visualization artifacts if history and city coordinates available
    try:
        from app.visualization.plots import animate_evolution, plot_route, plot_convergence
        # Save GIF if we have route history and cities
        if cities is not None and results.get('best_route_history'):
            gif_path = os.path.join(folder_path, 'evolution.gif')
            try:
                _, animation = animate_evolution(
                    cities=cities,
                    route_history=results['best_route_history'],
                    distance_history=results['best_distance_history'],
                    interval=config.get('animation_interval_ms', 120),
                    repeat=False,
                    save_gif=True,
                    gif_path=gif_path,
                )
            except Exception:
                pass

        # Save final route plot and convergence plot
        try:
            fig_route, _ = plot_route(cities if cities is not None else np.zeros((0, 2)), results['best_route'], results['best_distance'], title="Final Best Route")
            exporter.save_figure(folder_path, fig_route, 'final_route.png')
        except Exception:
            pass

        try:
            fig_conv, _ = plot_convergence(results.get('best_distance_history', []))
            exporter.save_figure(folder_path, fig_conv, 'convergence.png')
        except Exception:
            pass
    except Exception:
        # visualization module may not be available in minimal environments
        pass
    
    logger.info(f"Experiment completed. Best Distance: {results['best_distance']:.2f}")
    logger.info(f"Convergence Generation: {results['convergence_gen']}")
    logger.info(f"Runtime: {results['runtime']:.2f} seconds")
    
    return results, folder_path
