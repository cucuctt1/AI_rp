import numpy as np
import os
from core.ga_engine import GAEngine
from core.config import DEFAULT_CONFIG
from utils.exporter import Exporter
from utils.logger import setup_logger

def run_single_experiment(experiment_name: str, config: dict, dist_matrix: np.ndarray, cities: np.ndarray = None, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        
    exporter = Exporter()
    folder_path = exporter.create_experiment_folder(experiment_name)
    
    logger = setup_logger(folder_path)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output folder: {folder_path}")
    
    exporter.save_config(folder_path, config)
    
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

    # Generate visualization artifacts if history and city coordinates available
    try:
        from tsp_ga_app.visualize import animate_evolution, plot_route, plot_convergence
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
