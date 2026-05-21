# Refactored TSP Genetic Solver

This folder is a self-contained refactor of the original project. Original files outside `new_refract/` are not required at runtime.

## Run

```bash
python new_refract/main.py
```

Launches the current Studio PyQt5 UI.

Other entrypoints:

```bash
python new_refract/main.py --cli
python new_refract/main.py --legacy-gui
cd new_refract
python main.py
python -m pytest -q
```

## Layout

- `app/` contains the clean implementation.
- `app/algorithms/` contains shared TSP problem helpers, route operators, and the legacy core GA engine.
- `app/solvers/` contains the custom GA, SimpleAI GA, and BAT solver.
- `app/ui/` contains the Studio UI, UI worker objects, and the legacy GUI.
- `app/experiments/` and `app/reporting/` contain experiment runners, exporters, logging, and result summaries.
- `core/`, `ga/`, `utils/`, `experiments/`, and `tsp_ga_app/` are compatibility wrappers for old import paths.
- `outputs/` is the only default location for generated artifacts from this refactor.

## File Guide

### Root Files

- `main.py` is the main entrypoint for the refactored project. With no flags it launches the Studio GUI; `--cli` runs the non-GUI solver demo; `--legacy-gui` opens the older compact GUI.
- `requirements.txt` lists Python dependencies needed by the refactored app, tests, plotting, GUI, SimpleAI solver, and GIF export.
- `README.md` documents how to run the refactored project, how the folder is organized, and what each file is responsible for.
- `run_single.py` runs one sample experiment with generated cities, the legacy/core GA engine, result exporting, and reporting metadata.
- `run_batch.py` runs a sample grid-search batch experiment over mutation rate, crossover type, and selection type.
- `tsp_ga.py` preserves the old top-level CLI launcher behavior. It runs the CLI solver by default and supports `--gui`.
- `tsp_ga_gui.py` preserves the old top-level legacy GUI launcher.
- `gui_runner.py` preserves the old import/launch path for the legacy GUI while delegating implementation to `app.ui.legacy_window`.

### Canonical App Package

- `app/__init__.py` marks `app` as the canonical implementation package for the refactor.
- `app/paths.py` defines project-local paths, especially `PROJECT_ROOT` and `OUTPUT_ROOT`, so generated files stay inside `new_refract/`.
- `app/cli.py` contains the non-GUI solver workflow: seed setup, city generation, distance matrix creation, solver selection, printed summary, animation, final route plot, and convergence plot.

### Configuration

- `app/config/__init__.py` re-exports settings for convenient imports.
- `app/config/settings.py` contains all default constants: GA population/generation settings, mutation/crossover rates, SimpleAI tuning, BAT comparison default, animation settings, random seed, GIF path, and the legacy `DEFAULT_CONFIG`.

### Algorithms

- `app/algorithms/__init__.py` exposes the main algorithm helpers and `GAEngine`.
- `app/algorithms/problem.py` contains shared TSP problem utilities: active distance matrix state, random city generation, Euclidean distance matrix creation, route distance, and inverse-distance fitness.
- `app/algorithms/operators.py` contains route/population operators used across the project: population creation, distance-based tournament selection, OX1 crossover, inversion mutation, population evolution, fitness-based tournament/roulette selection, PMX/order crossover, swap/reverse/scramble mutation, elitism, and 2-opt local search.
- `app/algorithms/core_engine.py` contains the legacy configurable GA engine class used by experiment runners and the legacy GUI. It supports selectable selection/crossover/mutation operators, elitism, adaptive mutation, optional local search, metrics, history, runtime, and callback progress.

### Solvers

- `app/solvers/__init__.py` exposes the three solver entrypoints.
- `app/solvers/custom_ga.py` contains the app's custom GA solver used by the Studio UI and CLI. It tracks best route, best distance, convergence history, route history, initial best distance, optional metadata, and live progress payloads.
- `app/solvers/simpleai_ga.py` contains the SimpleAI-based GA solver. It validates/sanitizes distance matrices, defines the SimpleAI TSP problem, records history through a viewer, supports manual fallback GA behavior, optional elitism/diversity injection, multi-restart execution, and optional 2-opt refinement.
- `app/solvers/bat.py` contains the BAT-inspired TSP metaheuristic used for Studio UI comparison runs. It implements permutation-safe guided movement, local walks, inversion mutation, loudness/pulse updates, progress payloads, and convergence histories.

### UI

- `app/ui/__init__.py` exposes the Studio window and launcher.
- `app/ui/studio_window.py` contains the main PyQt5 Studio UI. It builds the full control panel and plot area, handles datasets, run/batch buttons, convergence views, metrics dialog, route drawing, batch overlays, export after completion, and all visible user-facing Studio behavior.
- `app/ui/studio_workers.py` contains Qt worker objects for background execution. `SolverWorker` runs the selected solver and optional BAT comparison; `BatchWorker` runs grid search batches without blocking the UI.
- `app/ui/city_io.py` contains reusable JSON city parsing helpers for point lists, point maps, named datasets, `cities`, `points`, and coordinate dictionaries.
- `app/ui/legacy_window.py` contains the older compact PyQt5 GUI. It preserves the previous simple controls, progress animation, route/convergence plots, JSON loading, and legacy run/reset behavior.

### Experiments

- `app/experiments/__init__.py` exposes experiment runner functions.
- `app/experiments/runner.py` runs a single experiment through `GAEngine`, exports config/metrics/best solution/summary/population snapshots, updates raw result and summary CSV/JSON files, and optionally saves plots/GIFs.
- `app/experiments/batch_runner.py` runs grid-search combinations over a parameter grid and trial count, creates a parent batch output folder, moves per-trial outputs under it, appends batch raw results, and saves a batch best-distance bar chart when possible.

### Reporting

- `app/reporting/__init__.py` exposes exporter and logging helpers.
- `app/reporting/exporter.py` creates experiment folders under `new_refract/outputs` and writes config JSON, metrics CSV, best solution JSON, summary JSON, population snapshots, figures, and batch raw result rows.
- `app/reporting/logger.py` configures the `GA_TSP` logger for console output and optional per-experiment log files.
- `app/reporting/results.py` owns result schemas and summary reporting. It converts NumPy values to JSON-safe data, reads/writes CSV/JSON, records dataset metadata, appends raw results, computes nearest-neighbor baselines, builds optimality fields, and updates summary statistics.

### Visualization

- `app/visualization/__init__.py` exposes plotting helpers.
- `app/visualization/plots.py` creates Matplotlib route plots, convergence plots, and route evolution animations. It can also save the evolution GIF to the configured project-local path.

### Tools

- `app/tools/__init__.py` marks the tools package.
- `app/tools/gen_data.py` converts TSPLIB-style CSV rows into JSON city datasets. It includes CSV field-size handling, filename slugging, coordinate parsing, and per-instance JSON export.
- `app/tools/reproduce_report_figures.py` reads generated reporting CSV files and saves report figures for best distance by run, summary mean with confidence intervals, and dataset city counts.

### Compatibility Wrappers

These files exist so old imports still work inside `new_refract/`. They should stay thin and delegate to `app.*`.

- `core/__init__.py`, `core/config.py`, and `core/ga_engine.py` preserve old `core.*` imports for `DEFAULT_CONFIG` and `GAEngine`.
- `ga/__init__.py`, `ga/selection.py`, `ga/crossover.py`, `ga/mutation.py`, `ga/elitism.py`, and `ga/local_search.py` preserve old GA operator imports.
- `utils/__init__.py`, `utils/exporter.py`, `utils/logger.py`, and `utils/results_reporting.py` preserve old reporting/logging imports.
- `experiments/__init__.py`, `experiments/runner.py`, and `experiments/batch_runner.py` preserve old experiment runner imports.
- `tsp_ga_app/__init__.py`, `tsp_ga_app/config.py`, `tsp_ga_app/problem.py`, `tsp_ga_app/operators.py`, `tsp_ga_app/solver.py`, `tsp_ga_app/simpleai_solver.py`, `tsp_ga_app/bat_solver.py`, `tsp_ga_app/visualization.py`, `tsp_ga_app/visualize.py`, `tsp_ga_app/gui.py`, and `tsp_ga_app/main.py` preserve the old app package API while redirecting to the refactored implementation.
- `data_gen/__init__.py` and `data_gen/gen_data.py` preserve the old data-conversion import and script path.
- `scripts/__init__.py` and `scripts/reproduce_report_figures.py` preserve the old report-figure script path.

### Tests

- `tests/conftest.py` ensures `new_refract/` is on `sys.path` during tests.
- `tests/test_selection.py` checks both legacy fitness-based selection and app distance-based tournament selection return valid routes.
- `tests/test_crossover.py` checks PMX, order crossover, and OX1 all produce valid permutations.
- `tests/test_mutation.py` checks swap, reverse, scramble, and inversion mutation preserve route validity.
- `tests/test_elitism.py` checks core elitism keeps the best fitness route and app evolution preserves the lowest-distance elite.
- `tests/test_distance.py` checks Euclidean distance matrix creation and cyclic route distance.
- `tests/test_route_validity.py` checks generated populations contain every city exactly once.
- `tests/test_reproducibility.py` checks the same seed gives the same best route and distance through `GAEngine`.
- `tests/test_result_schema.py` checks raw result exports include required fields.
- `tests/test_dataset_metadata.py` checks dataset metadata exports include required fields.
- `tests/test_optimality_gap.py` checks known optimum gap calculation and unknown optimum labeling.
- `tests/test_output_isolation.py` checks the default exporter output root is inside `new_refract/`.

### Outputs

- `outputs/.gitkeep` keeps the output folder present in version control.
- `outputs/` is where the CLI, GUI, experiment runners, reports, summaries, logs, figures, population snapshots, and GIFs are written by default.

## Compatibility

Old imports continue to work when run from inside `new_refract/`, for example:

```python
from ga.selection import tournament_selection
from core.ga_engine import GAEngine
from tsp_ga_app.problem import compute_distance_matrix
from utils.results_reporting import append_raw_result
```

The wrappers re-export functions/classes from `app/`; new code should prefer `app.*` imports.
