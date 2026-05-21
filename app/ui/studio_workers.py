from collections import deque
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Deque, Dict, List, Optional, Set

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from app.config import settings as app_config
from app.algorithms.problem import compute_distance_matrix, generate_cities
from app.solvers.bat import bat_algorithm_tsp
from app.solvers.simpleai_ga import genetic_algorithm_simpleai
from app.solvers.custom_ga import genetic_algorithm as genetic_algorithm_custom

STOP_EXCEPTION_TEXT = "Solver stopped by user."
DEFAULT_ANIMATION_INTERVAL_MS = 120


class SolverWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = params
        self._stop_requested = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        simpleai_overrides: Dict[str, Any] = {}
        bat_thread: Optional[threading.Thread] = None
        bat_state: Dict[str, Any] = {"result": None, "error": None, "stopped": False, "runtime": None}
        primary_result: Optional[Dict[str, Any]] = None
        comparison_result: Optional[Dict[str, Any]] = None
        error_text: Optional[str] = None
        stopped = False
        cities: Optional[np.ndarray] = None
        backend = ""
        compare_enabled = False
        primary_runtime: Optional[float] = None
        primary_metadata: Dict[str, Any] = {}
        total_runtime: Optional[float] = None
        start_overall = time.perf_counter()

        try:
            seed = self.params["seed"]
            if seed is None:
                seed = int(time.time_ns() % 1_000_000_000)
                self.params["seed"] = seed
                self.params["seed_is_generated"] = True
            random.seed(seed)
            np.random.seed(seed)

            cities = self.params.get("cities")
            if cities is None:
                cities = generate_cities(int(self.params["num_cities"]))
            else:
                cities = np.asarray(cities, dtype=float)
            dist_matrix = compute_distance_matrix(cities)

            backend = str(self.params["backend"]).strip().lower()
            compare_enabled = bool(self.params["enable_bat_comparison"])
            self.progress.emit(
                {
                    "event": "init",
                    "backend": backend,
                    "comparison_enabled": compare_enabled,
                    "cities": cities,
                }
            )

            if compare_enabled:
                def run_bat() -> None:
                    try:
                        bat_start = time.perf_counter()
                        bat_seed = int(seed) + 101 if seed is not None else None
                        bat_rng = random.Random(bat_seed) if bat_seed is not None else random.Random()

                        (
                            bat_best_route,
                            bat_best_distance,
                            bat_distance_history,
                            bat_route_history,
                            bat_initial_best_distance,
                        ) = bat_algorithm_tsp(
                            cities=cities,
                            dist_matrix=dist_matrix,
                            pop_size=int(self.params["pop_size"]),
                            generations=int(self.params["generations"]),
                            mutation_rate=float(self.params["mutation_rate"]),
                            progress_callback=self._build_progress_callback(source="bat", series_name="bat"),
                            rng=bat_rng,
                            frequency_min=float(self.params.get("bat_frequency_min", 0.0)),
                            frequency_max=float(self.params.get("bat_frequency_max", 2.0)),
                            initial_loudness=float(self.params.get("bat_initial_loudness", 0.9)),
                            initial_pulse_rate=float(self.params.get("bat_initial_pulse", 0.15)),
                            alpha=float(self.params.get("bat_alpha", 0.96)),
                            gamma=float(self.params.get("bat_gamma", 0.9)),
                            max_guided_moves=int(self.params.get("bat_max_guided_moves", 8)),
                            local_walk_segment=int(self.params.get("bat_local_walk_segment", 5)),
                        )

                        bat_state["result"] = self._pack_solver_result(
                            label="bat",
                            best_route=bat_best_route,
                            best_distance=bat_best_distance,
                            best_distance_history=bat_distance_history,
                            best_route_history=bat_route_history,
                            initial_best_distance=bat_initial_best_distance,
                        )
                        bat_state["runtime"] = time.perf_counter() - bat_start
                    except RuntimeError as err:
                        if str(err) == STOP_EXCEPTION_TEXT:
                            bat_state["stopped"] = True
                        else:
                            bat_state["error"] = traceback.format_exc()
                    except Exception:
                        bat_state["error"] = traceback.format_exc()

                bat_thread = threading.Thread(target=run_bat, name="bat-comparison")
                bat_thread.start()

            if backend == "simpleai":
                simpleai_overrides = self._apply_simpleai_runtime_overrides()
                solver = genetic_algorithm_simpleai
            elif backend == "custom":
                solver = genetic_algorithm_custom
            else:
                raise ValueError("Backend must be either 'custom' or 'simpleai'.")

            primary_start = time.perf_counter()
            solver_kwargs = {
                "cities": cities,
                "dist_matrix": dist_matrix,
                "pop_size": int(self.params["pop_size"]),
                "generations": int(self.params["generations"]),
                "mutation_rate": float(self.params["mutation_rate"]),
                "crossover_rate": float(self.params["crossover_rate"]),
                "elite_size": int(self.params["elite_size"]),
                "tournament_size": int(self.params["tournament_size"]),
                "progress_callback": self._build_progress_callback(source="primary", series_name=backend),
            }
            if solver is genetic_algorithm_custom:
                solver_kwargs["return_metadata"] = True

            solver_result = solver(**solver_kwargs)
            if len(solver_result) == 6:
                (
                    best_route,
                    best_distance,
                    best_distance_history,
                    best_route_history,
                    initial_best_distance,
                    primary_metadata,
                ) = solver_result
            else:
                (
                    best_route,
                    best_distance,
                    best_distance_history,
                    best_route_history,
                    initial_best_distance,
                ) = solver_result
            primary_runtime = time.perf_counter() - primary_start

            primary_result = self._pack_solver_result(
                label=backend,
                best_route=best_route,
                best_distance=best_distance,
                best_distance_history=best_distance_history,
                best_route_history=best_route_history,
                initial_best_distance=initial_best_distance,
            )

            if self._stop_requested:
                stopped = True
        except RuntimeError as err:
            if str(err) == STOP_EXCEPTION_TEXT:
                stopped = True
            else:
                error_text = traceback.format_exc()
        except Exception:
            error_text = traceback.format_exc()
        finally:
            self._restore_simpleai_runtime_overrides(simpleai_overrides)

        if bat_thread is not None:
            if stopped or error_text is not None:
                self._stop_requested = True
            bat_thread.join()
            if bat_state.get("error") and error_text is None:
                error_text = str(bat_state["error"])
            if bat_state.get("stopped"):
                stopped = True
            comparison_result = bat_state.get("result")

        if stopped:
            self.stopped.emit()
            return
        if error_text is not None:
            self.failed.emit(error_text)
            return
        if cities is None or primary_result is None:
            self.failed.emit("Solver failed to return a result.")
            return

        total_runtime = time.perf_counter() - start_overall

        self.finished.emit(
            {
                "cities": cities,
                "backend": backend,
                "comparison_enabled": compare_enabled,
                "primary_result": primary_result,
                "comparison_result": comparison_result,
                "runtime_primary": primary_runtime,
                "runtime_bat": bat_state.get("runtime"),
                "runtime_total": total_runtime,
                "seed": seed,
                "primary_metadata": dict(primary_metadata),
                # Compatibility keys kept for existing non-comparison code paths.
                "best_route": list(primary_result["best_route"]),
                "best_distance": float(primary_result["best_distance"]),
                "best_distance_history": list(primary_result["best_distance_history"]),
                "best_route_history": [list(route) for route in primary_result["best_route_history"]],
                "initial_best_distance": float(primary_result["initial_best_distance"]),
            }
        )

    def _build_progress_callback(self, source: str, series_name: str):
        def progress_callback(payload: Dict[str, Any]) -> None:
            if self._stop_requested:
                raise RuntimeError(STOP_EXCEPTION_TEXT)

            frame = dict(payload)
            frame["source"] = source
            frame["series_name"] = series_name
            self.progress.emit(frame)

        return progress_callback

    def _pack_solver_result(
        self,
        label: str,
        best_route: List[int],
        best_distance: float,
        best_distance_history: List[float],
        best_route_history: List[List[int]],
        initial_best_distance: float,
    ) -> Dict[str, Any]:
        return {
            "label": str(label),
            "best_route": list(best_route),
            "best_distance": float(best_distance),
            "best_distance_history": list(best_distance_history),
            "best_route_history": [list(route) for route in best_route_history],
            "initial_best_distance": float(initial_best_distance),
        }

    @QtCore.pyqtSlot()
    def request_stop(self) -> None:
        self._stop_requested = True

    def _apply_simpleai_runtime_overrides(self) -> Dict[str, Any]:
        from app.solvers import simpleai_ga as simpleai_module

        mapping = {
            "SIMPLEAI_RESTARTS": int(self.params["simpleai_restarts"]),
            "SIMPLEAI_ENABLE_2OPT": bool(self.params["simpleai_enable_2opt"]),
            "SIMPLEAI_2OPT_MAX_PASSES": int(self.params["simpleai_2opt_max_passes"]),
            "SIMPLEAI_FITNESS_POWER": float(self.params["simpleai_fitness_power"]),
            "SIMPLEAI_USE_NATIVE_GENETIC": bool(self.params["simpleai_use_native"]),
            "SIMPLEAI_ENABLE_ELITISM": bool(self.params["simpleai_enable_elitism"]),
            "SIMPLEAI_DIVERSITY_RATE": float(self.params["simpleai_diversity_rate"]),
            "SIMPLEAI_EPSILON": float(self.params["simpleai_epsilon"]),
        }

        previous: Dict[str, Any] = {}
        for key, value in mapping.items():
            previous[key] = getattr(simpleai_module, key)
            setattr(simpleai_module, key, value)
        return previous

    def _restore_simpleai_runtime_overrides(self, previous: Dict[str, Any]) -> None:
        if not previous:
            return

        from app.solvers import simpleai_ga as simpleai_module

        for key, value in previous.items():
            setattr(simpleai_module, key, value)


class BatchWorker(QtCore.QObject):
    trial = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        cities: np.ndarray,
        dist_matrix: np.ndarray,
        base_experiment_name: str,
        num_trials: int,
        seed_offset: int,
    ) -> None:
        super().__init__()
        self.base_config = base_config
        self.param_grid = param_grid
        self.cities = cities
        self.dist_matrix = dist_matrix
        self.base_experiment_name = base_experiment_name
        self.num_trials = num_trials
        self.seed_offset = seed_offset
        self._stop_requested = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            from app.experiments.batch_runner import run_grid_search

            def per_trial_callback(payload: Dict[str, Any]) -> None:
                if self._stop_requested:
                    raise RuntimeError(STOP_EXCEPTION_TEXT)
                self.trial.emit(dict(payload))

            run_grid_search(
                base_config=self.base_config,
                param_grid=self.param_grid,
                dist_matrix=self.dist_matrix,
                cities=self.cities,
                base_experiment_name=self.base_experiment_name,
                num_trials=self.num_trials,
                seed_offset=self.seed_offset,
                per_trial_callback=per_trial_callback,
            )
            self.finished.emit({"event": "complete"})
        except RuntimeError as err:
            if str(err) == STOP_EXCEPTION_TEXT:
                self.failed.emit(STOP_EXCEPTION_TEXT)
            else:
                self.failed.emit(traceback.format_exc())
        except Exception:
            self.failed.emit(traceback.format_exc())

    @QtCore.pyqtSlot()
    def request_stop(self) -> None:
        self._stop_requested = True
