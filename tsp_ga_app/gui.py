from collections import deque
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from . import config as app_config
from .bat_solver import bat_algorithm_tsp
from .problem import compute_distance_matrix, generate_cities
from .simpleai_solver import genetic_algorithm_simpleai
from .solver import genetic_algorithm as genetic_algorithm_custom

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
        total_runtime: Optional[float] = None
        start_overall = time.perf_counter()

        try:
            seed = self.params["seed"]
            if seed is not None:
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
            (
                best_route,
                best_distance,
                best_distance_history,
                best_route_history,
                initial_best_distance,
            ) = solver(
                cities=cities,
                dist_matrix=dist_matrix,
                pop_size=int(self.params["pop_size"]),
                generations=int(self.params["generations"]),
                mutation_rate=float(self.params["mutation_rate"]),
                crossover_rate=float(self.params["crossover_rate"]),
                elite_size=int(self.params["elite_size"]),
                tournament_size=int(self.params["tournament_size"]),
                progress_callback=self._build_progress_callback(source="primary", series_name=backend),
            )
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
        from . import simpleai_solver as simpleai_module

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

        from . import simpleai_solver as simpleai_module

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
            from experiments.batch_runner import run_grid_search

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


class TSPControlPanel(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TSP Genetic Solver Studio")
        self.resize(1280, 820)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[SolverWorker] = None
        self._batch_thread: Optional[QtCore.QThread] = None
        self._batch_worker: Optional[BatchWorker] = None

        self.current_cities: Optional[np.ndarray] = None
        self.primary_steps: List[int] = []
        self.primary_distances: List[float] = []
        self.compare_steps: List[int] = []
        self.compare_distances: List[float] = []
        self.frame_buffer: Deque[Dict[str, Any]] = deque()
        self.dropped_frame_count = 0
        self.rendered_primary_count = 0
        self.rendered_compare_count = 0
        self._waiting_for_frames = False
        self._comparison_enabled = False
        self._solver_params: Dict[str, Any] = {}
        self._final_result_payload: Optional[Dict[str, Any]] = None
        self._run_start_time: Optional[float] = None
        self.top_runs: List[Dict[str, Any]] = []
        self._overlay_top_runs = False
        self.loaded_datasets: Dict[str, np.ndarray] = {}
        self.batch_trial_distances: List[float] = []
        self.batch_convergence_mode = "trial"
        self._stop_requested_ui = False
        self.run_history: List[Dict[str, Any]] = []
        self._run_counter = 0
        self._focused_run_index: Optional[int] = None
        self._show_convergence_overlay = True
        self._raw_primary_steps: List[int] = []
        self._raw_primary_distances: List[float] = []
        self._raw_primary_avg_fitness: List[float] = []
        self._raw_primary_diversity: List[int] = []
        self._raw_primary_avg_steps: List[int] = []
        self._raw_primary_div_steps: List[int] = []
        self._raw_compare_steps: List[int] = []
        self._raw_compare_distances: List[float] = []
        self._raw_compare_avg_fitness: List[float] = []
        self._raw_compare_diversity: List[int] = []
        self._raw_compare_avg_steps: List[int] = []
        self._raw_compare_div_steps: List[int] = []
        self._latest_primary_metrics: Dict[str, Optional[float]] = {"avg_fitness": None, "diversity": None}
        self._metrics_dialog: Optional[QtWidgets.QDialog] = None
        self._metrics_canvas: Optional[FigureCanvas] = None
        self._metrics_ax = None

        self._animation_timer = QtCore.QTimer(self)
        self._animation_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._animation_timer.timeout.connect(self._consume_buffered_frame)

        self._build_ui()
        self._sync_population_dependent_ranges()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        self.setCentralWidget(root)

        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        controls = self._build_controls_panel()
        plots = self._build_plots_panel()

        controls_scroll = QtWidgets.QScrollArea(self)
        controls_scroll.setWidget(controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setMaximumWidth(380)

        layout.addWidget(controls_scroll)
        layout.addWidget(plots, stretch=1)

    def _build_controls_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setSpacing(8)

        data_group = QtWidgets.QGroupBox("Data")
        data_form = QtWidgets.QFormLayout(data_group)

        self.load_json_button = QtWidgets.QPushButton("Load Cities JSON")
        self.load_json_button.clicked.connect(self._load_cities_json)

        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItem("Random cities")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)

        self.apply_dataset_button = QtWidgets.QPushButton("Load to space")
        self.apply_dataset_button.clicked.connect(self._apply_selected_dataset)

        self.city_seed_check = QtWidgets.QCheckBox("Use city seed")
        self.city_seed_spin = QtWidgets.QSpinBox()
        self.city_seed_spin.setRange(0, 1_000_000_000)
        self.city_seed_spin.setValue(42)
        self.city_seed_spin.setEnabled(False)
        self.city_seed_check.toggled.connect(self.city_seed_spin.setEnabled)

        data_form.addRow(self.load_json_button)
        data_form.addRow("Dataset", self.dataset_combo)
        data_form.addRow(self.apply_dataset_button)
        data_form.addRow(self.city_seed_check, self.city_seed_spin)

        general_group = QtWidgets.QGroupBox("General")
        general_form = QtWidgets.QFormLayout(general_group)

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["custom", "simpleai"])
        self.backend_combo.setCurrentText(app_config.SOLVER_BACKEND)

        self.num_cities_spin = QtWidgets.QSpinBox()
        self.num_cities_spin.setRange(5, 500)
        self.num_cities_spin.setValue(app_config.NUM_CITIES)

        self.pop_size_spin = QtWidgets.QSpinBox()
        self.pop_size_spin.setRange(2, 3000)
        self.pop_size_spin.setValue(app_config.POP_SIZE)

        self.generations_spin = QtWidgets.QSpinBox()
        self.generations_spin.setRange(1, 10000)
        self.generations_spin.setValue(app_config.GENERATIONS)

        self.mutation_spin = QtWidgets.QDoubleSpinBox()
        self.mutation_spin.setRange(0.0, 1.0)
        self.mutation_spin.setSingleStep(0.01)
        self.mutation_spin.setDecimals(3)
        self.mutation_spin.setValue(app_config.MUTATION_RATE)

        self.crossover_spin = QtWidgets.QDoubleSpinBox()
        self.crossover_spin.setRange(0.0, 1.0)
        self.crossover_spin.setSingleStep(0.01)
        self.crossover_spin.setDecimals(3)
        self.crossover_spin.setValue(app_config.CROSSOVER_RATE)

        self.elite_spin = QtWidgets.QSpinBox()
        self.elite_spin.setRange(0, app_config.POP_SIZE)
        self.elite_spin.setValue(app_config.ELITE_SIZE)

        self.tournament_spin = QtWidgets.QSpinBox()
        self.tournament_spin.setRange(2, app_config.POP_SIZE)
        self.tournament_spin.setValue(app_config.TOURNAMENT_SIZE)

        self.seed_check = QtWidgets.QCheckBox("Use fixed seed")
        self.seed_check.setChecked(app_config.RANDOM_SEED is not None)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999999999)
        self.seed_spin.setValue(app_config.RANDOM_SEED if app_config.RANDOM_SEED is not None else 0)
        self.seed_spin.setEnabled(self.seed_check.isChecked())

        self.seed_check.toggled.connect(self.seed_spin.setEnabled)
        self.pop_size_spin.valueChanged.connect(self._sync_population_dependent_ranges)

        self.enable_bat_compare_check = QtWidgets.QCheckBox("Enable BAT comparison")
        self.enable_bat_compare_check.setChecked(bool(app_config.ENABLE_BAT_COMPARISON))
        self.enable_bat_compare_check.setToolTip(
            "When enabled, BAT-inspired solver runs after selected backend and is shown side-by-side."
        )

        general_form.addRow("Backend", self.backend_combo)
        general_form.addRow("Cities", self.num_cities_spin)
        general_form.addRow("Population", self.pop_size_spin)
        general_form.addRow("Generations", self.generations_spin)
        general_form.addRow("Mutation rate", self.mutation_spin)
        general_form.addRow("Crossover rate", self.crossover_spin)
        general_form.addRow("Elite size", self.elite_spin)
        general_form.addRow("Tournament size", self.tournament_spin)
        general_form.addRow(self.seed_check, self.seed_spin)
        general_form.addRow(self.enable_bat_compare_check)

        simpleai_group = QtWidgets.QGroupBox("simpleAI tuning")
        simpleai_form = QtWidgets.QFormLayout(simpleai_group)

        self.simpleai_restarts_spin = QtWidgets.QSpinBox()
        self.simpleai_restarts_spin.setRange(1, 200)
        self.simpleai_restarts_spin.setValue(app_config.SIMPLEAI_RESTARTS)

        self.simpleai_enable_2opt_check = QtWidgets.QCheckBox()
        self.simpleai_enable_2opt_check.setChecked(app_config.SIMPLEAI_ENABLE_2OPT)

        self.simpleai_2opt_passes_spin = QtWidgets.QSpinBox()
        self.simpleai_2opt_passes_spin.setRange(1, 1000)
        self.simpleai_2opt_passes_spin.setValue(app_config.SIMPLEAI_2OPT_MAX_PASSES)

        self.simpleai_fitness_power_spin = QtWidgets.QDoubleSpinBox()
        self.simpleai_fitness_power_spin.setRange(0.5, 5.0)
        self.simpleai_fitness_power_spin.setSingleStep(0.1)
        self.simpleai_fitness_power_spin.setDecimals(3)
        self.simpleai_fitness_power_spin.setValue(app_config.SIMPLEAI_FITNESS_POWER)

        self.simpleai_use_native_check = QtWidgets.QCheckBox()
        self.simpleai_use_native_check.setChecked(app_config.SIMPLEAI_USE_NATIVE_GENETIC)

        self.simpleai_elitism_check = QtWidgets.QCheckBox()
        self.simpleai_elitism_check.setChecked(app_config.SIMPLEAI_ENABLE_ELITISM)

        self.simpleai_diversity_spin = QtWidgets.QDoubleSpinBox()
        self.simpleai_diversity_spin.setRange(0.0, 1.0)
        self.simpleai_diversity_spin.setSingleStep(0.01)
        self.simpleai_diversity_spin.setDecimals(3)
        self.simpleai_diversity_spin.setValue(app_config.SIMPLEAI_DIVERSITY_RATE)

        self.simpleai_epsilon_spin = QtWidgets.QDoubleSpinBox()
        self.simpleai_epsilon_spin.setRange(1e-12, 1.0)
        self.simpleai_epsilon_spin.setDecimals(12)
        self.simpleai_epsilon_spin.setSingleStep(1e-4)
        self.simpleai_epsilon_spin.setValue(app_config.SIMPLEAI_EPSILON)

        simpleai_form.addRow("Restarts", self.simpleai_restarts_spin)
        simpleai_form.addRow("Enable 2-opt", self.simpleai_enable_2opt_check)
        simpleai_form.addRow("2-opt max passes", self.simpleai_2opt_passes_spin)
        simpleai_form.addRow("Fitness power", self.simpleai_fitness_power_spin)
        simpleai_form.addRow("Use native simpleAI", self.simpleai_use_native_check)
        simpleai_form.addRow("Enable elitism", self.simpleai_elitism_check)
        simpleai_form.addRow("Diversity rate", self.simpleai_diversity_spin)
        simpleai_form.addRow("Fitness epsilon", self.simpleai_epsilon_spin)

        playback_group = QtWidgets.QGroupBox("Live playback")
        playback_form = QtWidgets.QFormLayout(playback_group)

        self.animation_interval_spin = QtWidgets.QSpinBox()
        self.animation_interval_spin.setRange(10, 2000)
        self.animation_interval_spin.setValue(max(DEFAULT_ANIMATION_INTERVAL_MS, app_config.ANIMATION_INTERVAL_MS))
        self.animation_interval_spin.setSuffix(" ms")
        self.animation_interval_spin.setToolTip("Playback speed for buffered live animation.")

        self.animation_buffer_limit_spin = QtWidgets.QSpinBox()
        self.animation_buffer_limit_spin.setRange(0, 200000)
        self.animation_buffer_limit_spin.setValue(0)
        self.animation_buffer_limit_spin.setToolTip(
            "0 means unlimited cache. If >0 and full, oldest frames are dropped."
        )

        self.animation_interval_spin.valueChanged.connect(self._update_animation_interval)

        playback_form.addRow("Frame interval", self.animation_interval_spin)
        playback_form.addRow("Buffer limit", self.animation_buffer_limit_spin)

        convergence_group = QtWidgets.QGroupBox("Convergence view")
        convergence_form = QtWidgets.QFormLayout(convergence_group)

        self.convergence_metric_combo = QtWidgets.QComboBox()
        self.convergence_metric_combo.addItems([
            "Best distance",
            "Avg fitness",
            "Diversity",
            "Convergence speed",
        ])
        self.convergence_metric_combo.currentTextChanged.connect(self._on_convergence_metric_changed)

        self.run_focus_combo = QtWidgets.QComboBox()
        self.run_focus_combo.addItem("All runs")
        self.run_focus_combo.currentIndexChanged.connect(self._on_run_focus_changed)

        self.reset_convergence_button = QtWidgets.QPushButton("Reset convergence")
        self.reset_convergence_button.clicked.connect(self._reset_convergence_history)

        self.show_metrics_button = QtWidgets.QPushButton("Show metrics")
        self.show_metrics_button.clicked.connect(self._show_metrics_window)

        convergence_form.addRow("Metric", self.convergence_metric_combo)
        convergence_form.addRow("Focus run", self.run_focus_combo)
        convergence_buttons = QtWidgets.QHBoxLayout()
        convergence_buttons.addWidget(self.reset_convergence_button)
        convergence_buttons.addWidget(self.show_metrics_button)
        convergence_form.addRow(convergence_buttons)

        batch_group = QtWidgets.QGroupBox("Batch")
        batch_form = QtWidgets.QFormLayout(batch_group)

        self.batch_param_grid_edit = QtWidgets.QLineEdit()
        self.batch_param_grid_edit.setPlaceholderText('{"mutation_rate": [0.01, 0.05], "crossover_type": ["pmx", "order"]}')
        self.batch_param_grid_edit.setText('{"mutation_rate": [0.01, 0.05], "crossover_type": ["pmx", "order"]}')

        self.batch_trials_spin = QtWidgets.QSpinBox()
        self.batch_trials_spin.setRange(1, 100)
        self.batch_trials_spin.setValue(2)

        self.batch_seed_offset_spin = QtWidgets.QSpinBox()
        self.batch_seed_offset_spin.setRange(0, 1_000_000_000)
        self.batch_seed_offset_spin.setValue(42)

        self.batch_top_n_spin = QtWidgets.QSpinBox()
        self.batch_top_n_spin.setRange(1, 20)
        self.batch_top_n_spin.setValue(10)

        self.batch_convergence_mode_combo = QtWidgets.QComboBox()
        self.batch_convergence_mode_combo.addItems(["Off", "Trial best distance", "Running best distance"])
        self.batch_convergence_mode_combo.currentTextChanged.connect(self._on_batch_convergence_mode_changed)

        self.overlay_top_runs_check = QtWidgets.QCheckBox("Overlay top 10 batch runs")
        self.overlay_top_runs_check.toggled.connect(self._on_overlay_toggle)

        self.batch_run_button = QtWidgets.QPushButton("Run Batch")
        self.batch_clear_button = QtWidgets.QPushButton("Clear Outputs")

        self.batch_run_button.clicked.connect(self._start_batch)
        self.batch_clear_button.clicked.connect(self._clear_outputs)

        batch_button_row = QtWidgets.QHBoxLayout()
        batch_button_row.addWidget(self.batch_run_button)
        batch_button_row.addWidget(self.batch_clear_button)

        batch_form.addRow("Param grid", self.batch_param_grid_edit)
        batch_form.addRow("Batch trials", self.batch_trials_spin)
        batch_form.addRow("Seed offset", self.batch_seed_offset_spin)
        batch_form.addRow("Top overlays", self.batch_top_n_spin)
        batch_form.addRow("Convergence view", self.batch_convergence_mode_combo)
        batch_form.addRow(self.overlay_top_runs_check)
        batch_form.addRow(batch_button_row)

        buttons_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.reset_button = QtWidgets.QPushButton("Reset fields")

        buttons_row.addWidget(self.run_button)
        buttons_row.addWidget(self.stop_button)
        buttons_row.addWidget(self.reset_button)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)

        self.run_button.clicked.connect(self._start_solver)
        self.stop_button.clicked.connect(self._stop_solver)
        self.reset_button.clicked.connect(self._reset_fields)

        vbox.addWidget(general_group)
        vbox.addWidget(simpleai_group)
        vbox.addWidget(playback_group)
        vbox.addWidget(convergence_group)
        vbox.addWidget(data_group)
        vbox.addWidget(batch_group)
        vbox.addLayout(buttons_row)
        vbox.addWidget(self.status_label)
        vbox.addStretch(1)

        return panel

    def _build_plots_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setSpacing(8)

        route_group = QtWidgets.QGroupBox("Live routes")
        route_layout = QtWidgets.QVBoxLayout(route_group)
        self.route_figure = Figure(figsize=(10, 5), tight_layout=True)
        self.route_canvas = FigureCanvas(self.route_figure)
        self.route_ax_primary = self.route_figure.add_subplot(121)
        self.route_ax_compare = self.route_figure.add_subplot(122)
        route_layout.addWidget(self.route_canvas)

        convergence_group = QtWidgets.QGroupBox("Convergence")
        convergence_layout = QtWidgets.QVBoxLayout(convergence_group)
        self.conv_figure = Figure(figsize=(8, 3), tight_layout=True)
        self.conv_canvas = FigureCanvas(self.conv_figure)
        self.conv_ax = self.conv_figure.add_subplot(111)
        convergence_layout.addWidget(self.conv_canvas)

        vbox.addWidget(route_group, stretch=3)
        vbox.addWidget(convergence_group, stretch=2)

        self._draw_empty_route()
        self._draw_empty_convergence()

        return panel

    def _sync_population_dependent_ranges(self) -> None:
        pop_size = max(2, int(self.pop_size_spin.value()))
        self.elite_spin.setMaximum(pop_size)
        self.tournament_spin.setMaximum(pop_size)
        if self.tournament_spin.value() > pop_size:
            self.tournament_spin.setValue(pop_size)

    def _generate_random_cities(self, count: int) -> np.ndarray:
        if hasattr(self, "city_seed_check") and self.city_seed_check.isChecked():
            seed = int(self.city_seed_spin.value())
            rng = np.random.default_rng(seed)
            return rng.uniform(0.0, 100.0, size=(count, 2))
        return generate_cities(count)

    def _draw_city_scatter(self, title: str) -> None:
        if self.current_cities is None:
            self._draw_empty_route()
            return

        self.route_ax_primary.clear()
        self.route_ax_primary.scatter(self.current_cities[:, 0], self.current_cities[:, 1], c="tab:red", s=35)
        self.route_ax_primary.set_title(title)
        self.route_ax_primary.set_xlabel("X")
        self.route_ax_primary.set_ylabel("Y")
        self.route_ax_primary.grid(alpha=0.3)
        self.route_ax_primary.set_aspect("equal", adjustable="box")

        self._draw_route_axis_placeholder(
            axis=self.route_ax_compare,
            title="BAT comparison",
            message="Run solver to compare",
        )
        self.route_canvas.draw_idle()

    def _extract_city_point(self, item: Any) -> Optional[List[float]]:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                return [float(item["x"]), float(item["y"])]
            if "coord" in item and isinstance(item["coord"], (list, tuple)) and len(item["coord"]) >= 2:
                return [float(item["coord"][0]), float(item["coord"][1])]
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            return [float(item[0]), float(item[1])]
        return None

    def _looks_like_point_map(self, payload: Dict[Any, Any]) -> bool:
        if not payload:
            return False
        for value in payload.values():
            if self._extract_city_point(value) is None:
                return False
        return True

    def _parse_cities_json(self, payload: Any) -> Optional[np.ndarray]:
        if isinstance(payload, dict):
            if "cities" in payload:
                payload = payload["cities"]
            elif "points" in payload:
                payload = payload["points"]

        points: List[List[float]] = []

        if isinstance(payload, dict):
            def sort_key(item: Any) -> Any:
                key = item[0]
                if isinstance(key, int):
                    return (0, key)
                if isinstance(key, str) and key.isdigit():
                    return (0, int(key))
                return (1, str(key))

            for _, value in sorted(payload.items(), key=sort_key):
                point = self._extract_city_point(value)
                if point is not None:
                    points.append(point)
        elif isinstance(payload, list):
            for item in payload:
                point = self._extract_city_point(item)
                if point is not None:
                    points.append(point)
        else:
            return None

        if not points:
            return None
        return np.asarray(points, dtype=float)

    def _extract_city_datasets(self, payload: Any, fallback_name: str) -> Dict[str, np.ndarray]:
        datasets: Dict[str, np.ndarray] = {}

        if isinstance(payload, dict):
            if "cities" in payload or "points" in payload:
                dataset_name = str(payload.get("name")) if payload.get("name") else fallback_name
                cities = self._parse_cities_json(payload)
                if cities is not None:
                    datasets[dataset_name] = cities
                return datasets

            if self._looks_like_point_map(payload):
                cities = self._parse_cities_json(payload)
                if cities is not None:
                    datasets[fallback_name] = cities
                return datasets

            for key, value in payload.items():
                if key in {"name", "description"}:
                    continue
                cities = self._parse_cities_json(value)
                if cities is not None:
                    datasets[str(key)] = cities
            return datasets

        if isinstance(payload, list):
            cities = self._parse_cities_json(payload)
            if cities is not None:
                datasets[fallback_name] = cities
        return datasets

    def _load_cities_json(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load cities JSON", "", "JSON Files (*.json)")
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            file_label = os.path.basename(file_path)
            datasets = self._extract_city_datasets(payload, file_label)
            if not datasets:
                raise ValueError("JSON does not contain a valid city coordinate list or dataset map.")

            for name, cities in datasets.items():
                self.loaded_datasets[name] = cities
                if self.dataset_combo.findText(name) == -1:
                    self.dataset_combo.addItem(name)

            active_name = next(iter(datasets))
            self.dataset_combo.setCurrentText(active_name)
            self._apply_selected_dataset()

            if len(datasets) == 1:
                self.status_label.setText(
                    f"Loaded {self.current_cities.shape[0]} cities from {active_name}"
                )
            else:
                self.status_label.setText(
                    f"Loaded {len(datasets)} datasets from {file_label}"
                )
        except Exception as err:
            QtWidgets.QMessageBox.critical(self, "Load JSON", str(err))

    def _apply_selected_dataset(self) -> None:
        name = self.dataset_combo.currentText()
        if name in self.loaded_datasets:
            self.current_cities = self.loaded_datasets[name]
            self.num_cities_spin.setValue(int(self.current_cities.shape[0]))
            self._draw_city_scatter(f"Loaded dataset: {name}")
            self.status_label.setText(f"Loaded dataset {name}")
            return

        num_cities = int(self.num_cities_spin.value())
        self.current_cities = self._generate_random_cities(num_cities)
        self._draw_city_scatter(f"Random cities ({num_cities})")
        self.status_label.setText(f"Generated {num_cities} random cities")

    def _on_dataset_changed(self) -> None:
        name = self.dataset_combo.currentText()
        if name in self.loaded_datasets:
            self.status_label.setText(f"Selected dataset {name}. Click 'Load to space' to display.")
        else:
            self.status_label.setText("Selected random cities. Click 'Load to space' to display.")

    def _on_batch_convergence_mode_changed(self, text: str) -> None:
        self.batch_convergence_mode = "off" if text == "Off" else ("running" if text == "Running best distance" else "trial")
        self._draw_convergence()

    def _on_convergence_metric_changed(self, _text: str) -> None:
        self._draw_convergence()
        self._update_metrics_plot()

    def _on_run_focus_changed(self, index: int) -> None:
        self._focused_run_index = index - 1 if index > 0 else None
        self._draw_convergence()
        self._update_metrics_plot()

    def _get_focus_index(self) -> Optional[int]:
        return self._focused_run_index

    def _refresh_run_focus_combo(self) -> None:
        if not hasattr(self, "run_focus_combo"):
            return

        current_focus = self._focused_run_index
        self.run_focus_combo.blockSignals(True)
        self.run_focus_combo.clear()
        self.run_focus_combo.addItem("All runs")
        for idx in range(len(self.run_history)):
            self.run_focus_combo.addItem(f"Run {idx + 1}")
        self.run_focus_combo.blockSignals(False)

        if current_focus is None:
            self.run_focus_combo.setCurrentIndex(0)
        else:
            focus_index = min(current_focus + 1, self.run_focus_combo.count() - 1)
            self.run_focus_combo.setCurrentIndex(focus_index)

    def _reset_convergence_history(self) -> None:
        self.run_history = []
        self._run_counter = 0
        self.batch_trial_distances = []
        self._focused_run_index = None
        self._show_convergence_overlay = True
        self._refresh_run_focus_combo()
        self._draw_convergence()
        self._update_metrics_plot()

    def _record_live_metrics(self, payload: Dict[str, Any]) -> None:
        source = str(payload.get("source", "primary")).lower()
        generation = payload.get("generation")
        distance = payload.get("best_distance")
        avg_fitness = payload.get("avg_fitness")
        diversity = payload.get("diversity")

        if source == "bat":
            if generation is not None:
                self._raw_compare_steps.append(int(generation))
            if distance is not None:
                self._raw_compare_distances.append(float(distance))
            if avg_fitness is not None:
                self._raw_compare_avg_fitness.append(float(avg_fitness))
                if generation is not None:
                    self._raw_compare_avg_steps.append(int(generation))
            if diversity is not None:
                self._raw_compare_diversity.append(int(diversity))
                if generation is not None:
                    self._raw_compare_div_steps.append(int(generation))
            return

        if generation is not None:
            self._raw_primary_steps.append(int(generation))
        if distance is not None:
            self._raw_primary_distances.append(float(distance))
        if avg_fitness is not None:
            self._raw_primary_avg_fitness.append(float(avg_fitness))
            if generation is not None:
                self._raw_primary_avg_steps.append(int(generation))
            self._latest_primary_metrics["avg_fitness"] = float(avg_fitness)
        if diversity is not None:
            self._raw_primary_diversity.append(int(diversity))
            if generation is not None:
                self._raw_primary_div_steps.append(int(generation))
            self._latest_primary_metrics["diversity"] = float(diversity)

    def _show_metrics_window(self) -> None:
        if self._metrics_dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Run metrics")
            dialog.resize(800, 520)

            layout = QtWidgets.QVBoxLayout(dialog)
            controls = QtWidgets.QHBoxLayout()

            self.metrics_chart_combo = QtWidgets.QComboBox()
            self.metrics_chart_combo.addItems(["Dispersion (boxplot)", "Convergence speed"])
            self.metrics_chart_combo.currentTextChanged.connect(self._update_metrics_plot)

            self.metrics_value_combo = QtWidgets.QComboBox()
            self.metrics_value_combo.addItems(["Best distance", "Runtime", "Avg fitness", "Diversity"])
            self.metrics_value_combo.currentTextChanged.connect(self._update_metrics_plot)

            controls.addWidget(QtWidgets.QLabel("Chart"))
            controls.addWidget(self.metrics_chart_combo)
            controls.addWidget(QtWidgets.QLabel("Metric"))
            controls.addWidget(self.metrics_value_combo)
            controls.addStretch(1)

            layout.addLayout(controls)

            figure = Figure(figsize=(8, 4.5), tight_layout=True)
            self._metrics_canvas = FigureCanvas(figure)
            self._metrics_ax = figure.add_subplot(111)
            layout.addWidget(self._metrics_canvas)

            self._metrics_dialog = dialog

        self._update_metrics_plot()
        self._metrics_dialog.show()
        self._metrics_dialog.raise_()
        self._metrics_dialog.activateWindow()

    def _update_metrics_plot(self) -> None:
        if self._metrics_dialog is None or self._metrics_ax is None:
            return

        self._metrics_ax.clear()
        runs = list(self.run_history)
        focus_index = self._get_focus_index()

        if not runs:
            self._metrics_ax.set_title("No runs recorded")
            self._metrics_canvas.draw_idle()
            return

        chart_mode = self.metrics_chart_combo.currentText() if hasattr(self, "metrics_chart_combo") else "Dispersion (boxplot)"
        metric_name = self.metrics_value_combo.currentText() if hasattr(self, "metrics_value_combo") else "Best distance"

        if chart_mode == "Dispersion (boxplot)":
            values = []
            labels = []
            for idx, run in enumerate(runs):
                primary = run.get("primary", {})
                value = None
                if metric_name == "Best distance":
                    distances = primary.get("best_distance") or []
                    if distances:
                        value = float(distances[-1])
                elif metric_name == "Runtime":
                    value = run.get("runtime_total")
                elif metric_name == "Avg fitness":
                    avg_vals = primary.get("avg_fitness") or []
                    if avg_vals:
                        value = float(avg_vals[-1])
                elif metric_name == "Diversity":
                    div_vals = primary.get("diversity") or []
                    if div_vals:
                        value = float(div_vals[-1])

                if value is None:
                    continue

                values.append(value)
                labels.append(f"Run {idx + 1}")

            if not values:
                self._metrics_ax.set_title("No data for selected metric")
            else:
                self._metrics_ax.boxplot(values, vert=True, showmeans=True)
                self._metrics_ax.set_xticklabels(["All runs"])
                self._metrics_ax.set_title(f"Dispersion: {metric_name}")
            self._metrics_canvas.draw_idle()
            return

        # Convergence speed chart
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
        for idx, run in enumerate(runs):
            distances = run.get("primary", {}).get("best_distance") or []
            alpha = 1.0 if focus_index is None or idx == focus_index else 0.2
            linewidth = 2.2 if focus_index is None or idx == focus_index else 1.2

            if len(distances) >= 2:
                self._metrics_ax.plot(
                    list(range(2, len(distances) + 1)),
                    [float(prev) - float(curr) for prev, curr in zip(distances[:-1], distances[1:])],
                    color=colors[idx % len(colors)],
                    linewidth=linewidth,
                    alpha=alpha,
                    label=f"Run {idx + 1}",
                )

            compare_distances = run.get("compare", {}).get("best_distance") or []
            if len(compare_distances) >= 2:
                self._metrics_ax.plot(
                    list(range(2, len(compare_distances) + 1)),
                    [float(prev) - float(curr) for prev, curr in zip(compare_distances[:-1], compare_distances[1:])],
                    color=colors[idx % len(colors)],
                    linewidth=linewidth,
                    linestyle="--",
                    alpha=alpha,
                    label=f"Run {idx + 1} bat",
                )

        self._metrics_ax.set_title("Convergence speed (distance improvement per gen)")
        self._metrics_ax.set_xlabel("Generation")
        self._metrics_ax.set_ylabel("Improvement")
        self._metrics_ax.grid(alpha=0.3)
        if focus_index is not None or len(runs) <= 3:
            self._metrics_ax.legend(loc="upper right")
        self._metrics_canvas.draw_idle()

    def _is_solver_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _is_batch_running(self) -> bool:
        return self._batch_thread is not None and self._batch_thread.isRunning()

    def _is_busy(self) -> bool:
        return self._is_solver_running() or self._is_batch_running() or bool(self.frame_buffer) or self._final_result_payload is not None

    def _update_run_stop_buttons(self) -> None:
        self.run_button.setEnabled(not self._is_busy())
        self.stop_button.setEnabled(self._is_solver_running() or self._is_batch_running())
        if hasattr(self, "batch_run_button"):
            self.batch_run_button.setEnabled(not self._is_busy())

    def _update_animation_interval(self, _value: int = 0) -> None:
        if self._animation_timer.isActive():
            self._animation_timer.setInterval(int(self.animation_interval_spin.value()))

    def _start_playback_timer(self) -> None:
        self._animation_timer.setInterval(int(self.animation_interval_spin.value()))
        if not self._animation_timer.isActive():
            self._animation_timer.start()

    def _stop_playback_timer_if_idle(self) -> None:
        if self._is_solver_running() or self.frame_buffer or self._final_result_payload is not None:
            return
        if self._animation_timer.isActive():
            self._animation_timer.stop()

    def _format_runtime_status(self, prefix: str) -> str:
        backend = str(self._solver_params.get("backend", "-")).lower()
        return (
            f"{prefix} | backend: {backend} | buffer: {len(self.frame_buffer)} | "
            f"rendered primary: {self.rendered_primary_count} | "
            f"rendered bat: {self.rendered_compare_count} | dropped: {self.dropped_frame_count}"
        )

    def _reset_live_state(self) -> None:
        self.frame_buffer.clear()
        self._final_result_payload = None
        self.primary_steps = []
        self.primary_distances = []
        self.compare_steps = []
        self.compare_distances = []
        self._raw_primary_steps = []
        self._raw_primary_distances = []
        self._raw_primary_avg_fitness = []
        self._raw_primary_diversity = []
        self._raw_primary_avg_steps = []
        self._raw_primary_div_steps = []
        self._raw_compare_steps = []
        self._raw_compare_distances = []
        self._raw_compare_avg_fitness = []
        self._raw_compare_diversity = []
        self._raw_compare_avg_steps = []
        self._raw_compare_div_steps = []
        self._latest_primary_metrics = {"avg_fitness": None, "diversity": None}
        self.dropped_frame_count = 0
        self.rendered_primary_count = 0
        self.rendered_compare_count = 0
        self._waiting_for_frames = False

    def _on_overlay_toggle(self, checked: bool) -> None:
        self._overlay_top_runs = bool(checked)
        if self._overlay_top_runs and self.top_runs:
            self._draw_overlay_routes(self.route_ax_primary)
            self.route_canvas.draw_idle()

    def _collect_params(self) -> Dict[str, Any]:
        seed_value: Optional[int]
        if self.seed_check.isChecked():
            seed_value = int(self.seed_spin.value())
        else:
            seed_value = None

        pop_size = int(self.pop_size_spin.value())
        elite_size = min(int(self.elite_spin.value()), pop_size)
        tournament_size = min(max(2, int(self.tournament_spin.value())), pop_size)

        return {
            "backend": self.backend_combo.currentText(),
            "enable_bat_comparison": bool(self.enable_bat_compare_check.isChecked()),
            "num_cities": int(self.num_cities_spin.value()),
            "pop_size": pop_size,
            "generations": int(self.generations_spin.value()),
            "mutation_rate": float(self.mutation_spin.value()),
            "crossover_rate": float(self.crossover_spin.value()),
            "elite_size": elite_size,
            "tournament_size": tournament_size,
            "seed": seed_value,
            "simpleai_restarts": int(self.simpleai_restarts_spin.value()),
            "simpleai_enable_2opt": bool(self.simpleai_enable_2opt_check.isChecked()),
            "simpleai_2opt_max_passes": int(self.simpleai_2opt_passes_spin.value()),
            "simpleai_fitness_power": float(self.simpleai_fitness_power_spin.value()),
            "simpleai_use_native": bool(self.simpleai_use_native_check.isChecked()),
            "simpleai_enable_elitism": bool(self.simpleai_elitism_check.isChecked()),
            "simpleai_diversity_rate": float(self.simpleai_diversity_spin.value()),
            "simpleai_epsilon": float(self.simpleai_epsilon_spin.value()),
        }

    def _start_solver(self) -> None:
        if self._is_busy():
            self.status_label.setText(self._format_runtime_status("Busy with current run/playback"))
            return

        self._stop_requested_ui = False
        self._show_convergence_overlay = False
        params = self._collect_params()
        self._solver_params = dict(params)
        self._comparison_enabled = bool(params.get("enable_bat_comparison", False))
        self._reset_live_state()
        self._draw_empty_route()
        self._draw_convergence()
        self._start_playback_timer()

        self._run_start_time = time.time()

        if self.current_cities is None:
            self.current_cities = self._generate_random_cities(int(params["num_cities"]))

        params["num_cities"] = int(self.current_cities.shape[0])
        params["cities"] = np.asarray(self.current_cities, dtype=float)
        self._solver_params["num_cities"] = int(self.current_cities.shape[0])

        self._thread = QtCore.QThread(self)
        self._worker = SolverWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.stopped.connect(self._on_stopped)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.stopped.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker_thread)

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if hasattr(self, "batch_run_button"):
            self.batch_run_button.setEnabled(False)
        self.status_label.setText(self._format_runtime_status("Running solver + live animation"))
        self._thread.start()

    def _stop_solver(self) -> None:
        self._stop_requested_ui = True
        if self._worker is not None:
            self._worker.request_stop()
        if self._batch_worker is not None:
            self._batch_worker.request_stop()
        self.frame_buffer.clear()
        self._final_result_payload = None
        self._waiting_for_frames = False
        if self._animation_timer.isActive():
            self._animation_timer.stop()
        self.status_label.setText("Stop requested. Waiting for current iteration...")
        self._update_run_stop_buttons()

    def _start_batch(self) -> None:
        if self._is_busy():
            self.status_label.setText("Busy with current run/playback")
            return

        self._stop_requested_ui = False

        try:
            param_grid = json.loads(self.batch_param_grid_edit.text().strip() or "{}")
        except json.JSONDecodeError as err:
            QtWidgets.QMessageBox.warning(self, "Invalid JSON", f"Batch parameter grid is invalid: {err}")
            return

        if not isinstance(param_grid, dict):
            QtWidgets.QMessageBox.warning(self, "Invalid JSON", "Batch parameter grid must be a JSON object.")
            return

        if self.current_cities is None:
            self.current_cities = self._generate_random_cities(int(self.num_cities_spin.value()))

        dist_matrix = compute_distance_matrix(self.current_cities)
        base_config = self._collect_params()
        self.top_runs = []
        self._overlay_top_runs = self.overlay_top_runs_check.isChecked()
        self.batch_trial_distances = []
        num_trials = int(self.batch_trials_spin.value())
        seed_offset = int(self.batch_seed_offset_spin.value())
        base_experiment_name = f"batch_{self.backend_combo.currentText()}"

        self._batch_thread = QtCore.QThread(self)
        self._batch_worker = BatchWorker(
            base_config,
            param_grid,
            self.current_cities,
            dist_matrix,
            base_experiment_name,
            num_trials,
            seed_offset,
        )
        self._batch_worker.moveToThread(self._batch_thread)

        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.trial.connect(self._on_batch_trial)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._cleanup_batch_thread)

        self.batch_run_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Running batch experiment...")
        self._batch_thread.start()

    @QtCore.pyqtSlot(dict)
    def _on_batch_trial(self, payload: Dict[str, Any]) -> None:
        if self._stop_requested_ui:
            return
        best_distance = payload.get("best_distance")
        best_route = payload.get("best_route")
        experiment_id = str(payload.get("experiment_id", "batch_trial"))

        if best_distance is None or best_route is None:
            return

        self.top_runs.append(
            {
                "experiment_id": experiment_id,
                "best_distance": float(best_distance),
                "best_route": list(best_route),
                "folder_path": payload.get("folder_path"),
            }
        )
        self.top_runs.sort(key=lambda item: item["best_distance"])
        self.top_runs = self.top_runs[: max(1, int(self.batch_top_n_spin.value()))]
        self.batch_trial_distances.append(float(best_distance))

        if self._overlay_top_runs:
            self._draw_overlay_routes(self.route_ax_primary)
            self.route_canvas.draw_idle()

        self._draw_convergence()
        self.status_label.setText(
            f"Batch trial done | best: {float(best_distance):.4f} | top runs: {len(self.top_runs)}"
        )

    @QtCore.pyqtSlot(dict)
    def _on_batch_finished(self, payload: Dict[str, Any]) -> None:
        if self.top_runs:
            self.status_label.setText(
                f"Batch completed | best: {self.top_runs[0]['best_distance']:.4f} | runs: {len(self.top_runs)}"
            )
        else:
            self.status_label.setText("Batch completed")

    @QtCore.pyqtSlot(str)
    def _on_batch_failed(self, message: str) -> None:
        if message == STOP_EXCEPTION_TEXT:
            self.status_label.setText("Batch stopped by user.")
            return
        self.status_label.setText("Batch failed")
        QtWidgets.QMessageBox.critical(self, "Batch Error", message)

    def _cleanup_batch_thread(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
            self._batch_worker = None
        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
            self._batch_thread = None
        self._stop_requested_ui = False
        self._update_run_stop_buttons()

    def _draw_overlay_routes(self, axis) -> None:
        if self.current_cities is None or not self.top_runs:
            return

        cities = self.current_cities
        axis.clear()
        axis.scatter(cities[:, 0], cities[:, 1], c="tab:red", s=35, zorder=3)
        colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "tab:red", "tab:blue"]

        for index, run in enumerate(self.top_runs):
            route = run.get("best_route")
            if not route:
                continue
            closed_route = list(route) + [route[0]]
            ordered = cities[closed_route]
            axis.plot(
                ordered[:, 0],
                ordered[:, 1],
                color=colors[index % len(colors)],
                linewidth=1.2,
                alpha=max(0.15, 0.85 - 0.07 * index),
                zorder=1,
            )

        axis.set_title(f"Batch overlay top {len(self.top_runs)} routes")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid(alpha=0.3)
        axis.set_aspect("equal", adjustable="box")

    def _clear_outputs(self) -> None:
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        if not os.path.isdir(outputs_dir):
            QtWidgets.QMessageBox.information(self, "Clear Outputs", "No outputs folder found.")
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Outputs",
            f"Delete all files and folders inside {outputs_dir}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        try:
            for name in os.listdir(outputs_dir):
                path = os.path.join(outputs_dir, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            self.status_label.setText("Outputs cleared.")
        except Exception as err:
            QtWidgets.QMessageBox.critical(self, "Clear Outputs", str(err))

    def _reset_fields(self) -> None:
        self.backend_combo.setCurrentText(app_config.SOLVER_BACKEND)
        self.num_cities_spin.setValue(app_config.NUM_CITIES)
        self.pop_size_spin.setValue(app_config.POP_SIZE)
        self.generations_spin.setValue(app_config.GENERATIONS)
        self.mutation_spin.setValue(app_config.MUTATION_RATE)
        self.crossover_spin.setValue(app_config.CROSSOVER_RATE)
        self.elite_spin.setValue(app_config.ELITE_SIZE)
        self.tournament_spin.setValue(app_config.TOURNAMENT_SIZE)
        self.enable_bat_compare_check.setChecked(bool(app_config.ENABLE_BAT_COMPARISON))

        self.seed_check.setChecked(app_config.RANDOM_SEED is not None)
        self.seed_spin.setValue(app_config.RANDOM_SEED if app_config.RANDOM_SEED is not None else 0)

        self.simpleai_restarts_spin.setValue(app_config.SIMPLEAI_RESTARTS)
        self.simpleai_enable_2opt_check.setChecked(app_config.SIMPLEAI_ENABLE_2OPT)
        self.simpleai_2opt_passes_spin.setValue(app_config.SIMPLEAI_2OPT_MAX_PASSES)
        self.simpleai_fitness_power_spin.setValue(app_config.SIMPLEAI_FITNESS_POWER)
        self.simpleai_use_native_check.setChecked(app_config.SIMPLEAI_USE_NATIVE_GENETIC)
        self.simpleai_elitism_check.setChecked(app_config.SIMPLEAI_ENABLE_ELITISM)
        self.simpleai_diversity_spin.setValue(app_config.SIMPLEAI_DIVERSITY_RATE)
        self.simpleai_epsilon_spin.setValue(app_config.SIMPLEAI_EPSILON)
        self.animation_interval_spin.setValue(max(DEFAULT_ANIMATION_INTERVAL_MS, app_config.ANIMATION_INTERVAL_MS))
        self.animation_buffer_limit_spin.setValue(0)
        if hasattr(self, "city_seed_check"):
            self.city_seed_check.setChecked(False)
            self.city_seed_spin.setValue(42)
        self._sync_population_dependent_ranges()

    @QtCore.pyqtSlot(dict)
    def _on_progress(self, payload: Dict[str, Any]) -> None:
        if self._stop_requested_ui:
            return
        event = payload.get("event")
        if event == "init":
            cities = payload.get("cities")
            if isinstance(cities, np.ndarray):
                self.current_cities = cities
            self._comparison_enabled = bool(payload.get("comparison_enabled", False))
            self._draw_empty_route()
            self._draw_convergence()
            return

        if payload.get("best_route") is None or payload.get("best_distance") is None:
            return

        self._record_live_metrics(payload)
        self._enqueue_progress_frame(payload)
        self._update_run_stop_buttons()

    @QtCore.pyqtSlot(dict)
    def _on_finished(self, payload: Dict[str, Any]) -> None:
        self.current_cities = payload["cities"]
        self._comparison_enabled = bool(payload.get("comparison_enabled", False))
        self._final_result_payload = dict(payload)
        if self.frame_buffer:
            self.status_label.setText(
                self._format_runtime_status(
                    f"Solver finished; replaying {len(self.frame_buffer)} buffered frames"
                )
            )
            self._start_playback_timer()
        else:
            self._finalize_completed_run(payload)

        self._update_run_stop_buttons()

    @QtCore.pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self.frame_buffer.clear()
        self._final_result_payload = None
        self._waiting_for_frames = False
        self._stop_requested_ui = False
        self._show_convergence_overlay = True
        if self._animation_timer.isActive():
            self._animation_timer.stop()
        self.status_label.setText("Solver failed. See error details.")
        QtWidgets.QMessageBox.critical(self, "Solver Error", message)
        self._draw_convergence()
        self._update_run_stop_buttons()

    @QtCore.pyqtSlot()
    def _on_stopped(self) -> None:
        self.frame_buffer.clear()
        self._final_result_payload = None
        self._waiting_for_frames = False
        self._stop_requested_ui = False
        self._show_convergence_overlay = True
        if self._animation_timer.isActive():
            self._animation_timer.stop()
        self.status_label.setText("Solver stopped by user.")
        self._draw_convergence()
        self._update_run_stop_buttons()

    @QtCore.pyqtSlot()
    def _cleanup_worker_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

        self._stop_requested_ui = False

        if self._final_result_payload is not None and not self.frame_buffer:
            self._finalize_completed_run(self._final_result_payload)

        self._update_run_stop_buttons()
        self._stop_playback_timer_if_idle()

    def _enqueue_progress_frame(self, payload: Dict[str, Any]) -> None:
        frame = dict(payload)
        buffer_limit = int(self.animation_buffer_limit_spin.value())

        if buffer_limit > 0 and len(self.frame_buffer) >= buffer_limit:
            self.frame_buffer.popleft()
            self.dropped_frame_count += 1

        self.frame_buffer.append(frame)
        self._start_playback_timer()

    @QtCore.pyqtSlot()
    def _consume_buffered_frame(self) -> None:
        if self.frame_buffer:
            payload = self.frame_buffer.popleft()
            self._waiting_for_frames = False
            self._render_progress_frame(payload)

            if self._final_result_payload is not None and not self.frame_buffer and not self._is_solver_running():
                self._finalize_completed_run(self._final_result_payload)

            self._update_run_stop_buttons()
            return

        if self._final_result_payload is not None and not self._is_solver_running():
            self._finalize_completed_run(self._final_result_payload)
            self._update_run_stop_buttons()
            self._stop_playback_timer_if_idle()
            return

        if self._is_solver_running():
            if not self._waiting_for_frames:
                self.status_label.setText(self._format_runtime_status("Animation waiting for new frames"))
                self._waiting_for_frames = True
            return

        self._update_run_stop_buttons()
        self._stop_playback_timer_if_idle()

    def _render_progress_frame(self, payload: Dict[str, Any]) -> None:
        route = payload.get("best_route")
        distance = payload.get("best_distance")
        generation = int(payload.get("generation", 0))
        total_generations = int(payload.get("total_generations", 0))
        restart_index = int(payload.get("restart_index", 1))
        restart_count = int(payload.get("restart_count", 1))
        source = str(payload.get("source", "primary")).lower()
        series_name = str(payload.get("series_name", source)).lower()

        if route is None or distance is None:
            return

        distance_value = float(distance)

        if source == "bat":
            self.rendered_compare_count += 1
            self.compare_steps.append(self.rendered_compare_count)
            self.compare_distances.append(distance_value)

            if self.current_cities is not None and self._comparison_enabled:
                self._draw_route_on_axis(
                    axis=self.route_ax_compare,
                    route=list(route),
                    distance=distance_value,
                    title_prefix="BAT",
                    generation=generation,
                    total_generations=total_generations,
                    restart_index=restart_index,
                    restart_count=restart_count,
                    final=False,
                    line_color="tab:purple",
                )
        else:
            self.rendered_primary_count += 1
            self.primary_steps.append(self.rendered_primary_count)
            self.primary_distances.append(distance_value)

            if self.current_cities is not None:
                self._draw_route_on_axis(
                    axis=self.route_ax_primary,
                    route=list(route),
                    distance=distance_value,
                    title_prefix=series_name,
                    generation=generation,
                    total_generations=total_generations,
                    restart_index=restart_index,
                    restart_count=restart_count,
                    final=False,
                    line_color="tab:blue",
                )

        if not self._comparison_enabled:
            self._draw_route_axis_placeholder(
                axis=self.route_ax_compare,
                title="BAT Comparison",
                message="Comparison disabled",
            )

        self._draw_convergence()
        self.route_canvas.draw_idle()

        self.status_label.setText(
            self._format_runtime_status(
                (
                    f"Animating {series_name} | restart {restart_index}/{max(restart_count, 1)} | "
                    f"gen {generation}/{max(total_generations, 1)} | dist {float(distance):.4f}"
                )
            )
        )

        # If extended metrics present, append average/std/diversity info
        try:
            avg_f = payload.get('avg_fitness', None)
            div = payload.get('diversity', None)
            if avg_f is not None:
                self.status_label.setText(self.status_label.text() + f" | avg_fitness: {float(avg_f):.4f}")
            if div is not None:
                self.status_label.setText(self.status_label.text() + f" | diversity: {int(div)}")
        except Exception:
            pass

    def _finalize_completed_run(self, payload: Dict[str, Any]) -> None:
        self.current_cities = payload["cities"]
        backend = str(payload.get("backend", self._solver_params.get("backend", "primary"))).lower()

        runtime_total = payload.get("runtime_total")
        runtime_primary = payload.get("runtime_primary")
        runtime_bat = payload.get("runtime_bat")
        if runtime_total is None and self._run_start_time is not None:
            runtime_total = float(time.time() - self._run_start_time)

        def fmt_runtime(value: Optional[float]) -> str:
            if value is None:
                return "n/a"
            return f"{value:.3f}s"

        primary_result = payload.get("primary_result")
        if not isinstance(primary_result, dict):
            primary_result = {
                "label": backend,
                "best_route": list(payload["best_route"]),
                "best_distance": float(payload["best_distance"]),
                "best_distance_history": list(payload["best_distance_history"]),
                "best_route_history": [list(route) for route in payload["best_route_history"]],
                "initial_best_distance": float(payload["initial_best_distance"]),
            }

        comparison_result = payload.get("comparison_result")
        self._comparison_enabled = bool(payload.get("comparison_enabled", False)) and isinstance(
            comparison_result, dict
        )

        self.primary_distances = list(primary_result.get("best_distance_history", []))
        self.primary_steps = list(range(1, len(self.primary_distances) + 1))
        self.rendered_primary_count = len(self.primary_distances)

        self.compare_distances = []
        self.compare_steps = []
        self.rendered_compare_count = 0
        if self._comparison_enabled and isinstance(comparison_result, dict):
            self.compare_distances = list(comparison_result.get("best_distance_history", []))
            self.compare_steps = list(range(1, len(self.compare_distances) + 1))
            self.rendered_compare_count = len(self.compare_distances)

        self._draw_convergence()

        self._draw_route_on_axis(
            axis=self.route_ax_primary,
            route=list(primary_result["best_route"]),
            distance=float(primary_result["best_distance"]),
            title_prefix=str(primary_result.get("label", backend)),
            generation=len(self.primary_distances),
            total_generations=max(1, len(self.primary_distances)),
            restart_index=1,
            restart_count=1,
            final=True,
            line_color="tab:blue",
        )

        if self._comparison_enabled and isinstance(comparison_result, dict):
            self._draw_route_on_axis(
                axis=self.route_ax_compare,
                route=list(comparison_result["best_route"]),
                distance=float(comparison_result["best_distance"]),
                title_prefix="BAT",
                generation=len(self.compare_distances),
                total_generations=max(1, len(self.compare_distances)),
                restart_index=1,
                restart_count=1,
                final=True,
                line_color="tab:purple",
            )
        else:
            self._draw_route_axis_placeholder(
                axis=self.route_ax_compare,
                title="BAT Comparison",
                message="Comparison disabled",
            )

        self.route_canvas.draw_idle()

        primary_best = float(primary_result["best_distance"])
        primary_initial = float(primary_result["initial_best_distance"])
        primary_improvement = primary_initial - primary_best
        primary_improvement_pct = (primary_improvement / primary_initial * 100.0) if primary_initial > 0 else 0.0

        if self._comparison_enabled and isinstance(comparison_result, dict):
            bat_best = float(comparison_result["best_distance"])
            delta = bat_best - primary_best
            runtime_note = f" | Solve time: {fmt_runtime(runtime_total)}"
            if runtime_primary is not None or runtime_bat is not None:
                runtime_note += f" | primary: {fmt_runtime(runtime_primary)} | BAT: {fmt_runtime(runtime_bat)}"
            metrics_note = ""
            avg_fitness = self._latest_primary_metrics.get("avg_fitness")
            diversity = self._latest_primary_metrics.get("diversity")
            if avg_fitness is not None:
                metrics_note += f" | avg fitness: {float(avg_fitness):.4f}"
            else:
                best_fitness = (1.0 / primary_best) if primary_best > 0 else None
                if best_fitness is not None:
                    metrics_note += f" | fitness: {best_fitness:.4f}"
            if diversity is not None:
                metrics_note += f" | diversity: {int(diversity)}"
            self.status_label.setText(
                (
                    f"Completed | {backend}: {primary_best:.4f} | BAT: {bat_best:.4f} | "
                    f"BAT-primary delta: {delta:.4f}{runtime_note}{metrics_note} | Dropped: {self.dropped_frame_count}"
                )
            )
        else:
            runtime_note = f" | Solve time: {fmt_runtime(runtime_total)}"
            metrics_note = ""
            avg_fitness = self._latest_primary_metrics.get("avg_fitness")
            diversity = self._latest_primary_metrics.get("diversity")
            if avg_fitness is not None:
                metrics_note += f" | avg fitness: {float(avg_fitness):.4f}"
            else:
                best_fitness = (1.0 / primary_best) if primary_best > 0 else None
                if best_fitness is not None:
                    metrics_note += f" | fitness: {best_fitness:.4f}"
            if diversity is not None:
                metrics_note += f" | diversity: {int(diversity)}"
            self.status_label.setText(
                (
                    f"Completed | Best distance: {primary_best:.4f} | "
                    f"Improvement: {primary_improvement:.4f} ({primary_improvement_pct:.2f}%) | "
                    f"Rendered primary: {self.rendered_primary_count}{runtime_note}{metrics_note} | Dropped: {self.dropped_frame_count}"
                )
            )

        self._final_result_payload = None
        self._waiting_for_frames = False
        self._stop_playback_timer_if_idle()

        try:
            self._run_counter += 1
            run_label = f"Run {self._run_counter}"
            primary_history = list(primary_result.get("best_distance_history", []))
            primary_steps = list(range(1, len(primary_history) + 1))
            compare_history = []
            if self._comparison_enabled and isinstance(comparison_result, dict):
                compare_history = list(comparison_result.get("best_distance_history", []))

            run_record = {
                "label": run_label,
                "primary": {
                    "best_distance": primary_history,
                    "steps": primary_steps,
                    "avg_fitness": list(self._raw_primary_avg_fitness),
                    "avg_fitness_steps": list(self._raw_primary_avg_steps),
                    "diversity": list(self._raw_primary_diversity),
                    "diversity_steps": list(self._raw_primary_div_steps),
                },
                "compare": {
                    "best_distance": compare_history,
                    "steps": list(range(1, len(compare_history) + 1)),
                },
                "runtime_total": runtime_total,
                "runtime_primary": runtime_primary,
                "runtime_bat": runtime_bat,
            }

            self.run_history.append(run_record)
            self._show_convergence_overlay = True
            self._refresh_run_focus_combo()
            self._draw_convergence()
            self._update_metrics_plot()
        except Exception:
            pass

        # Export experiment artifacts for GUI-run experiments
        try:
            from utils.exporter import Exporter
            from utils.logger import setup_logger

            exporter = Exporter()
            # Experiment name reflect GUI run and basic params
            cfg_name = (
                f"gui_experiment_{self._solver_params.get('backend','run')}_pop{self._solver_params.get('pop_size',0)}"
            )
            folder_path = exporter.create_experiment_folder(cfg_name)
            # Setup per-experiment logger file
            setup_logger(folder_path)

            # Save config
            exporter.save_config(folder_path, dict(self._solver_params))

            # Build metrics from recorded primary distances (best-so-far history)
            metrics = []
            prim_distances = list(self.primary_distances)
            for i, d in enumerate(prim_distances, start=1):
                metric = {
                    'generation': i,
                    'best_fitness': (1.0 / d) if d and d > 0 else None,
                    'avg_fitness': None,
                    'worst_fitness': None,
                }
                metrics.append(metric)

            exporter.save_metrics(folder_path, metrics)

            # Best solution
            best_distance = float(primary_result.get('best_distance', float('inf')))
            best_route = list(primary_result.get('best_route', []))
            # convergence generation: index of minimum in best_distance_history
            conv_gen = 0
            try:
                history = list(primary_result.get('best_distance_history', []))
                if history:
                    conv_gen = int(np.argmin(history)) + 1
            except Exception:
                conv_gen = 0

            exporter.save_best_solution(folder_path, best_route, best_distance, conv_gen)

            # Summary
            runtime = None
            try:
                runtime = payload.get("runtime_total")
                if runtime is None and self._run_start_time is not None:
                    runtime = float(time.time() - self._run_start_time)
            except Exception:
                runtime = None
            exporter.save_summary(
                folder_path,
                best_distance,
                len(primary_result.get('best_distance_history', [])),
                conv_gen,
                runtime if runtime is not None else 0.0,
            )

            # Population snapshot: save best route history as a lightweight snapshot
            try:
                snapshot = list(primary_result.get('best_route_history', []))
                exporter.save_population_snapshot(folder_path, snapshot, len(snapshot))
            except Exception:
                # best_route_history may not be present; ignore
                pass

        except Exception:
            # Do not interrupt UI on export failure; log to status
            self.status_label.setText("Export failed: see console for details")

    def _draw_empty_route(self) -> None:
        backend_label = str(self._solver_params.get("backend", "backend")).lower()
        self._draw_route_axis_placeholder(
            axis=self.route_ax_primary,
            title=f"{backend_label} route",
            message="Run solver to stream live frames",
        )
        if self._comparison_enabled:
            self._draw_route_axis_placeholder(
                axis=self.route_ax_compare,
                title="BAT comparison route",
                message="BAT stream will appear here",
            )
        else:
            self._draw_route_axis_placeholder(
                axis=self.route_ax_compare,
                title="BAT comparison route",
                message="Comparison disabled",
            )
        self.route_canvas.draw_idle()

    def _draw_empty_convergence(self) -> None:
        self.conv_ax.clear()
        self.conv_ax.set_title("Convergence comparison")
        self.conv_ax.set_xlabel("Step")
        self.conv_ax.set_ylabel("Best Distance")
        self.conv_ax.grid(alpha=0.3)
        self.conv_canvas.draw_idle()

    def _draw_route_axis_placeholder(self, axis, title: str, message: str) -> None:
        axis.clear()
        axis.set_title(title)
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid(alpha=0.3)

    def _draw_route_on_axis(
        self,
        axis,
        route: List[int],
        distance: float,
        title_prefix: str,
        generation: int,
        total_generations: int,
        restart_index: int,
        restart_count: int,
        line_color: str,
        final: bool = False,
    ) -> None:
        if self.current_cities is None or not route:
            return

        axis.clear()
        cities = self.current_cities
        axis.scatter(cities[:, 0], cities[:, 1], c="tab:red", s=35, zorder=3)

        for city_idx, (x_coord, y_coord) in enumerate(cities):
            axis.text(x_coord + 0.8, y_coord + 0.8, str(city_idx), fontsize=7)

        closed_route = list(route) + [route[0]]
        ordered = cities[closed_route]
        axis.plot(
            ordered[:, 0],
            ordered[:, 1],
            color=line_color,
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=2,
        )

        if final:
            title = f"{title_prefix} final | Distance: {distance:.4f}"
        else:
            title = (
                f"{title_prefix} | Restart {restart_index}/{restart_count} | "
                f"Gen {generation}/{max(total_generations, 1)} | Distance: {distance:.4f}"
            )
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid(alpha=0.3)
        axis.set_aspect("equal", adjustable="box")

    def _draw_convergence(self) -> None:
        self.conv_ax.clear()

        metric = (
            self.convergence_metric_combo.currentText()
            if hasattr(self, "convergence_metric_combo")
            else "Best distance"
        )
        focus_index = self._get_focus_index()

        def series_for(section: Dict[str, Any], metric_name: str) -> Optional[Dict[str, List[float]]]:
            if not section:
                return None

            if metric_name == "Best distance":
                values = section.get("best_distance") or []
                steps = section.get("steps") or list(range(1, len(values) + 1))
                return {"steps": steps, "values": values}

            if metric_name == "Avg fitness":
                values = section.get("avg_fitness") or []
                steps = section.get("avg_fitness_steps") or list(range(1, len(values) + 1))
                return {"steps": steps, "values": values}

            if metric_name == "Diversity":
                values = section.get("diversity") or []
                steps = section.get("diversity_steps") or list(range(1, len(values) + 1))
                return {"steps": steps, "values": values}

            if metric_name == "Convergence speed":
                distances = section.get("best_distance") or []
                if len(distances) < 2:
                    return None
                speed = [float(prev) - float(curr) for prev, curr in zip(distances[:-1], distances[1:])]
                steps = list(range(2, len(distances) + 1))
                return {"steps": steps, "values": speed}

            return None

        runs: List[Dict[str, Any]] = []

        def slice_series(values: List[float], steps: List[int], count: int) -> Dict[str, List[float]]:
            if not values:
                return {"values": [], "steps": []}
            limit = min(len(values), max(0, int(count)))
            if limit <= 0:
                return {"values": [], "steps": []}
            if steps:
                return {"values": list(values[:limit]), "steps": list(steps[:limit])}
            return {"values": list(values[:limit]), "steps": list(range(1, limit + 1))}

        if self._show_convergence_overlay:
            runs = list(self.run_history)
        else:
            primary_slice = slice_series(self._raw_primary_distances, self._raw_primary_steps, self.rendered_primary_count)
            compare_slice = slice_series(self._raw_compare_distances, self._raw_compare_steps, self.rendered_compare_count)
            avg_slice = slice_series(self._raw_primary_avg_fitness, self._raw_primary_avg_steps, self.rendered_primary_count)
            div_slice = slice_series(self._raw_primary_diversity, self._raw_primary_div_steps, self.rendered_primary_count)
            avg_compare_slice = slice_series(self._raw_compare_avg_fitness, self._raw_compare_avg_steps, self.rendered_compare_count)
            div_compare_slice = slice_series(self._raw_compare_diversity, self._raw_compare_div_steps, self.rendered_compare_count)

            current_primary = {
                "best_distance": primary_slice["values"],
                "steps": primary_slice["steps"],
                "avg_fitness": avg_slice["values"],
                "avg_fitness_steps": avg_slice["steps"],
                "diversity": div_slice["values"],
                "diversity_steps": div_slice["steps"],
            }
            current_compare = {
                "best_distance": compare_slice["values"],
                "steps": compare_slice["steps"],
                "avg_fitness": avg_compare_slice["values"],
                "avg_fitness_steps": avg_compare_slice["steps"],
                "diversity": div_compare_slice["values"],
                "diversity_steps": div_compare_slice["steps"],
            }

            if current_primary["best_distance"]:
                runs.append({"label": "Current", "primary": current_primary, "compare": current_compare, "temp": True})

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
        show_legend = focus_index is not None or len(runs) <= 3

        for idx, run in enumerate(runs):
            if run.get("temp") and focus_index is not None:
                continue

            alpha = 1.0 if focus_index is None or idx == focus_index or run.get("temp") else 0.2
            linewidth = 2.2 if focus_index is None or idx == focus_index or run.get("temp") else 1.2
            color = colors[idx % len(colors)]

            primary_series = series_for(run.get("primary", {}), metric)
            if primary_series:
                self.conv_ax.plot(
                    primary_series["steps"],
                    primary_series["values"],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=f"{run.get('label', f'Run {idx + 1}')} primary",
                )

            compare_series = series_for(run.get("compare", {}), metric)
            if compare_series:
                self.conv_ax.plot(
                    compare_series["steps"],
                    compare_series["values"],
                    color=color,
                    linewidth=linewidth,
                    linestyle="--",
                    alpha=alpha,
                    label=f"{run.get('label', f'Run {idx + 1}')} bat",
                )

        if (
            self._show_convergence_overlay
            and metric == "Best distance"
            and self.batch_convergence_mode != "off"
            and self.batch_trial_distances
        ):
            trial_steps = list(range(1, len(self.batch_trial_distances) + 1))
            if self.batch_convergence_mode == "running":
                batch_curve = []
                running_best = float("inf")
                for value in self.batch_trial_distances:
                    running_best = min(running_best, float(value))
                    batch_curve.append(running_best)
                label = "batch running best"
            else:
                batch_curve = [float(value) for value in self.batch_trial_distances]
                label = "batch trial best"

            self.conv_ax.plot(
                trial_steps,
                batch_curve,
                color="tab:olive",
                linewidth=2,
                marker="o",
                label=label,
            )

        title_map = {
            "Best distance": "Best-so-far distance",
            "Avg fitness": "Average fitness",
            "Diversity": "Population diversity",
            "Convergence speed": "Convergence speed",
        }
        ylabel_map = {
            "Best distance": "Distance",
            "Avg fitness": "Avg fitness",
            "Diversity": "Diversity",
            "Convergence speed": "Improvement",
        }

        self.conv_ax.set_title(title_map.get(metric, "Convergence"))
        self.conv_ax.set_xlabel("Step")
        self.conv_ax.set_ylabel(ylabel_map.get(metric, "Value"))
        self.conv_ax.grid(alpha=0.3)
        if show_legend:
            handles, labels = self.conv_ax.get_legend_handles_labels()
            if handles and labels:
                self.conv_ax.legend(loc="upper right")
        self.conv_canvas.draw_idle()


def launch_gui() -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = TSPControlPanel()
    window.show()

    if owns_app:
        return int(app.exec_())
    return 0


def main() -> None:
    raise SystemExit(launch_gui())


if __name__ == "__main__":
    main()
