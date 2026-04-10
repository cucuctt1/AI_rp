from collections import deque
import random
import sys
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
        try:
            seed = self.params["seed"]
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

            cities = generate_cities(int(self.params["num_cities"]))
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

            if backend == "simpleai":
                simpleai_overrides = self._apply_simpleai_runtime_overrides()
                solver = genetic_algorithm_simpleai
            elif backend == "custom":
                solver = genetic_algorithm_custom
            else:
                raise ValueError("Backend must be either 'custom' or 'simpleai'.")

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

            primary_result = self._pack_solver_result(
                label=backend,
                best_route=best_route,
                best_distance=best_distance,
                best_distance_history=best_distance_history,
                best_route_history=best_route_history,
                initial_best_distance=initial_best_distance,
            )

            if self._stop_requested:
                self.stopped.emit()
                return

            comparison_result: Optional[Dict[str, Any]] = None
            if compare_enabled:
                if seed is not None:
                    random.seed(int(seed) + 101)
                    np.random.seed(int(seed) + 101)

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
                )

                comparison_result = self._pack_solver_result(
                    label="bat",
                    best_route=bat_best_route,
                    best_distance=bat_best_distance,
                    best_distance_history=bat_distance_history,
                    best_route_history=bat_route_history,
                    initial_best_distance=bat_initial_best_distance,
                )

            if self._stop_requested:
                self.stopped.emit()
                return

            self.finished.emit(
                {
                    "cities": cities,
                    "backend": backend,
                    "comparison_enabled": compare_enabled,
                    "primary_result": primary_result,
                    "comparison_result": comparison_result,
                    # Compatibility keys kept for existing non-comparison code paths.
                    "best_route": list(primary_result["best_route"]),
                    "best_distance": float(primary_result["best_distance"]),
                    "best_distance_history": list(primary_result["best_distance_history"]),
                    "best_route_history": [list(route) for route in primary_result["best_route_history"]],
                    "initial_best_distance": float(primary_result["initial_best_distance"]),
                }
            )
        except RuntimeError as err:
            if str(err) == STOP_EXCEPTION_TEXT:
                self.stopped.emit()
            else:
                self.failed.emit(traceback.format_exc())
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self._restore_simpleai_runtime_overrides(simpleai_overrides)

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


class TSPControlPanel(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TSP Genetic Solver Studio")
        self.resize(1280, 820)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[SolverWorker] = None

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

        controls.setMaximumWidth(380)
        layout.addWidget(controls)
        layout.addWidget(plots, stretch=1)

    def _build_controls_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setSpacing(8)

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

    def _is_solver_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _is_busy(self) -> bool:
        return self._is_solver_running() or bool(self.frame_buffer) or self._final_result_payload is not None

    def _update_run_stop_buttons(self) -> None:
        self.run_button.setEnabled(not self._is_busy())
        self.stop_button.setEnabled(self._is_solver_running())

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
        self.dropped_frame_count = 0
        self.rendered_primary_count = 0
        self.rendered_compare_count = 0
        self._waiting_for_frames = False

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

        params = self._collect_params()
        self._solver_params = dict(params)
        self._comparison_enabled = bool(params.get("enable_bat_comparison", False))
        self.current_cities = None
        self._reset_live_state()
        self._draw_empty_route()
        self._draw_empty_convergence()
        self._start_playback_timer()

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
        self.status_label.setText(self._format_runtime_status("Running solver + live animation"))
        self._thread.start()

    def _stop_solver(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.status_label.setText("Stop requested. Waiting for current iteration...")

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
        self._sync_population_dependent_ranges()

    @QtCore.pyqtSlot(dict)
    def _on_progress(self, payload: Dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "init":
            cities = payload.get("cities")
            if isinstance(cities, np.ndarray):
                self.current_cities = cities
            self._comparison_enabled = bool(payload.get("comparison_enabled", False))
            self._draw_empty_route()
            self._draw_empty_convergence()
            return

        if payload.get("best_route") is None or payload.get("best_distance") is None:
            return

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
        if self._animation_timer.isActive():
            self._animation_timer.stop()
        self.status_label.setText("Solver failed. See error details.")
        QtWidgets.QMessageBox.critical(self, "Solver Error", message)
        self._update_run_stop_buttons()

    @QtCore.pyqtSlot()
    def _on_stopped(self) -> None:
        self.frame_buffer.clear()
        self._final_result_payload = None
        self._waiting_for_frames = False
        if self._animation_timer.isActive():
            self._animation_timer.stop()
        self.status_label.setText("Solver stopped by user.")
        self._update_run_stop_buttons()

    @QtCore.pyqtSlot()
    def _cleanup_worker_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

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

    def _finalize_completed_run(self, payload: Dict[str, Any]) -> None:
        self.current_cities = payload["cities"]
        backend = str(payload.get("backend", self._solver_params.get("backend", "primary"))).lower()

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
            self.status_label.setText(
                (
                    f"Completed | {backend}: {primary_best:.4f} | BAT: {bat_best:.4f} | "
                    f"BAT-primary delta: {delta:.4f} | Dropped: {self.dropped_frame_count}"
                )
            )
        else:
            self.status_label.setText(
                (
                    f"Completed | Best distance: {primary_best:.4f} | "
                    f"Improvement: {primary_improvement:.4f} ({primary_improvement_pct:.2f}%) | "
                    f"Rendered primary: {self.rendered_primary_count} | Dropped: {self.dropped_frame_count}"
                )
            )

        self._final_result_payload = None
        self._waiting_for_frames = False
        self._stop_playback_timer_if_idle()

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

        primary_label = str(self._solver_params.get("backend", "primary")).lower()
        if self.primary_distances:
            self.conv_ax.plot(
                self.primary_steps,
                self.primary_distances,
                color="tab:green",
                linewidth=2,
                label=f"{primary_label}",
            )
            self.conv_ax.scatter(
                [self.primary_steps[-1]],
                [self.primary_distances[-1]],
                color="tab:green",
                zorder=3,
            )

        if self._comparison_enabled and self.compare_distances:
            self.conv_ax.plot(
                self.compare_steps,
                self.compare_distances,
                color="tab:purple",
                linewidth=2,
                linestyle="--",
                label="bat",
            )
            self.conv_ax.scatter(
                [self.compare_steps[-1]],
                [self.compare_distances[-1]],
                color="tab:purple",
                zorder=3,
            )

        self.conv_ax.set_title("Best-so-far distance (comparison)")
        self.conv_ax.set_xlabel("Step")
        self.conv_ax.set_ylabel("Distance")
        self.conv_ax.grid(alpha=0.3)
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
