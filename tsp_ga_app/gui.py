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
            self.progress.emit({"event": "init", "backend": backend, "cities": cities})

            if backend == "simpleai":
                simpleai_overrides = self._apply_simpleai_runtime_overrides()
                solver = genetic_algorithm_simpleai
            elif backend == "custom":
                solver = genetic_algorithm_custom
            else:
                raise ValueError("Backend must be either 'custom' or 'simpleai'.")

            def progress_callback(payload: Dict[str, Any]) -> None:
                if self._stop_requested:
                    raise RuntimeError(STOP_EXCEPTION_TEXT)
                self.progress.emit(dict(payload))

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
                progress_callback=progress_callback,
            )

            if self._stop_requested:
                self.stopped.emit()
                return

            self.finished.emit(
                {
                    "cities": cities,
                    "best_route": best_route,
                    "best_distance": float(best_distance),
                    "best_distance_history": list(best_distance_history),
                    "best_route_history": [list(route) for route in best_route_history],
                    "initial_best_distance": float(initial_best_distance),
                    "backend": backend,
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
        self.progress_steps: List[int] = []
        self.progress_distances: List[float] = []
        self.frame_buffer: Deque[Dict[str, Any]] = deque()
        self.dropped_frame_count = 0
        self.rendered_frame_count = 0
        self._waiting_for_frames = False
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

        general_form.addRow("Backend", self.backend_combo)
        general_form.addRow("Cities", self.num_cities_spin)
        general_form.addRow("Population", self.pop_size_spin)
        general_form.addRow("Generations", self.generations_spin)
        general_form.addRow("Mutation rate", self.mutation_spin)
        general_form.addRow("Crossover rate", self.crossover_spin)
        general_form.addRow("Elite size", self.elite_spin)
        general_form.addRow("Tournament size", self.tournament_spin)
        general_form.addRow(self.seed_check, self.seed_spin)

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

        route_group = QtWidgets.QGroupBox("Live route")
        route_layout = QtWidgets.QVBoxLayout(route_group)
        self.route_figure = Figure(figsize=(8, 5), tight_layout=True)
        self.route_canvas = FigureCanvas(self.route_figure)
        self.route_ax = self.route_figure.add_subplot(111)
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
            f"rendered: {self.rendered_frame_count} | dropped: {self.dropped_frame_count}"
        )

    def _reset_live_state(self) -> None:
        self.frame_buffer.clear()
        self._final_result_payload = None
        self.progress_steps = []
        self.progress_distances = []
        self.dropped_frame_count = 0
        self.rendered_frame_count = 0
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
            return
        self._enqueue_progress_frame(payload)
        self._update_run_stop_buttons()

    @QtCore.pyqtSlot(dict)
    def _on_finished(self, payload: Dict[str, Any]) -> None:
        self.current_cities = payload["cities"]
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

        if route is None or distance is None:
            return

        self.rendered_frame_count += 1
        self.progress_steps.append(self.rendered_frame_count)
        self.progress_distances.append(float(distance))

        if self.current_cities is not None:
            self._draw_route(
                route=list(route),
                distance=float(distance),
                generation=generation,
                total_generations=total_generations,
                restart_index=restart_index,
                restart_count=restart_count,
            )
        self._draw_convergence(self.progress_steps, self.progress_distances)

        self.status_label.setText(
            self._format_runtime_status(
                (
                    f"Animating restart {restart_index}/{max(restart_count, 1)} | "
                    f"gen {generation}/{max(total_generations, 1)} | dist {float(distance):.4f}"
                )
            )
        )

    def _finalize_completed_run(self, payload: Dict[str, Any]) -> None:
        self.current_cities = payload["cities"]

        best_route = list(payload["best_route"])
        best_distance = float(payload["best_distance"])
        best_distance_history = list(payload["best_distance_history"])
        initial_best_distance = float(payload["initial_best_distance"])

        if best_distance_history:
            self._draw_convergence(list(range(1, len(best_distance_history) + 1)), best_distance_history)

        self._draw_route(
            route=best_route,
            distance=best_distance,
            generation=len(best_distance_history),
            total_generations=max(1, len(best_distance_history)),
            restart_index=1,
            restart_count=1,
            final=True,
        )

        improvement = initial_best_distance - best_distance
        improvement_pct = (improvement / initial_best_distance * 100.0) if initial_best_distance > 0 else 0.0
        self.status_label.setText(
            (
                f"Completed | Best distance: {best_distance:.4f} | "
                f"Improvement: {improvement:.4f} ({improvement_pct:.2f}%) | "
                f"Rendered: {self.rendered_frame_count} | Dropped: {self.dropped_frame_count}"
            )
        )

        self._final_result_payload = None
        self._waiting_for_frames = False
        self._stop_playback_timer_if_idle()

    def _draw_empty_route(self) -> None:
        self.route_ax.clear()
        self.route_ax.set_title("Run the solver to see route updates")
        self.route_ax.set_xlabel("X")
        self.route_ax.set_ylabel("Y")
        self.route_ax.grid(alpha=0.3)
        self.route_canvas.draw_idle()

    def _draw_empty_convergence(self) -> None:
        self.conv_ax.clear()
        self.conv_ax.set_title("Convergence")
        self.conv_ax.set_xlabel("Step")
        self.conv_ax.set_ylabel("Best Distance")
        self.conv_ax.grid(alpha=0.3)
        self.conv_canvas.draw_idle()

    def _draw_route(
        self,
        route: List[int],
        distance: float,
        generation: int,
        total_generations: int,
        restart_index: int,
        restart_count: int,
        final: bool = False,
    ) -> None:
        if self.current_cities is None or not route:
            return

        self.route_ax.clear()
        cities = self.current_cities
        self.route_ax.scatter(cities[:, 0], cities[:, 1], c="tab:red", s=35, zorder=3)

        for city_idx, (x_coord, y_coord) in enumerate(cities):
            self.route_ax.text(x_coord + 0.8, y_coord + 0.8, str(city_idx), fontsize=7)

        closed_route = list(route) + [route[0]]
        ordered = cities[closed_route]
        self.route_ax.plot(
            ordered[:, 0],
            ordered[:, 1],
            color="tab:blue",
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=2,
        )

        if final:
            title = f"Final route | Distance: {distance:.4f}"
        else:
            title = (
                f"Restart {restart_index}/{restart_count} | "
                f"Gen {generation}/{max(total_generations, 1)} | Distance: {distance:.4f}"
            )
        self.route_ax.set_title(title)
        self.route_ax.set_xlabel("X")
        self.route_ax.set_ylabel("Y")
        self.route_ax.grid(alpha=0.3)
        self.route_ax.set_aspect("equal", adjustable="box")
        self.route_canvas.draw_idle()

    def _draw_convergence(self, steps: List[int], distances: List[float]) -> None:
        self.conv_ax.clear()
        self.conv_ax.plot(steps, distances, color="tab:green", linewidth=2)
        if distances:
            self.conv_ax.scatter([steps[-1]], [distances[-1]], color="tab:orange", zorder=3)
        self.conv_ax.set_title("Best-so-far distance")
        self.conv_ax.set_xlabel("Step")
        self.conv_ax.set_ylabel("Distance")
        self.conv_ax.grid(alpha=0.3)
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
