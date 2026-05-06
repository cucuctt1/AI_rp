import sys
import json
import random
import time
import traceback
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from core.config import DEFAULT_CONFIG
from core.ga_engine import GAEngine

STOP_EXCEPTION_TEXT = "Solver stopped by user."
DEFAULT_ANIMATION_INTERVAL_MS = 120

def compute_distance_matrix(cities: np.ndarray) -> np.ndarray:
    deltas = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))

def generate_cities(n: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 100, size=(n, 2))

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
        try:
            seed = self.params.get("seed")
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

            num_cities = int(self.params.get("num_cities", 50))
            cities = generate_cities(num_cities, seed)
            dist_matrix = compute_distance_matrix(cities)

            self.progress.emit({
                "event": "init",
                "cities": cities,
                "num_cities": num_cities,
            })

            config = {
                'population_size': int(self.params.get('population_size', DEFAULT_CONFIG['population_size'])),
                'generations': int(self.params.get('generations', DEFAULT_CONFIG['generations'])),
                'mutation_rate': float(self.params.get('mutation_rate', DEFAULT_CONFIG['mutation_rate'])),
                'crossover_type': str(self.params.get('crossover_type', DEFAULT_CONFIG['crossover_type'])),
                'mutation_type': str(self.params.get('mutation_type', DEFAULT_CONFIG['mutation_type'])),
                'selection_type': str(self.params.get('selection_type', DEFAULT_CONFIG['selection_type'])),
                'adaptive_mutation': bool(self.params.get('adaptive_mutation', DEFAULT_CONFIG['adaptive_mutation'])),
                'elitism_k': int(self.params.get('elitism_k', DEFAULT_CONFIG['elitism_k'])),
                'local_search_freq': int(self.params.get('local_search_freq', DEFAULT_CONFIG['local_search_freq'])),
            }

            frame_count = 0
            def progress_callback(state):
                if self._stop_requested:
                    raise InterruptedError(STOP_EXCEPTION_TEXT)
                # Only emit every 5 generations to avoid overwhelming Qt
                nonlocal frame_count
                frame_count += 1
                if frame_count % 5 == 0 or state['generation'] == config['generations']:
                    self.progress.emit({
                        "event": "progress",
                        "generation": state['generation'],
                        "best_distance": state['global_best_distance'],
                        "best_route": state['global_best_route'],
                    })
                time.sleep(0.001)  # Yield GIL

            engine = GAEngine(config, dist_matrix)
            results = engine.run(callback=progress_callback)

            if self._stop_requested:
                self.stopped.emit()
                return

            self.finished.emit({
                "event": "complete",
                "best_route": results['best_route'],
                "best_distance": results['best_distance'],
                "convergence_gen": results['convergence_gen'],
                "metrics": results['metrics'],
                "runtime": results['runtime'],
                "cities": cities,
            })

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            self.failed.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

    def request_stop(self) -> None:
        self._stop_requested = True

class TSPControlPanel(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TSP Genetic Algorithm Solver")
        self.resize(1280, 820)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[SolverWorker] = None

        self.current_cities: Optional[np.ndarray] = None
        self.steps: List[int] = []
        self.distances: List[float] = []
        self.frame_buffer: Deque[Dict[str, Any]] = deque()
        self.dropped_frame_count = 0
        self.rendered_count = 0
        self._waiting_for_frames = False
        self._solver_params: Dict[str, Any] = {}
        self._final_result_payload: Optional[Dict[str, Any]] = None

        self._animation_timer = QtCore.QTimer(self)
        self._animation_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._animation_timer.timeout.connect(self._consume_buffered_frame)

        self.loaded_datasets = {}
        self._build_ui()

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

        # Data Loading Group
        data_group = QtWidgets.QGroupBox("Custom JSON Data")
        data_form = QtWidgets.QVBoxLayout(data_group)
        
        load_data_btn = QtWidgets.QPushButton("Load JSON File...")
        load_data_btn.clicked.connect(self._load_json_data)
        data_form.addWidget(load_data_btn)
        
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItem("Random Cities (default)")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        data_form.addWidget(self.dataset_combo)
        
        data_group.setLayout(data_form)
        vbox.addWidget(data_group)

        # General Group
        general_group = QtWidgets.QGroupBox("General")
        general_form = QtWidgets.QFormLayout(general_group)

        self.num_cities_spin = QtWidgets.QSpinBox()
        self.num_cities_spin.setRange(5, 500)
        self.num_cities_spin.setValue(50)

        self.pop_size_spin = QtWidgets.QSpinBox()
        self.pop_size_spin.setRange(2, 3000)
        self.pop_size_spin.setValue(DEFAULT_CONFIG['population_size'])

        self.generations_spin = QtWidgets.QSpinBox()
        self.generations_spin.setRange(1, 50000)
        self.generations_spin.setValue(DEFAULT_CONFIG['generations'])

        self.mutation_spin = QtWidgets.QDoubleSpinBox()
        self.mutation_spin.setRange(0.0, 1.0)
        self.mutation_spin.setSingleStep(0.01)
        self.mutation_spin.setDecimals(3)
        self.mutation_spin.setValue(DEFAULT_CONFIG['mutation_rate'])

        self.crossover_combo = QtWidgets.QComboBox()
        self.crossover_combo.addItems(["pmx", "order"])
        self.crossover_combo.setCurrentText(DEFAULT_CONFIG['crossover_type'])

        self.mutation_type_combo = QtWidgets.QComboBox()
        self.mutation_type_combo.addItems(["swap", "reverse", "scramble"])
        self.mutation_type_combo.setCurrentText(DEFAULT_CONFIG['mutation_type'])

        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.addItems(["tournament", "roulette"])
        self.selection_combo.setCurrentText(DEFAULT_CONFIG['selection_type'])

        self.elitism_spin = QtWidgets.QSpinBox()
        self.elitism_spin.setRange(0, 100)
        self.elitism_spin.setValue(DEFAULT_CONFIG['elitism_k'])

        self.adaptive_check = QtWidgets.QCheckBox()
        self.adaptive_check.setChecked(DEFAULT_CONFIG['adaptive_mutation'])

        general_form.addRow("Num Cities:", self.num_cities_spin)
        general_form.addRow("Population:", self.pop_size_spin)
        general_form.addRow("Generations:", self.generations_spin)
        general_form.addRow("Mutation Rate:", self.mutation_spin)
        general_form.addRow("Crossover:", self.crossover_combo)
        general_form.addRow("Mutation Type:", self.mutation_type_combo)
        general_form.addRow("Selection:", self.selection_combo)
        general_form.addRow("Elitism K:", self.elitism_spin)
        general_form.addRow("Adaptive Mut:", self.adaptive_check)

        vbox.addWidget(general_group)

        # Playback Group
        playback_group = QtWidgets.QGroupBox("Animation & Progress")
        playback_form = QtWidgets.QFormLayout(playback_group)

        self.animation_interval_spin = QtWidgets.QSpinBox()
        self.animation_interval_spin.setRange(10, 2000)
        self.animation_interval_spin.setValue(DEFAULT_ANIMATION_INTERVAL_MS)
        self.animation_interval_spin.setSuffix(" ms")
        self.animation_interval_spin.valueChanged.connect(self._update_animation_interval)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)

        playback_form.addRow("Frame Interval:", self.animation_interval_spin)
        playback_form.addRow("Progress:", self.progress_bar)

        vbox.addWidget(playback_group)

        # Buttons
        buttons_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run Single")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.reset_button = QtWidgets.QPushButton("Reset Graph")

        buttons_row.addWidget(self.run_button)
        buttons_row.addWidget(self.stop_button)
        buttons_row.addWidget(self.reset_button)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)

        self.run_button.clicked.connect(self._start_solver)
        self.stop_button.clicked.connect(self._stop_solver)
        self.reset_button.clicked.connect(self._reset_graph)

        vbox.addLayout(buttons_row)
        vbox.addWidget(self.status_label)
        vbox.addStretch(1)

        return panel

    def _build_plots_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setSpacing(8)

        route_group = QtWidgets.QGroupBox("Live Route")
        route_layout = QtWidgets.QVBoxLayout(route_group)
        self.route_figure = Figure(figsize=(10, 5), tight_layout=True)
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

    def _draw_empty_route(self) -> None:
        self.route_ax.clear()
        if self.current_cities is not None:
            self.route_ax.scatter(self.current_cities[:, 0], self.current_cities[:, 1], c='blue', s=30)
        self.route_ax.set_title("Best Route")
        self.route_ax.set_aspect('equal')
        self.route_canvas.draw()

    def _draw_empty_convergence(self) -> None:
        self.conv_ax.clear()
        self.conv_ax.set_title("Best Distance over Generations")
        self.conv_ax.set_xlabel("Generation")
        self.conv_ax.set_ylabel("Distance")
        self.conv_canvas.draw()

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

    def _reset_live_state(self) -> None:
        self.frame_buffer.clear()
        self._final_result_payload = None
        self.steps = []
        self.distances = []
        self.rendered_count = 0
        self.dropped_frame_count = 0

    def _load_json_data(self) -> None:
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON", "", "JSON Files (*.json)")
        if not fname:
            return
        try:
            with open(fname, "r") as f:
                data = json.load(f)
            for key, val in data.items():
                points = []
                if isinstance(val, dict):
                    for k in sorted(val.keys(), key=lambda x: int(x) if x.isdigit() else x):
                        points.append(val[k])
                elif isinstance(val, list):
                    points = val
                if len(points) > 0:
                    arr = np.array(points, dtype=float)
                    if arr.shape[1] >= 2:
                        name = f"{key} ({len(points)} cities)"
                        self.loaded_datasets[name] = arr[:, :2]
                        self.dataset_combo.addItem(name)
            self.status_label.setText(f"Loaded {len(self.loaded_datasets)} datasets from JSON")
        except Exception as e:
            self.status_label.setText(f"Error loading JSON: {e}")

    def _on_dataset_changed(self) -> None:
        curr = self.dataset_combo.currentText()
        if curr in self.loaded_datasets:
            self.current_cities = self.loaded_datasets[curr]
        else:
            self.current_cities = generate_cities(int(self.num_cities_spin.value()))
        self._draw_empty_route()

    def _start_solver(self) -> None:
        self._reset_live_state()
        self._update_run_stop_buttons()

        if self.current_cities is None:
            self.current_cities = generate_cities(int(self.num_cities_spin.value()))

        self._solver_params = {
            "seed": 42,
            "num_cities": len(self.current_cities),
            "population_size": int(self.pop_size_spin.value()),
            "generations": int(self.generations_spin.value()),
            "mutation_rate": float(self.mutation_spin.value()),
            "crossover_type": self.crossover_combo.currentText(),
            "mutation_type": self.mutation_type_combo.currentText(),
            "selection_type": self.selection_combo.currentText(),
            "adaptive_mutation": self.adaptive_check.isChecked(),
            "elitism_k": int(self.elitism_spin.value()),
            "local_search_freq": 0,
        }

        self.status_label.setText("Running GA Experiment...")
        self.progress_bar.setMaximum(int(self.generations_spin.value()))
        self.progress_bar.setValue(0)

        self._thread = QtCore.QThread(self)
        self._worker = SolverWorker(self._solver_params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.stopped.connect(self._on_worker_stopped)

        self._thread.start()
        self._start_playback_timer()

    def _stop_solver(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self._thread.quit()
            self._thread.wait()
            self.status_label.setText("Stopped by user")
        self._update_run_stop_buttons()

    def _reset_graph(self) -> None:
        self._reset_live_state()
        self._draw_empty_route()
        self._draw_empty_convergence()
        self.progress_bar.setValue(0)
        self.status_label.setText("Graph reset")

    @QtCore.pyqtSlot(dict)
    def _on_worker_progress(self, payload: Dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "init":
            self.current_cities = payload["cities"]
            self._draw_empty_route()
        elif event == "progress":
            self.frame_buffer.append(payload)

    @QtCore.pyqtSlot(dict)
    def _on_worker_finished(self, payload: Dict[str, Any]) -> None:
        self._final_result_payload = payload
        self.status_label.setText(f"Finished. Best distance: {payload['best_distance']:.2f}")

    @QtCore.pyqtSlot(str)
    def _on_worker_failed(self, msg: str) -> None:
        self.status_label.setText(msg)

    @QtCore.pyqtSlot()
    def _on_worker_stopped(self) -> None:
        self.status_label.setText("Stopped")

    @QtCore.pyqtSlot()
    def _consume_buffered_frame(self) -> None:
        if not self.frame_buffer and self._final_result_payload is None:
            self._stop_playback_timer_if_idle()
            return

        if self.frame_buffer:
            frame = self.frame_buffer.popleft()
            self.steps.append(frame['generation'])
            self.distances.append(frame['best_distance'])
            self.progress_bar.setValue(frame['generation'])
            self.status_label.setText(f"Gen {frame['generation']} | Best: {frame['best_distance']:.2f}")
            self._render_frame(frame)
            self.rendered_count += 1

        if self._final_result_payload and not self.frame_buffer:
            self._render_final_result()
            self._final_result_payload = None
            self._stop_playback_timer_if_idle()

    def _render_frame(self, payload: Dict[str, Any]) -> None:
        best_route = payload['best_route']
        best_distance = payload['best_distance']

        self.route_ax.clear()
        self.route_ax.scatter(self.current_cities[:, 0], self.current_cities[:, 1], c='blue', s=30)
        
        route_cities = self.current_cities[best_route]
        route_cities = np.vstack((route_cities, route_cities[0]))
        self.route_ax.plot(route_cities[:, 0], route_cities[:, 1], 'r-')
        self.route_ax.set_title(f"Best Route (Distance: {best_distance:.2f})")
        self.route_ax.set_aspect('equal')
        self.route_canvas.draw()

        self.conv_ax.clear()
        self.conv_ax.plot(self.steps, self.distances, 'g-', linewidth=2)
        self.conv_ax.set_title("Best Distance over Generations")
        self.conv_ax.set_xlabel("Generation")
        self.conv_ax.set_ylabel("Distance")
        self.conv_canvas.draw()

    def _render_final_result(self) -> None:
        payload = self._final_result_payload
        self._render_frame({
            'best_route': payload['best_route'],
            'best_distance': payload['best_distance']
        })

def main():
    # Delegate to the canonical tsp_ga_app GUI launcher when available
    try:
        from tsp_ga_app.gui import launch_gui

        return launch_gui()
    except Exception:
        # Fallback to local runner if import fails
        app = QtWidgets.QApplication(sys.argv)
        window = TSPControlPanel()
        window.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
