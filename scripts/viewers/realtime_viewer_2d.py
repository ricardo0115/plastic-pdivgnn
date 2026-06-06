"""Interactive real-time 2D viewer for the coupled LSTM + PlasticGNN model.

A PyQt5 application that runs the inference live: edit the macroscopic strain path
(per-component sliders, multi-step paths, sub-steps, range), pick the stress
component or von Mises, scrub the load history with the timestep slider, and
regenerate the mesh (type, hole radius/refinement, global size, dimensions) on the
fly. The local stress field is shown in a pyvista view (jet colormap, vertical
scalar bar) next to the three macroscopic stress-strain curves.

    python scripts/viewers/realtime_viewer_2d.py \
        --lstm-checkpoint weights/lstm.pt --gnn-checkpoint weights/gnn.pt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Qt5Agg")

import fire
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from plgnn.datagen import hole_plate_mesh
from plgnn.models import LstmConstitutiveLaw, PlasticGNN

COMPONENTS: list[str] = ["XX", "YY", "XY"]
COMP_LOWER: list[str] = ["xx", "yy", "xy"]
STRAIN_LABELS: list[str] = ["ε_xx", "ε_yy", "ε_xy"]

SLIDER_STEPS: int = 10000

DEFAULT_WIDTH: float = 1.0
DEFAULT_HEIGHT: float = 1.0
DEFAULT_RADIUS: float = 0.2
DEFAULT_HOLE_REFINEMENT_FACTOR: int = 7
DEFAULT_GLOBAL_MESH_REFINEMENT_SIZE: float = 0.06
DEFAULT_MESH_TYPE: str = "quad"

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
WEIGHTS_DIR: Path = REPO_ROOT / "weights"


def _weights_dialog_dir() -> str:
    return str(WEIGHTS_DIR if WEIGHTS_DIR.is_dir() else Path.home())


def _von_mises_2d(stress: np.ndarray) -> np.ndarray:
    xx = stress[:, 0]
    yy = stress[:, 1]
    xy = stress[:, 2]
    return np.sqrt(xx * xx - xx * yy + yy * yy + 3.0 * xy * xy)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StrainSliderWidget(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        strain_range: float = 0.05,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._strain_range = strain_range
        self._syncing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setFixedWidth(30)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(-SLIDER_STEPS, SLIDER_STEPS)
        self._slider.setValue(0)
        layout.addWidget(self._slider, stretch=1)

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(4)
        self._spin.setRange(-strain_range, strain_range)
        self._spin.setSingleStep(strain_range / 100.0)
        self._spin.setValue(0.0)
        self._spin.setFixedWidth(90)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._slider_moved)
        self._spin.valueChanged.connect(self._spin_changed)

    def _slider_to_float(self, v: int) -> float:
        return float(v) / SLIDER_STEPS * self._strain_range

    def _float_to_slider(self, v: float) -> int:
        return int(round(v / self._strain_range * SLIDER_STEPS))

    def _slider_moved(self, v: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        fv = self._slider_to_float(v)
        self._spin.setValue(fv)
        self._syncing = False
        self.valueChanged.emit(fv)

    def _spin_changed(self, v: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self._slider.setValue(self._float_to_slider(v))
        self._syncing = False
        self.valueChanged.emit(v)

    def set_value(self, v: float) -> None:
        clamped = max(-self._strain_range, min(self._strain_range, v))
        self._syncing = True
        self._spin.setValue(clamped)
        self._slider.setValue(self._float_to_slider(clamped))
        self._syncing = False

    def set_range(self, strain_range: float) -> None:
        current = self._spin.value()
        self._strain_range = strain_range
        self._spin.setRange(-strain_range, strain_range)
        self._spin.setSingleStep(strain_range / 100.0)
        self.set_value(max(-strain_range, min(strain_range, current)))


class RealtimeGUI(QMainWindow):
    def __init__(
        self,
        lstm_weights: str,
        gnn_weights: str,
        device: str | None = None,
        compile: bool = False,
        amp: bool = False,
    ) -> None:
        super().__init__()
        self.device: str = device if device is not None else _pick_device()
        self.lstm_weights_path: str = lstm_weights
        self.gnn_weights_path: str = gnn_weights
        self._use_compile: bool = compile
        self._use_amp: bool = amp

        print(f"Device: {self.device}")
        print("Loading LSTM...")
        self.lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(
            lstm_weights, self.device, n_components=3
        )

        self.mesh_type: str = DEFAULT_MESH_TYPE
        self.width: float = DEFAULT_WIDTH
        self.height: float = DEFAULT_HEIGHT
        self.radius: float = DEFAULT_RADIUS
        self.hole_refinement_factor: int = DEFAULT_HOLE_REFINEMENT_FACTOR
        self.global_mesh_refinement_size: float = DEFAULT_GLOBAL_MESH_REFINEMENT_SIZE

        print("Building default mesh + GNN...")
        self.mesh: pv.PolyData = self._build_mesh()
        self.gnn: PlasticGNN = self._build_gnn(self.mesh)
        self._apply_inference_optimizations()

        self.macro_steps: np.ndarray = np.zeros((1, 3))
        self.n_substeps: int = 25
        self.current_step_idx: int = 0
        self.current_component: int = 0
        self.current_timestep: int = 0
        self.strain_range: float = 0.05
        self.last_stress: np.ndarray = np.zeros((self.mesh.n_points, 3))

        self.lstm_time: float = 0.0
        self.gnn_time: float = 0.0
        self.show_edges: bool = False

        self._run_lstm()
        self._build_ui()
        self._run_inference_and_update()

    def _build_mesh(self) -> pv.PolyData:
        mesh = hole_plate_mesh(
            width=self.width,
            height=self.height,
            radius=self.radius,
            hole_center=(self.width / 2.0, self.height / 2.0),
            hole_refinement_factor=self.hole_refinement_factor,
            global_mesh_refinement_size=self.global_mesh_refinement_size,
            mesh_type=self.mesh_type,
        ).extract_surface()
        if self.mesh_type == "tri":
            mesh = mesh.triangulate().clean()
        return mesh

    def _build_gnn(self, mesh: pv.PolyData) -> PlasticGNN:
        gnn = PlasticGNN(self.gnn_weights_path, self.device, mesh)
        return gnn

    def _apply_inference_optimizations(self) -> None:
        if not (self._use_compile or self._use_amp):
            return
        amp_dtype: torch.dtype | None = None
        if self._use_amp:
            amp_dtype = (
                torch.float16 if self.device == "mps" else torch.bfloat16
            )
        print(
            f"Optimizing GNN: compile={self._use_compile}, amp={amp_dtype}"
        )
        self.gnn.optimize_for_inference(
            compile=self._use_compile, amp_dtype=amp_dtype
        )

    def _build_strain_path(self) -> np.ndarray:
        all_steps = np.vstack([np.zeros(3), self.macro_steps])
        path: list[np.ndarray] = []
        for i in range(len(all_steps) - 1):
            for j in range(self.n_substeps):
                alpha = j / self.n_substeps
                path.append(
                    (1 - alpha) * all_steps[i] + alpha * all_steps[i + 1]
                )
        path.append(all_steps[-1])
        return np.array(path[1:])

    def _run_lstm(self) -> None:
        self.strain_path = self._build_strain_path()
        t0 = time.perf_counter()
        self.stress_path, self.hidden_states = self.lstm.forward(
            self.strain_path, return_hidden_states=True
        )
        self.lstm_time = (time.perf_counter() - t0) * 1000
        self.n_timesteps = len(self.strain_path)
        self.current_timestep = min(
            self.current_timestep, self.n_timesteps - 1
        )

    def _get_local_stress(self, t: int) -> np.ndarray:
        if len(self.macro_steps) == 1 and np.allclose(self.macro_steps, 0):
            self.gnn_time = 0.0
            return np.zeros((self.mesh.n_points, 3))
        ms = torch.tensor(self.stress_path[t], dtype=torch.float32)
        hs = torch.tensor(self.hidden_states[t], dtype=torch.float32)
        t0 = time.perf_counter()
        pred = self.gnn.forward([ms], [hs], chunk_size=1)
        self.gnn_time = (time.perf_counter() - t0) * 1000
        return pred[0]

    def _run_inference_and_update(self) -> None:
        self._run_lstm()
        self.current_timestep = self.n_timesteps - 1
        self.last_stress = self._get_local_stress(self.current_timestep)
        self._update_timestep_slider_range()
        self._update_scalars()
        self._update_curves()

    def _build_ui(self) -> None:
        self.setWindowTitle("LSTM+PlasticGNN Real-Time 2D")
        self.resize(1600, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.controls_panel = QWidget()
        controls_layout = QVBoxLayout(self.controls_panel)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        self._build_controls(controls_layout)
        self.controls_panel.setFixedWidth(340)

        self._view_frame = QWidget()
        view_layout = QVBoxLayout(self._view_frame)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self._build_view(view_layout)

        self.curves_panel = QWidget()
        curves_layout = QVBoxLayout(self.curves_panel)
        curves_layout.setContentsMargins(0, 0, 0, 0)
        self._build_curves(curves_layout)

        self.outer_splitter = QSplitter(Qt.Horizontal)
        self.outer_splitter.addWidget(self.controls_panel)

        self.view_splitter = QSplitter(Qt.Vertical)
        self.view_splitter.addWidget(self._view_frame)
        self.view_splitter.addWidget(self.curves_panel)
        self.view_splitter.setStretchFactor(0, 3)
        self.view_splitter.setStretchFactor(1, 2)

        self.outer_splitter.addWidget(self.view_splitter)
        self.outer_splitter.setStretchFactor(0, 0)
        self.outer_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.outer_splitter, stretch=1)

        ts_layout = QHBoxLayout()
        ts_layout.addWidget(QLabel("Timestep"))
        self.timestep_slider = QSlider(Qt.Horizontal)
        self.timestep_slider.setRange(0, max(0, self.n_timesteps - 1))
        self.timestep_slider.setValue(0)
        self.timestep_slider.valueChanged.connect(self._on_timestep_change)
        ts_layout.addWidget(self.timestep_slider, stretch=1)
        self.timestep_label = QLabel(f"1/{self.n_timesteps}")
        self.timestep_label.setFixedWidth(70)
        ts_layout.addWidget(self.timestep_label)
        main_layout.addLayout(ts_layout)

        self.setStatusBar(QStatusBar())

    def _build_controls(self, layout: QVBoxLayout) -> None:
        comp_group = QGroupBox("Component")
        comp_layout = QVBoxLayout(comp_group)
        self.comp_combo = QComboBox()
        self.comp_combo.addItems(
            [f"σ_{c}" for c in COMPONENTS] + ["Von Mises"]
        )
        self.comp_combo.currentIndexChanged.connect(self._on_component_change)
        comp_layout.addWidget(self.comp_combo)
        layout.addWidget(comp_group)

        strain_group = QGroupBox("Strain")
        strain_layout = QVBoxLayout(strain_group)
        self.strain_sliders: list[StrainSliderWidget] = []
        for i, name in enumerate(STRAIN_LABELS):
            sw = StrainSliderWidget(name, self.strain_range)
            sw.valueChanged.connect(
                lambda v, idx=i: self._on_strain_change(idx, v)
            )
            strain_layout.addWidget(sw)
            self.strain_sliders.append(sw)
        layout.addWidget(strain_group)

        path_group = QGroupBox("Path")
        path_layout = QVBoxLayout(path_group)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1)
        self.step_spin.setValue(1)
        self.step_spin.valueChanged.connect(self._on_step_change)
        step_row.addWidget(self.step_spin)
        self.step_count_label = QLabel("of 1")
        step_row.addWidget(self.step_count_label)
        path_layout.addLayout(step_row)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ Add")
        btn_add.clicked.connect(self._add_macro_step)
        btn_row.addWidget(btn_add)
        btn_del = QPushButton("- Del")
        btn_del.clicked.connect(self._delete_macro_step)
        btn_row.addWidget(btn_del)
        btn_rst = QPushButton("Reset")
        btn_rst.clicked.connect(self._reset_steps)
        btn_row.addWidget(btn_rst)
        btn_rnd = QPushButton("Rand")
        btn_rnd.clicked.connect(self._randomize_steps)
        btn_row.addWidget(btn_rnd)
        path_layout.addLayout(btn_row)

        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Sub-steps"))
        self.substeps_spin = QSpinBox()
        self.substeps_spin.setRange(5, 100)
        self.substeps_spin.setValue(self.n_substeps)
        self.substeps_spin.valueChanged.connect(self._on_substeps_change)
        sub_row.addWidget(self.substeps_spin)
        path_layout.addLayout(sub_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range"))
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setDecimals(3)
        self.range_spin.setRange(0.005, 0.200)
        self.range_spin.setSingleStep(0.005)
        self.range_spin.setValue(self.strain_range)
        self.range_spin.valueChanged.connect(self._on_range_change)
        range_row.addWidget(self.range_spin)
        path_layout.addLayout(range_row)

        layout.addWidget(path_group)

        mesh_group = QGroupBox("Mesh")
        mesh_layout = QVBoxLayout(mesh_group)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type"))
        self.mesh_type_combo = QComboBox()
        self.mesh_type_combo.addItems(["quad", "tri"])
        self.mesh_type_combo.setCurrentText(self.mesh_type)
        type_row.addWidget(self.mesh_type_combo)
        mesh_layout.addLayout(type_row)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Hole radius"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setDecimals(3)
        self.radius_spin.setRange(0.02, 0.45)
        self.radius_spin.setSingleStep(0.01)
        self.radius_spin.setValue(self.radius)
        radius_row.addWidget(self.radius_spin)
        mesh_layout.addLayout(radius_row)

        hrf_row = QHBoxLayout()
        hrf_row.addWidget(QLabel("Hole refinement"))
        self.hrf_spin = QSpinBox()
        self.hrf_spin.setRange(1, 12)
        self.hrf_spin.setValue(self.hole_refinement_factor)
        hrf_row.addWidget(self.hrf_spin)
        mesh_layout.addLayout(hrf_row)

        gms_row = QHBoxLayout()
        gms_row.addWidget(QLabel("Global size"))
        self.gms_spin = QDoubleSpinBox()
        self.gms_spin.setDecimals(3)
        self.gms_spin.setRange(0.02, 0.5)
        self.gms_spin.setSingleStep(0.01)
        self.gms_spin.setValue(self.global_mesh_refinement_size)
        gms_row.addWidget(self.gms_spin)
        mesh_layout.addLayout(gms_row)

        wh_row = QHBoxLayout()
        wh_row.addWidget(QLabel("W×H"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setDecimals(2)
        self.width_spin.setRange(0.2, 5.0)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.setValue(self.width)
        wh_row.addWidget(self.width_spin)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setDecimals(2)
        self.height_spin.setRange(0.2, 5.0)
        self.height_spin.setSingleStep(0.1)
        self.height_spin.setValue(self.height)
        wh_row.addWidget(self.height_spin)
        mesh_layout.addLayout(wh_row)

        btn_regen = QPushButton("Regenerate mesh")
        btn_regen.clicked.connect(self._on_regenerate_mesh)
        mesh_layout.addWidget(btn_regen)

        btn_reset_geom = QPushButton("Training defaults")
        btn_reset_geom.clicked.connect(self._on_reset_mesh_defaults)
        mesh_layout.addWidget(btn_reset_geom)

        layout.addWidget(mesh_group)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        self.edges_check = QCheckBox("Show edges")
        self.edges_check.toggled.connect(self._on_edges_toggle)
        display_layout.addWidget(self.edges_check)
        self.markers_check = QCheckBox("Step markers")
        self.markers_check.setChecked(True)
        self.markers_check.toggled.connect(self._on_markers_toggle)
        display_layout.addWidget(self.markers_check)
        layout.addWidget(display_group)

        load_group = QGroupBox("Load")
        load_layout = QHBoxLayout(load_group)
        btn_lstm = QPushButton("LSTM...")
        btn_lstm.clicked.connect(self._load_lstm_dialog)
        load_layout.addWidget(btn_lstm)
        btn_gnn = QPushButton("GNN...")
        btn_gnn.clicked.connect(self._load_gnn_dialog)
        load_layout.addWidget(btn_gnn)
        layout.addWidget(load_group)

        layout.addStretch()

    def _build_view(self, layout: QVBoxLayout) -> None:
        self.plotter = QtInteractor(self._view_frame)
        layout.addWidget(self.plotter.interactor)

        self.mesh.point_data["stress"] = np.zeros(self.mesh.n_points)
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars="stress",
            cmap="jet",
            show_edges=self.show_edges,
            copy_mesh=False,
            scalar_bar_args=dict(
                title="σ_XX [MPa]",
                vertical=True,
                title_font_size=14,
                label_font_size=12,
            ),
        )
        self.plotter.view_xy()
        self.plotter.enable_parallel_projection()

    def _build_curves(self, layout: QVBoxLayout) -> None:
        self.fig_curves = Figure(figsize=(12, 3))
        self.canvas_curves = FigureCanvasQTAgg(self.fig_curves)
        layout.addWidget(self.canvas_curves)

        self.axes_curves = self.fig_curves.subplots(1, 3)
        self.curve_lines: list[plt.Line2D] = []
        self.curve_dots: list[plt.Line2D] = []
        self.curve_markers: list[plt.Line2D] = []
        for idx, ax in enumerate(self.axes_curves):
            comp = COMP_LOWER[idx]
            (line,) = ax.plot([], [], "b-", linewidth=1.2)
            (dot,) = ax.plot([], [], "ro", markersize=6)
            (markers,) = ax.plot([], [], "kx", markersize=8, markeredgewidth=2)
            ax.set_xlabel(f"ε_{comp}")
            ax.set_ylabel(f"σ_{comp} [MPa]")
            ax.grid(True, alpha=0.3)
            self.curve_lines.append(line)
            self.curve_dots.append(dot)
            self.curve_markers.append(markers)
        self.fig_curves.tight_layout()

    def _status_text(self) -> str:
        comp = (
            COMPONENTS[self.current_component]
            if self.current_component < 3
            else "VM"
        )
        n = len(self.macro_steps)
        T = self.n_timesteps
        t = self.current_timestep + 1
        return (
            f"σ_{comp} | Step {self.current_step_idx + 1}/{n} | "
            f"t={t}/{T} ({n}×{self.n_substeps}) | "
            f"Mesh {self.mesh_type} {self.mesh.n_points}n/{self.mesh.n_cells}e | "
            f"LSTM {self.lstm_time:.1f}ms  GNN {self.gnn_time:.1f}ms"
        )

    def _write_field(self) -> None:
        if self.current_component < 3:
            field = self.last_stress[:, self.current_component]
        else:
            field = _von_mises_2d(self.last_stress)
        self.mesh.point_data["stress"][:] = field

    def _update_scalars(self) -> None:
        self._write_field()
        self.mesh.GetPointData().Modified()

        field = self.mesh.point_data["stress"]
        clim = [float(field.min()), float(field.max())]
        self.mesh_actor.mapper.scalar_range = clim

        if self.current_component < 3:
            title = f"σ_{COMPONENTS[self.current_component]} [MPa]"
        else:
            title = "Von Mises [MPa]"
        for sb in self.plotter.scalar_bars.values():
            sb.SetTitle(title)

        self.plotter.render()
        self.statusBar().showMessage(self._status_text())

    def _rebuild_actor(self) -> None:
        if self.current_component < 3:
            title = f"σ_{COMPONENTS[self.current_component]} [MPa]"
        else:
            title = "Von Mises [MPa]"

        field = self.mesh.point_data["stress"]
        clim = [float(field.min()), float(field.max())]

        self.plotter.clear()
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars="stress",
            cmap="jet",
            clim=clim,
            show_edges=self.show_edges,
            copy_mesh=False,
            scalar_bar_args=dict(
                title=title,
                vertical=True,
                title_font_size=14,
                label_font_size=12,
            ),
            reset_camera=False,
        )
        self.plotter.render()

    def _update_curves(self) -> None:
        t = self.current_timestep
        show_markers = self.markers_check.isChecked()
        step_indices = [
            (k + 1) * self.n_substeps - 1
            for k in range(len(self.macro_steps))
            if (k + 1) * self.n_substeps - 1 < len(self.strain_path)
        ]
        for idx, ax in enumerate(self.axes_curves):
            strain = self.strain_path[:, idx]
            stress = self.stress_path[:, idx]
            self.curve_lines[idx].set_data(strain, stress)
            self.curve_dots[idx].set_data([strain[t]], [stress[t]])
            if show_markers:
                sx = [strain[i] for i in step_indices]
                sy = [stress[i] for i in step_indices]
                self.curve_markers[idx].set_data(sx, sy)
            else:
                self.curve_markers[idx].set_data([], [])
        x_min = float(self.strain_path.min())
        x_max = float(self.strain_path.max())
        y_min = float(self.stress_path.min())
        y_max = float(self.stress_path.max())
        x_pad = max(0.05 * (x_max - x_min), 1e-6)
        y_pad = max(0.05 * (y_max - y_min), 1e-6)
        for ax in self.axes_curves:
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        self.canvas_curves.draw_idle()

    def _update_timestep_slider_range(self) -> None:
        self.timestep_slider.blockSignals(True)
        self.timestep_slider.setRange(0, max(0, self.n_timesteps - 1))
        self.timestep_slider.setValue(self.current_timestep)
        self.timestep_slider.blockSignals(False)
        self.timestep_label.setText(
            f"{self.current_timestep + 1}/{self.n_timesteps}"
        )

    def _sync_strain_sliders(self) -> None:
        row = self.macro_steps[self.current_step_idx]
        for i, sw in enumerate(self.strain_sliders):
            sw.set_value(float(row[i]))

    def _update_step_controls(self) -> None:
        n = len(self.macro_steps)
        self.step_spin.blockSignals(True)
        self.step_spin.setRange(1, max(1, n))
        self.step_spin.setValue(self.current_step_idx + 1)
        self.step_spin.blockSignals(False)
        self.step_count_label.setText(f"of {n}")

    def _on_strain_change(self, component_idx: int, value: float) -> None:
        self.macro_steps[self.current_step_idx, component_idx] = value
        self._run_inference_and_update()

    def _on_step_change(self, one_based: int) -> None:
        self.current_step_idx = max(0, one_based - 1)
        self._sync_strain_sliders()

    def _add_macro_step(self) -> None:
        new_row = self.macro_steps[self.current_step_idx].copy()
        self.macro_steps = np.vstack([self.macro_steps, new_row])
        self.current_step_idx = len(self.macro_steps) - 1
        self._update_step_controls()
        self._sync_strain_sliders()
        self._run_inference_and_update()

    def _delete_macro_step(self) -> None:
        if len(self.macro_steps) <= 1:
            return
        self.macro_steps = np.delete(
            self.macro_steps, self.current_step_idx, axis=0
        )
        self.current_step_idx = min(
            self.current_step_idx, len(self.macro_steps) - 1
        )
        self._update_step_controls()
        self._sync_strain_sliders()
        self._run_inference_and_update()

    def _reset_steps(self) -> None:
        self.macro_steps = np.zeros((1, 3))
        self.current_step_idx = 0
        self._update_step_controls()
        self._sync_strain_sliders()
        self._run_inference_and_update()

    def _randomize_steps(self) -> None:
        n = max(1, len(self.macro_steps))
        self.macro_steps = np.random.uniform(
            -self.strain_range, self.strain_range, size=(n, 3)
        )
        self._update_step_controls()
        self._sync_strain_sliders()
        self._run_inference_and_update()

    def _on_substeps_change(self, v: int) -> None:
        self.n_substeps = v
        self._run_inference_and_update()

    def _on_range_change(self, v: float) -> None:
        self.strain_range = v
        for sw in self.strain_sliders:
            sw.set_range(v)

    def _on_component_change(self, idx: int) -> None:
        self.current_component = idx
        self._update_scalars()

    def _on_timestep_change(self, t: int) -> None:
        self.current_timestep = t
        self.last_stress = self._get_local_stress(t)
        self.timestep_label.setText(
            f"{self.current_timestep + 1}/{self.n_timesteps}"
        )
        self._update_scalars()
        self._update_curves()

    def _on_edges_toggle(self, on: bool) -> None:
        self.show_edges = on
        self._rebuild_actor()

    def _on_markers_toggle(self, on: bool) -> None:
        self._update_curves()

    def _on_regenerate_mesh(self) -> None:
        new_type = self.mesh_type_combo.currentText()
        new_radius = float(self.radius_spin.value())
        new_hrf = int(self.hrf_spin.value())
        new_gms = float(self.gms_spin.value())
        new_w = float(self.width_spin.value())
        new_h = float(self.height_spin.value())

        self.statusBar().showMessage(
            f"Rebuilding mesh ({new_type}, r={new_radius})..."
        )
        QApplication.processEvents()

        try:
            self.mesh_type = new_type
            self.radius = new_radius
            self.hole_refinement_factor = new_hrf
            self.global_mesh_refinement_size = new_gms
            self.width = new_w
            self.height = new_h
            self.mesh = self._build_mesh()
            self.gnn = self._build_gnn(self.mesh)
            self._apply_inference_optimizations()
            self.mesh.point_data["stress"] = np.zeros(self.mesh.n_points)
            self.last_stress = np.zeros((self.mesh.n_points, 3))
            self._rebuild_actor()
            self.plotter.reset_camera()
            self.plotter.view_xy()
            self._run_inference_and_update()
        except Exception as exc:
            self.statusBar().showMessage(f"Mesh regeneration failed: {exc}")
            raise

    def _on_reset_mesh_defaults(self) -> None:
        self.mesh_type_combo.setCurrentText(DEFAULT_MESH_TYPE)
        self.radius_spin.setValue(DEFAULT_RADIUS)
        self.hrf_spin.setValue(DEFAULT_HOLE_REFINEMENT_FACTOR)
        self.gms_spin.setValue(DEFAULT_GLOBAL_MESH_REFINEMENT_SIZE)
        self.width_spin.setValue(DEFAULT_WIDTH)
        self.height_spin.setValue(DEFAULT_HEIGHT)

    def _load_lstm_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load LSTM checkpoint", _weights_dialog_dir(),
            "Checkpoint (*.pth *.pt);;All files (*)",
        )
        if not path:
            return
        self.lstm_weights_path = path
        self.lstm = LstmConstitutiveLaw(path, self.device, n_components=3)
        self._run_inference_and_update()

    def _load_gnn_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load GNN checkpoint", _weights_dialog_dir(),
            "Checkpoint (*.pt *.pth);;All files (*)",
        )
        if not path:
            return
        self.gnn_weights_path = path
        self.gnn = self._build_gnn(self.mesh)
        self._apply_inference_optimizations()
        self._run_inference_and_update()


def main(
    lstm_checkpoint: str,
    gnn_checkpoint: str,
    device: str | None = None,
    compile: bool = False,
    amp: bool = False,
) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = RealtimeGUI(
        lstm_weights=lstm_checkpoint,
        gnn_weights=gnn_checkpoint,
        device=device,
        compile=compile,
        amp=amp,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    fire.Fire(main)
