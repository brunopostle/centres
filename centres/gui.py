import json
import sys

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .properties import normalize_all
from .visualize import draw_onto_ax
from .worker import AnalyseWorker

_PROPERTY_LABELS = [
    ("levels_of_scale", "Levels of scale"),
    ("strong_centres", "Strong centres"),
    ("boundaries", "Boundaries"),
    ("alternating_repetition", "Alternating repetition"),
    ("positive_space", "Positive space"),
    ("good_shape", "Good shape"),
    ("local_symmetries", "Local symmetries"),
    ("deep_interlock", "Deep interlock"),
    ("contrast", "Contrast"),
    ("gradients", "Gradients"),
    ("roughness", "Roughness"),
    ("echoes", "Echoes"),
    ("the_void", "The void"),
    ("simplicity", "Simplicity"),
    ("not_separateness", "Not-separateness"),
]

_MAX_SIZE = 1024


def _load_and_rescale(path: str, max_size: int = _MAX_SIZE) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


class _FieldCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas that shows the structural field and centres overlay."""

    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 5), tight_layout=True)
        self._ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._show_placeholder()

    def _show_placeholder(self):
        self._ax.clear()
        self._ax.text(
            0.5, 0.5, "Open an image to begin",
            ha="center", va="center", transform=self._ax.transAxes,
            color="gray", fontsize=12,
        )
        self._ax.axis("off")
        self.draw()

    def update_field(self, field, centers, G, image=None):
        draw_onto_ax(self._ax, field, centers, G, image)
        self.draw()


class CentresMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Centres — Structural Wholeness Analyser")
        self.resize(1100, 680)
        self._image: np.ndarray | None = None
        self._image_path: str | None = None
        self._last_result: tuple | None = None  # (field, centers, G, energy, raw)
        self._worker: AnalyseWorker | None = None
        self._settings = QSettings("centres", "gui")
        self._build_ui()
        self._restore_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Menu bar
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction(QAction("Open Image…", self, triggered=self._open_image))
        file_menu.addAction(QAction("Save Visualisation…", self, triggered=self._save_viz))
        file_menu.addAction(QAction("Export JSON…", self, triggered=self._export_json))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Quit", self, triggered=self.close))

        # Toolbar
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._btn_open = _make_button("Open Image…", self._open_image)
        self._btn_analyse = _make_button("Analyse", self._run_analyse)
        self._btn_analyse.setEnabled(False)
        self._lbl_status = QLabel("Ready")
        self._lbl_status.setContentsMargins(8, 0, 0, 0)

        for w in (self._btn_open, self._btn_analyse, self._lbl_status):
            tb.addWidget(w)

        # Central area
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self._splitter)

        # Left: field canvas
        self._canvas = _FieldCanvas()
        self._splitter.addWidget(self._canvas)

        # Right: summary + properties table
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._lbl_summary = QLabel("—")
        self._lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self._lbl_summary)

        self._table = self._build_table()
        right_layout.addWidget(self._table)

        self._splitter.addWidget(right)
        self._splitter.setSizes([640, 460])

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(len(_PROPERTY_LABELS), 3)
        table.setHorizontalHeaderLabels(["Property", "Score (0–10)", "Raw"])
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.verticalHeader().setVisible(False)
        for row, (_, label) in enumerate(_PROPERTY_LABELS):
            table.setItem(row, 0, QTableWidgetItem(label))
            table.setItem(row, 1, QTableWidgetItem("—"))
            table.setItem(row, 2, QTableWidgetItem("—"))
        return table

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All files (*)",
        )
        if not path:
            return
        try:
            self._image = _load_and_rescale(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        self._image_path = path
        import os
        self._lbl_status.setText(os.path.basename(path))
        self._btn_analyse.setEnabled(True)
        self._last_result = None

    def _run_analyse(self):
        if self._image is None:
            return
        self._set_running()
        self._worker = AnalyseWorker(self._image, parent=self)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _save_viz(self):
        if self._last_result is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Run Analyse first to generate a visualisation.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualisation", "centres.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        self._canvas.figure.savefig(path, dpi=150, bbox_inches="tight")

    def _export_json(self):
        if self._last_result is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Run Analyse first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "centres.json",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        _, centers, _, energy, raw = self._last_result
        norm = normalize_all(raw)
        out = {
            "centres": len(centers),
            "structural_energy": round(energy, 4),
            "properties": {
                key: {"score": round(norm[key], 2), "raw": round(raw[key], 6)}
                for key, _ in _PROPERTY_LABELS
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    # Worker slots
    # ------------------------------------------------------------------

    def _on_finished(self, field, centers, G, energy, raw):
        self._last_result = (field, centers, G, energy, raw)
        self._canvas.update_field(field, centers, G, self._image)
        self._update_table(raw)
        self._lbl_summary.setText(
            f"<b>{len(centers)}</b> centres  |  energy <b>{energy:.4f}</b>"
        )
        self._set_idle()

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Analysis failed", msg)
        self._set_idle()

    # ------------------------------------------------------------------
    # Properties table
    # ------------------------------------------------------------------

    def _update_table(self, raw: dict):
        norm = normalize_all(raw)
        for row, (key, _) in enumerate(_PROPERTY_LABELS):
            score = norm[key]
            raw_val = raw[key]
            score_item = QTableWidgetItem(f"{score:.1f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            color = (
                QColor("#c8e6c9") if score >= 7
                else QColor("#fff9c4") if score >= 4
                else QColor("#ffcdd2")
            )
            score_item.setBackground(color)
            self._table.setItem(row, 1, score_item)
            raw_item = QTableWidgetItem(f"{raw_val:.4g}")
            raw_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 2, raw_item)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_running(self):
        self._btn_open.setEnabled(False)
        self._btn_analyse.setEnabled(False)
        import os
        name = os.path.basename(self._image_path) if self._image_path else ""
        self._lbl_status.setText(f"Analysing {name}…")

    def _set_idle(self):
        self._btn_open.setEnabled(True)
        self._btn_analyse.setEnabled(self._image is not None)
        import os
        self._lbl_status.setText(
            os.path.basename(self._image_path) if self._image_path else "Ready"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _restore_state(self):
        state = self._settings.value("splitter")
        if state is not None:
            self._splitter.restoreState(state)
        geo = self._settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.wait(3000)
        self._settings.setValue("splitter", self._splitter.saveState())
        self._settings.setValue("geometry", self.saveGeometry())
        event.accept()


def _make_button(label, slot):
    from PyQt6.QtWidgets import QPushButton
    btn = QPushButton(label)
    btn.clicked.connect(slot)
    return btn


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("centres")
    app.setOrganizationName("centres")
    win = CentresMainWindow()
    win.show()
    sys.exit(app.exec())
