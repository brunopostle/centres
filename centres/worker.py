import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .pipeline import analyze
from .properties import compute_all


class AnalyseWorker(QThread):
    """Run analyze() off the main thread and emit results via Qt signals."""

    finished = pyqtSignal(object, object, object, float, object)
    # field (ndarray), centers (list[Center]), G (Graph), energy (float),
    # raw_scores (dict[str, float])
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.image = image

    def run(self):
        try:
            field, centers, G, energy = analyze(self.image)
            raw = compute_all(field, centers, G)
            self.finished.emit(field, centers, G, energy, raw)
        except Exception as exc:
            self.error.emit(str(exc))
