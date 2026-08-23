import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .cli import robust_analyse
from .pipeline import analyze
from .properties import compute_all
from .transforms import BENIGN


class AnalyseWorker(QThread):
    """Run analyze() off the main thread and emit results via Qt signals."""

    finished = pyqtSignal(object, object, object, float, object, object)
    # field (ndarray), centers (list[Center]), G (Graph), energy (float),
    # raw_scores (dict[str, float]), robust (None, or (n_med, life, props,
    # n_transforms) from cli.robust_analyse -- see #23)
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, robust: bool = False, parent=None):
        super().__init__(parent)
        self.image = image
        self.robust = robust

    def run(self):
        try:
            field, centers, G, energy = analyze(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            raw = compute_all(field, centers, G, gray)
            robust = None
            if self.robust:
                n_med, life, props = robust_analyse(self.image)
                robust = (n_med, life, props, len(BENIGN))
            self.finished.emit(field, centers, G, energy, raw, robust)
        except Exception as exc:
            self.error.emit(str(exc))
