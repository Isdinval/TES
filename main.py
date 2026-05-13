"""
main.py — Point d'entrée de l'agent GUI.

Usage :
    python main.py

Prérequis :
    pip install -r requirements.txt
    Ollama doit tourner :  ollama serve
    OmniParser weights dans ./weights/
"""
import sys
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "GUI Agent"

# ── Patch de compatibilité NumPy 2.0 ─────────────────────────────────────────
# np.sctypes a été supprimé dans NumPy 2.0.
# imgaug (dépendance de paddleocr) l'utilise encore → on le recrée AVANT
# tout import de paddleocr / imgaug.
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int':     [np.int8,    np.int16,    np.int32,    np.int64],
        'uint':    [np.uint8,   np.uint16,   np.uint32,   np.uint64],
        'float':   [np.float32, np.float64,  np.longdouble],
        'complex': [np.complex64, np.complex128],
        'others':  [bool, object, bytes, str, np.void],
    }
# ─────────────────────────────────────────────────────────────────────────────

# Assure que le répertoire courant est dans le PATH Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from ui.main_window  import MainWindow


def main() -> None:
    # Silencer le warning DPI de Qt sur Windows (SetProcessDpiAwarenessContext)
    # Qt6 le gère déjà en DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 par défaut
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "RoundPreferFloor")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()