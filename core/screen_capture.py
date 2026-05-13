"""
core/screen_capture.py
Capture d'écran multi-écrans via mss.
Retourne des images PIL à résolution native logique.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image

try:
    import mss
    import mss.tools
except ImportError:
    raise ImportError("Installe mss : pip install mss")


@dataclass
class ScreenInfo:
    """Informations sur un écran cible."""
    index: int          # index dans mss.monitors (1-based, 0 = virtuel total)
    name:  str
    left:  int          # coords logiques
    top:   int
    width: int
    height: int

    @property
    def monitor_dict(self) -> dict:
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height}

    def norm_to_px(self, nx: float, ny: float) -> Tuple[int, int]:
        """Coordonnées normalisées [0,1] → pixels écran logiques."""
        return (
            int(self.left + nx * self.width),
            int(self.top  + ny * self.height),
        )

    def widget_to_norm(self, wx: float, wy: float,
                       widget_w: int, widget_h: int) -> Tuple[float, float]:
        """Coords widget Qt (px) → normalisées [0,1]."""
        return wx / widget_w, wy / widget_h


def list_screens() -> List[ScreenInfo]:
    """Retourne la liste des écrans disponibles (index 1, 2, ...)."""
    screens: List[ScreenInfo] = []
    with mss.mss() as sct:
        for i, mon in enumerate(sct.monitors[1:], start=1):   # skip monitors[0] (all)
            screens.append(ScreenInfo(
                index  = i,
                name   = f"Écran {i}  ({mon['width']}×{mon['height']})",
                left   = mon['left'],
                top    = mon['top'],
                width  = mon['width'],
                height = mon['height'],
            ))
    return screens


def capture(screen_info: ScreenInfo) -> Image.Image:
    """
    Capture l'écran décrit par screen_info.
    Retourne une PIL.Image en RGB.
    """
    with mss.mss() as sct:
        raw = sct.grab(screen_info.monitor_dict)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    return img


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """PIL Image RGB → numpy uint8 BGR (pour OpenCV / affichage Qt)."""
    arr = np.array(img)
    return arr[:, :, ::-1].copy()   # RGB → BGR


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy BGR → PIL Image RGB."""
    return Image.fromarray(arr[:, :, ::-1])