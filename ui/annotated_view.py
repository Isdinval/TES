"""
ui/annotated_view.py
Screenshot interactif enrichi :
  • Overlays colorés par subtype (toujours visibles, pas juste au survol)
  • Toggle "interactifs seulement" (haut droite)
  • Minimap en bas à droite (thumbnail + éléments + curseur)
  • Badge ID sur chaque bbox
  • Croix blanche sur le click_target des text_input
  • Clignotement de l'élément en cours d'action
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from PIL import Image

from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QSizePolicy
from PyQt6.QtCore    import Qt, pyqtSignal, QRect, QTimer
from PyQt6.QtGui     import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush,
)

from core.omniparser_bridge import UIElement
from core.screen_capture    import ScreenInfo


# ── Palette subtype : (couleur_bordure, alpha_fill, couleur_label) ───────────
_STYLE: dict = {
    "text_input":  ("#22c55e", 35, "#22c55e"),
    "button":      ("#3b82f6", 30, "#3b82f6"),
    "checkbox":    ("#f59e0b", 35, "#f59e0b"),
    "radio":       ("#f59e0b", 35, "#f59e0b"),
    "dropdown":    ("#8b5cf6", 30, "#8b5cf6"),
    "link":        ("#06b6d4", 25, "#06b6d4"),
    "icon_button": ("#ec4899", 25, "#ec4899"),
    "toggle":      ("#10b981", 30, "#10b981"),
    "slider":      ("#6366f1", 25, "#6366f1"),
    "label":       ("#64748b",  6, "#64748b"),
    "unknown":     ("#475569",  4, "#475569"),
}
_DEFAULT = ("#94a3b8", 4, "#94a3b8")

_MINIMAP_W = 160
_MINIMAP_H = 100
_MINIMAP_M = 8     # marge depuis le bord


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGB":
        img = img.convert("RGB")
    data  = img.tobytes("raw", "RGB")
    qimg  = QImage(data, img.width, img.height,
                   img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class AnnotatedView(QWidget):
    coords_changed   = pyqtSignal(float, float, int, int)
    element_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap:            Optional[QPixmap]    = None
        self._minimap_px:        Optional[QPixmap]    = None
        self._elements:          List[UIElement]      = []
        self._screen_info:       Optional[ScreenInfo] = None
        self._hovered_id:        Optional[int]        = None
        self._selected_id:       Optional[int]        = None
        self._active_id:         Optional[int]        = None
        self._flash_on:          bool                 = True
        self._interactive_only:  bool                 = False

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(600, 400)
        self.setStyleSheet("background:#1a1a2e; border-radius:8px;")

        # Clignotement élément actif
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(380)
        self._flash_timer.timeout.connect(self._tick_flash)

        # ── Overlays Qt (transparents aux événements souris) ──────────────────
        _tp = Qt.WidgetAttribute.WA_TransparentForMouseEvents

        self._coord_lbl = QLabel("🖱️  —", self)
        self._coord_lbl.setStyleSheet(
            "color:#a855f7; background:rgba(0,0,0,185);"
            "padding:4px 8px; border-radius:4px; font-size:11px;"
        )
        self._coord_lbl.setAttribute(_tp)

        self._elem_lbl = QLabel("", self)
        self._elem_lbl.setStyleSheet(
            "color:#f0f0f0; background:rgba(0,0,0,185);"
            "padding:4px 8px; border-radius:4px; font-size:11px;"
        )
        self._elem_lbl.setAttribute(_tp)
        self._elem_lbl.hide()

        self._legend_lbl = QLabel("", self)
        self._legend_lbl.setStyleSheet("background:rgba(0,0,0,0); font-size:9px;")
        self._legend_lbl.setAttribute(_tp)
        self._update_legend()

        # Toggle interactifs
        self._toggle_btn = QPushButton("👁  Tous", self)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFixedSize(114, 26)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background:rgba(20,20,50,200); color:#94a3b8;
                border:1px solid #3a3a5e; border-radius:5px;
                font-size:10px; font-weight:bold;
            }
            QPushButton:checked {
                background:rgba(124,58,237,210); color:#fff;
                border-color:#a855f7;
            }
            QPushButton:hover { border-color:#7c3aed; }
        """)
        self._toggle_btn.toggled.connect(self._on_toggle)

    # ── API publique ──────────────────────────────────────────────────────────

    def update_image(self, img: Image.Image) -> None:
        self._pixmap = _pil_to_qpixmap(img)
        self._minimap_px = self._pixmap.scaled(
            _MINIMAP_W, _MINIMAP_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.update()

    def set_elements(self, elements: List[UIElement]) -> None:
        self._elements = elements
        self.update()

    def set_screen_info(self, info: ScreenInfo) -> None:
        self._screen_info = info

    def highlight_element(self, eid: Optional[int]) -> None:
        self._selected_id = eid
        self.update()

    def flash_element(self, eid: Optional[int]) -> None:
        """Clignotement pendant qu'une action s'exécute sur cet élément."""
        self._active_id = eid
        self._flash_on  = True
        if eid is not None:
            self._flash_timer.start()
        else:
            self._flash_timer.stop()
        self.update()

    # ── Slots privés ──────────────────────────────────────────────────────────

    def _tick_flash(self) -> None:
        self._flash_on = not self._flash_on
        self.update()

    def _on_toggle(self, checked: bool) -> None:
        self._interactive_only = checked
        self._toggle_btn.setText("⚡ Interactifs" if checked else "👁  Tous")
        self.update()

    # ── Events souris ─────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        nx, ny = self._to_norm(event.position().x(), event.position().y())
        if nx is None:
            self._elem_lbl.hide()
            return

        px_x = py_y = 0
        if self._screen_info:
            px_x, py_y = self._screen_info.norm_to_px(nx, ny)

        self._coord_lbl.setText(
            f"🖱️  norm ({nx:.3f}, {ny:.3f})   px ({px_x}, {py_y})"
        )
        self._coord_lbl.adjustSize()
        self.coords_changed.emit(nx, ny, px_x, py_y)

        hov = self._elem_at(nx, ny)
        self._hovered_id = hov.id if hov else None
        if hov:
            col = _STYLE.get(hov.subtype, _DEFAULT)[2]
            self._elem_lbl.setStyleSheet(
                f"color:{col}; background:rgba(0,0,0,185);"
                "padding:4px 8px; border-radius:4px; font-size:11px;"
            )
            imark = "  ⚡" if hov.is_interactive else ""
            self._elem_lbl.setText(
                f"📍 id={hov.id}  [{hov.subtype}]{imark}  {hov.label}"
            )
            self._elem_lbl.adjustSize()
            self._elem_lbl.show()
        else:
            self._elem_lbl.hide()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        nx, ny = self._to_norm(event.position().x(), event.position().y())
        if nx is None:
            return
        elem = self._elem_at(nx, ny)
        self._selected_id = elem.id if elem else None
        self.element_selected.emit(elem)
        self.update()

    # ── paintEvent ────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._pixmap:
            p.drawPixmap(self._image_rect(), self._pixmap)
            self._paint_subtype_overlays(p)
            self._paint_selected_outline(p)
            self._paint_minimap(p)
        else:
            p.fillRect(self.rect(), QColor("#1a1a2e"))
            p.setPen(QColor("#4a4a6a"))
            p.setFont(QFont("Arial", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "En attente du premier screenshot…")

    # ── Rendu : overlays colorés par subtype ──────────────────────────────────

    def _paint_subtype_overlays(self, p: QPainter) -> None:
        r = self._rect_tuple()
        if r is None:
            return
        ox, oy, dw, dh = r

        badge_font = QFont("Arial", 7)
        badge_font.setBold(True)
        p.setFont(badge_font)

        for elem in self._elements:
            if self._interactive_only and not elem.is_interactive:
                continue

            eid = elem.id
            is_hov = eid == self._hovered_id
            is_sel = eid == self._selected_id
            is_act = eid == self._active_id and self._flash_on

            style = _STYLE.get(elem.subtype, _DEFAULT)
            bcol  = style[0]
            alpha = style[1]
            if is_act or is_sel:
                alpha = min(alpha + 90, 210)
            elif is_hov:
                alpha = min(alpha + 45, 140)

            x1 = ox + elem.bbox_norm[0] * dw
            y1 = oy + elem.bbox_norm[1] * dh
            x2 = ox + elem.bbox_norm[2] * dw
            y2 = oy + elem.bbox_norm[3] * dh
            bw = x2 - x1
            bh = y2 - y1

            # Fill semi-transparent
            fill = QColor(bcol)
            fill.setAlpha(alpha)
            p.fillRect(int(x1), int(y1), int(bw), int(bh), fill)

            # Bordure
            pw = 2 if (is_sel or is_act or is_hov) else 1
            ps = Qt.PenStyle.SolidLine if (is_sel or is_act) else (
                 Qt.PenStyle.DashLine  if is_hov              else
                 Qt.PenStyle.SolidLine)
            p.setPen(QPen(QColor(bcol), pw + (1 if is_act else 0), ps))
            p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            p.drawRect(int(x1), int(y1), int(bw), int(bh))

            # Badge ID
            if bw > 18 and bh > 10:
                badge_bg = QColor(bcol)
                badge_bg.setAlpha(220)
                p.fillRect(int(x1), int(y1), 20, 11, badge_bg)
                p.setPen(QColor("#000000"))
                p.drawText(int(x1)+1, int(y1), 20, 11,
                           Qt.AlignmentFlag.AlignCenter, str(eid))

            # Croix click_target pour text_input localisé par OVD
            if elem.subtype == "text_input" and elem.input_center_norm:
                tx = ox + elem.input_center_norm[0] * dw
                ty = oy + elem.input_center_norm[1] * dh
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.drawLine(int(tx)-7, int(ty), int(tx)+7, int(ty))
                p.drawLine(int(tx), int(ty)-7, int(tx), int(ty)+7)

    # ── Rendu : contour blanc élément sélectionné ────────────────────────────

    def _paint_selected_outline(self, p: QPainter) -> None:
        r = self._rect_tuple()
        if r is None or self._selected_id is None:
            return
        ox, oy, dw, dh = r
        for elem in self._elements:
            if elem.id != self._selected_id:
                continue
            x1 = ox + elem.bbox_norm[0] * dw - 2
            y1 = oy + elem.bbox_norm[1] * dh - 2
            bw = (elem.bbox_norm[2] - elem.bbox_norm[0]) * dw + 4
            bh = (elem.bbox_norm[3] - elem.bbox_norm[1]) * dh + 4
            p.setPen(QPen(QColor("#ffffff"), 3))
            p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            p.drawRect(int(x1), int(y1), int(bw), int(bh))
            break

    # ── Rendu : minimap ───────────────────────────────────────────────────────

    def _paint_minimap(self, p: QPainter) -> None:
        if not self._minimap_px:
            return
        mw = self._minimap_px.width()
        mh = self._minimap_px.height()
        mx = self.width()  - mw - _MINIMAP_M
        my = self.height() - mh - _MINIMAP_M - 14   # -14 pour le label "minimap"

        # Fond
        p.fillRect(mx - 1, my - 1, mw + 2, mh + 2, QColor(8, 8, 20, 210))
        p.setPen(QPen(QColor("#3a3a6e"), 1))
        p.drawRect(mx - 1, my - 1, mw + 2, mh + 2)
        p.drawPixmap(mx, my, self._minimap_px)

        # Dots colorés pour chaque élément
        for elem in self._elements:
            if self._interactive_only and not elem.is_interactive:
                continue
            col = QColor(_STYLE.get(elem.subtype, _DEFAULT)[0])
            col.setAlpha(180)
            ex1 = mx + int(elem.bbox_norm[0] * mw)
            ey1 = my + int(elem.bbox_norm[1] * mh)
            ew  = max(2, int((elem.bbox_norm[2] - elem.bbox_norm[0]) * mw))
            eh  = max(2, int((elem.bbox_norm[3] - elem.bbox_norm[1]) * mh))
            p.fillRect(ex1, ey1, ew, eh, col)

        # Croix sur l'élément survolé
        if self._hovered_id is not None:
            for elem in self._elements:
                if elem.id == self._hovered_id:
                    tx = mx + int(elem.center_norm[0] * mw)
                    ty = my + int(elem.center_norm[1] * mh)
                    p.setPen(QPen(QColor("#ffffff"), 1))
                    p.drawLine(tx-4, ty, tx+4, ty)
                    p.drawLine(tx, ty-4, tx, ty+4)
                    break

        # Label sous la minimap
        p.setPen(QColor("#475569"))
        p.setFont(QFont("Arial", 7))
        p.drawText(mx, my + mh + 2, mw, 12,
                   Qt.AlignmentFlag.AlignCenter, "minimap")

    # ── Géométrie ─────────────────────────────────────────────────────────────

    def _image_rect(self) -> QRect:
        if not self._pixmap:
            return self.rect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        s      = min(ww / pw, wh / ph)
        dw, dh = int(pw * s), int(ph * s)
        return QRect((ww - dw) // 2, (wh - dh) // 2, dw, dh)

    def _rect_tuple(self) -> Optional[Tuple[int,int,int,int]]:
        if not self._pixmap:
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        s      = min(ww / pw, wh / ph)
        dw, dh = int(pw * s), int(ph * s)
        ox, oy = (ww - dw) // 2, (wh - dh) // 2
        return ox, oy, dw, dh

    def _to_norm(self, wx: float, wy: float
                 ) -> Tuple[Optional[float], Optional[float]]:
        r = self._rect_tuple()
        if r is None:
            return None, None
        ox, oy, dw, dh = r
        nx, ny = (wx - ox) / dw, (wy - oy) / dh
        return (nx, ny) if 0 <= nx <= 1 and 0 <= ny <= 1 else (None, None)

    def _elem_at(self, nx: float, ny: float) -> Optional[UIElement]:
        cands = [
            e for e in self._elements
            if (not self._interactive_only or e.is_interactive)
            and e.bbox_norm[0] <= nx <= e.bbox_norm[2]
            and e.bbox_norm[1] <= ny <= e.bbox_norm[3]
        ]
        if not cands:
            return None
        return min(cands,
                   key=lambda e: (e.bbox_norm[2]-e.bbox_norm[0]) *
                                 (e.bbox_norm[3]-e.bbox_norm[1]))

    # ── Layout des widgets overlay ────────────────────────────────────────────

    def _update_legend(self) -> None:
        pairs = [
            ("text_input",  "#22c55e"), ("button",      "#3b82f6"),
            ("checkbox",    "#f59e0b"), ("dropdown",    "#8b5cf6"),
            ("icon_button", "#ec4899"), ("label",       "#64748b"),
        ]
        html = "  ".join(
            f'<span style="color:{c}">■ {t}</span>' for t, c in pairs
        )
        self._legend_lbl.setText(f"<html>{html}</html>")
        self._legend_lbl.adjustSize()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._coord_lbl.move(8, 8)
        self._elem_lbl.move(8, 32)
        self._toggle_btn.move(self.width() - self._toggle_btn.width() - 8, 8)
        self._legend_lbl.move(8, self.height() - self._legend_lbl.height() - 4)