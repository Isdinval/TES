"""
ui/element_list.py
Tableau des éléments UI détectés :
  • Barre de recherche (description / OCR)
  • Filtre par subtype (combobox)
  • Clignotement de l'élément en cours d'action (fond coloré animé)
  • Compteur interactifs / total
"""
from __future__ import annotations
from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QLineEdit, QComboBox,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QTimer
from PyQt6.QtGui   import QColor, QFont

from core.omniparser_bridge import UIElement

_COLS       = ["ID", "Subtype", "I?", "Description", "Cx", "Cy"]
_COL_WIDTHS = [30, 82, 22, 0, 46, 46]   # 0 = stretch sur Description

_SUBTYPE_COLORS = {
    "text_input":  "#22c55e",
    "button":      "#3b82f6",
    "checkbox":    "#f59e0b",
    "radio":       "#f59e0b",
    "dropdown":    "#8b5cf6",
    "link":        "#06b6d4",
    "icon_button": "#ec4899",
    "toggle":      "#10b981",
    "slider":      "#6366f1",
    "label":       "#64748b",
    "unknown":     "#475569",
}

_ALL_SUBTYPES = ["— tous —"] + sorted(_SUBTYPE_COLORS.keys())

_CSS_INPUT = """
QLineEdit, QComboBox {
    background:#0b1220; color:#dbe4f0;
    border:1px solid #334155; border-radius:6px;
    padding:3px 6px; font-size:11px;
}
QLineEdit:focus, QComboBox:focus { border-color:#c8a95a; }
QComboBox QAbstractItemView { background:#0b1220; color:#dbe4f0; }
"""

_CSS_TABLE = """
QTableWidget {
    background:#0f172a; color:#dbe4f0;
    border:none; font-size:11px;
    alternate-background-color:#131f36;
    gridline-color:#1e293b;
}
QTableWidget::item:selected { background:#7a5a1f; color:#fffdf6; }
QHeaderView::section {
    background:#111827; color:#c8a95a;
    border:none; padding:3px 4px; font-size:10px;
    border-bottom:1px solid #2f3a52;
}
"""


class ElementList(QWidget):
    element_clicked = pyqtSignal(object)   # UIElement

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_elements:  List[UIElement] = []
        self._visible_rows:  List[int]       = []   # ids affichés
        self._flash_id:      Optional[int]   = None
        self._flash_on:      bool            = True
        self._flash_color:   str             = "#7c3aed"

        # Timer clignotement
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(380)
        self._flash_timer.timeout.connect(self._tick_flash)

        self._build_ui()

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ── En-tête ───────────────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        title = QLabel("Éléments détectés")
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title.setStyleSheet("color:#f3f6fb; padding:4px 4px 0 4px;")
        hdr_row.addWidget(title)
        hdr_row.addStretch()
        self._count_lbl = QLabel("—")
        self._count_lbl.setStyleSheet("color:#b8c4da; font-size:10px; padding:0 4px;")
        hdr_row.addWidget(self._count_lbl)
        root.addLayout(hdr_row)

        # ── Barre de filtre ───────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        filter_row.setContentsMargins(4, 0, 4, 0)

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 Rechercher…")
        self._search.setStyleSheet(_CSS_INPUT)
        self._search.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._search, stretch=2)

        self._subtype_combo = QComboBox()
        self._subtype_combo.addItems(_ALL_SUBTYPES)
        self._subtype_combo.setStyleSheet(_CSS_INPUT)
        self._subtype_combo.setFixedWidth(110)
        self._subtype_combo.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(self._subtype_combo)

        root.addLayout(filter_row)

        # ── Tableau ───────────────────────────────────────────────────────────
        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSortingEnabled(False)
        self._table.setStyleSheet(_CSS_TABLE)

        hdr = self._table.horizontalHeader()
        for i, w in enumerate(_COL_WIDTHS):
            if w == 0:
                hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            else:
                self._table.setColumnWidth(i, w)

        self._table.itemSelectionChanged.connect(self._on_selection)
        root.addWidget(self._table)

    # ── API publique ──────────────────────────────────────────────────────────

    def update_elements(self, elements: List[UIElement]) -> None:
        self._all_elements = elements
        self._apply_filter()

    def highlight_element(self, eid: Optional[int]) -> None:
        """Sélectionne et scroll vers la ligne de l'élément."""
        if eid is None:
            self._table.clearSelection()
            return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.text() == str(eid):
                self._table.selectRow(row)
                self._table.scrollToItem(item)
                return

    def flash_element(self, eid: Optional[int]) -> None:
        """Fait clignoter la ligne de l'élément en cours d'action."""
        self._flash_id  = eid
        self._flash_on  = True
        if eid is not None:
            col = _SUBTYPE_COLORS.get(
                next((e.subtype for e in self._all_elements if e.id == eid), ""),
                "#7c3aed"
            )
            self._flash_color = col
            self.highlight_element(eid)
            self._flash_timer.start()
        else:
            self._flash_timer.stop()
            self._repaint_flash_row(clear=True)

    def clear(self) -> None:
        self._all_elements = []
        self._table.setRowCount(0)
        self._count_lbl.setText("—")

    # ── Filtrage ──────────────────────────────────────────────────────────────

    def _apply_filter(self) -> None:
        query   = self._search.text().strip().lower()
        subtype = self._subtype_combo.currentText()
        if subtype == "— tous —":
            subtype = ""

        filtered = [
            e for e in self._all_elements
            if (not subtype or e.subtype == subtype)
            and (not query or query in (e.description + e.ocr_text).lower()
                           or query in str(e.id))
        ]

        n_total       = len(self._all_elements)
        n_interactive = sum(1 for e in self._all_elements if e.is_interactive)
        n_shown       = len(filtered)
        self._count_lbl.setText(
            f"{n_shown}/{n_total} éléments  •  ⚡{n_interactive} interactifs"
        )

        self._table.setRowCount(0)
        for elem in filtered:
            self._insert_row(elem)

    def _insert_row(self, elem: UIElement) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        imark = "✓" if elem.is_interactive else ""
        vals  = [
            str(elem.id),
            elem.subtype,
            imark,
            elem.description or elem.ocr_text,
            f"{elem.click_target[0]:.3f}",
            f"{elem.click_target[1]:.3f}",
        ]
        for col, text in enumerate(vals):
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            if col == 1:
                c = _SUBTYPE_COLORS.get(elem.subtype, "#94a3b8")
                item.setForeground(QColor(c))
                item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            elif col == 2:
                item.setForeground(
                    QColor("#22c55e" if elem.is_interactive else "#334155")
                )
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Fond clignotant si c'est l'élément actif
            if elem.id == self._flash_id and self._flash_on:
                bg = QColor(self._flash_color)
                bg.setAlpha(55)
                item.setBackground(bg)

            self._table.setItem(row, col, item)
        self._table.setRowHeight(row, 22)

    # ── Flash animation ───────────────────────────────────────────────────────

    def _tick_flash(self) -> None:
        self._flash_on = not self._flash_on
        self._repaint_flash_row()

    def _repaint_flash_row(self, clear: bool = False) -> None:
        if self._flash_id is None:
            return
        for row in range(self._table.rowCount()):
            id_item = self._table.item(row, 0)
            if not id_item or id_item.text() != str(self._flash_id):
                continue
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item is None:
                    continue
                if clear or not self._flash_on:
                    item.setBackground(QColor(0, 0, 0, 0))
                else:
                    bg = QColor(self._flash_color)
                    bg.setAlpha(55)
                    item.setBackground(bg)
            break

    # ── Sélection ─────────────────────────────────────────────────────────────

    def _on_selection(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        id_item = self._table.item(row, 0)
        if not id_item:
            return
        try:
            eid = int(id_item.text())
        except ValueError:
            return
        for e in self._all_elements:
            if e.id == eid:
                self.element_clicked.emit(e)
                return