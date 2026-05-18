from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit, QComboBox, QMessageBox
)

from core.mapper import build_mapping


class MappingEditorDialog(QDialog):
    """Éditeur humain de mapping : ajout/suppression/validation/réassignation."""

    def __init__(self, template: Dict[str, Any], elements: List[Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mapping dynamique — validation humaine")
        self.resize(1200, 700)

        self._elements = elements
        self._template = copy.deepcopy(template)
        self._mapped = build_mapping(self._template, self._elements)
        self._history: List[Dict[str, Any]] = []
        self._selected_element: Optional[Any] = None

        self._build_ui()
        self._reload_table()

    def set_selected_element(self, elem: Any) -> None:
        self._selected_element = elem
        if elem is None:
            self._selected_label.setText("Aucun élément sélectionné")
        else:
            self._selected_label.setText(f"Élément sélectionné: id={elem.id} [{elem.subtype}] {elem.label}")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self._selected_label = QLabel("Aucun élément sélectionné")
        self._selected_label.setStyleSheet("color:#94a3b8;")
        top.addWidget(self._selected_label)
        top.addStretch(1)
        layout.addLayout(top)

        form = QHBoxLayout()
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("logical_key")
        self._ui_type = QComboBox()
        self._ui_type.addItems(["text_input", "button", "checkbox", "dropdown", "radio", "unknown"])
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("path (ex: Fiche Patient > Diagnostic)")
        self._action = QComboBox()
        self._action.addItems(["click", "click_then_type", "select", "toggle", "none"])

        self._btn_add = QPushButton("➕ Ajouter champ")
        self._btn_add.clicked.connect(self._add_field)
        form.addWidget(self._key_input, 2)
        form.addWidget(self._ui_type, 1)
        form.addWidget(self._path_input, 2)
        form.addWidget(self._action, 1)
        form.addWidget(self._btn_add)
        layout.addLayout(form)

        self._table = QTableWidget(0, 10)
        self._table.setHorizontalHeaderLabels([
            "logical_key", "ui_type", "path", "action", "mapped", "validated",
            "confidence", "detected_label", "bbox_relative", "click_target"
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.SelectedClicked)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table)

        actions = QHBoxLayout()
        self._btn_assign = QPushButton("🎯 Assigner depuis élément sélectionné")
        self._btn_assign.clicked.connect(self._assign_from_selected_element)
        self._btn_remove = QPushButton("🗑️ Supprimer champ")
        self._btn_remove.clicked.connect(self._remove_selected_field)
        self._btn_validate = QPushButton("✅ Toggle human-validated")
        self._btn_validate.clicked.connect(self._toggle_validated)
        self._btn_recompute = QPushButton("♻️ Recalcul auto-mapping")
        self._btn_recompute.clicked.connect(self._recompute)
        self._btn_close = QPushButton("✔️ Terminer")
        self._btn_close.clicked.connect(self.accept)

        for b in [self._btn_assign, self._btn_remove, self._btn_validate, self._btn_recompute, self._btn_close]:
            actions.addWidget(b)
        layout.addLayout(actions)

    def _reload_table(self) -> None:
        self._table.blockSignals(True)
        fields = self._mapped.get("fields", [])
        self._table.setRowCount(len(fields))
        for r, f in enumerate(fields):
            vals = [
                str(f.get("logical_key", "")),
                str(f.get("ui_type", "")),
                str(f.get("path", "")),
                str(f.get("action", "")),
                str(bool(f.get("mapped", False))),
                str(bool(f.get("human_validated", False))),
                str(f.get("mapping_confidence", 0.0)),
                str(f.get("detected_label", "")),
                str(f.get("bbox_relative", {})),
                str(f.get("click_target", {})),
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c >= 4:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)
        self._table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row, col = item.row(), item.column()
        fields = self._mapped.get("fields", [])
        if not (0 <= row < len(fields)):
            return
        target = fields[row]
        keys = {0: "logical_key", 1: "ui_type", 2: "path", 3: "action"}
        if col in keys:
            old = target.get(keys[col])
            target[keys[col]] = item.text().strip()
            self._history.append({"ts": datetime.utcnow().isoformat(), "event": "edit_property", "row": row, "field": keys[col], "old": old, "new": target[keys[col]]})

    def _selected_row(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _add_field(self) -> None:
        key = self._key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Champ requis", "logical_key est obligatoire.")
            return
        f = {
            "logical_key": key,
            "ui_type": self._ui_type.currentText(),
            "path": self._path_input.text().strip(),
            "action": self._action.currentText(),
            "mapped": False,
            "mapping_confidence": 0.0,
            "human_validated": False,
        }
        self._mapped.setdefault("fields", []).append(f)
        self._history.append({"ts": datetime.utcnow().isoformat(), "event": "add_field", "logical_key": key})
        self._reload_table()

    def _remove_selected_field(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        fields = self._mapped.get("fields", [])
        if row >= len(fields):
            return
        removed = fields.pop(row)
        self._history.append({"ts": datetime.utcnow().isoformat(), "event": "remove_field", "logical_key": removed.get("logical_key", "")})
        self._reload_table()

    def _assign_from_selected_element(self) -> None:
        row = self._selected_row()
        if row < 0:
            QMessageBox.information(self, "Sélection", "Sélectionnez d'abord une ligne de champ.")
            return
        if self._selected_element is None:
            QMessageBox.information(self, "Élément", "Sélectionnez d'abord un élément sur la capture (ou la liste).")
            return
        fields = self._mapped.get("fields", [])
        f = fields[row]
        e = self._selected_element
        f["mapped"] = True
        f["human_validated"] = True
        f["mapping_source"] = "human"
        f["detected_label"] = getattr(e, "label", "")
        x1, y1, x2, y2 = getattr(e, "bbox_norm", [0, 0, 0, 0])
        f["bbox_relative"] = {"x": round(float(x1), 6), "y": round(float(y1), 6), "w": round(float(max(0, x2 - x1)), 6), "h": round(float(max(0, y2 - y1)), 6)}
        cx, cy = e.click_target
        f["click_target"] = {"x": round(float(cx), 6), "y": round(float(cy), 6)}
        f["mapping_confidence"] = 1.0
        self._history.append({"ts": datetime.utcnow().isoformat(), "event": "manual_assign", "logical_key": f.get("logical_key", ""), "element_id": getattr(e, "id", None)})
        self._reload_table()

    def _toggle_validated(self) -> None:
        row = self._selected_row()
        if row < 0:
            return
        fields = self._mapped.get("fields", [])
        f = fields[row]
        new_val = not bool(f.get("human_validated", False))
        f["human_validated"] = new_val
        self._history.append({"ts": datetime.utcnow().isoformat(), "event": "toggle_validated", "logical_key": f.get("logical_key", ""), "value": new_val})
        self._reload_table()

    def _recompute(self) -> None:
        self._mapped = build_mapping(self._template, self._elements)
        self._history.append({"ts": datetime.utcnow().isoformat(), "event": "recompute_auto_mapping"})
        self._reload_table()

    def get_result(self) -> Dict[str, Any]:
        result = copy.deepcopy(self._mapped)
        result["mapping_editor"] = {
            "human_assisted": True,
            "updated_at": datetime.utcnow().isoformat(),
            "history": self._history,
            "summary": {
                "fields_total": len(result.get("fields", [])),
                "auto_mapped": sum(1 for f in result.get("fields", []) if f.get("mapped") and f.get("mapping_source", "auto") == "auto"),
                "human_validated": sum(1 for f in result.get("fields", []) if f.get("human_validated")),
            },
        }
        return result
