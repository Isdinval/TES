from __future__ import annotations
from typing import List, Optional
import json

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QToolBar, QComboBox,
    QPushButton, QLabel, QStatusBar, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSlot

from config import OMNIPARSER_CONFIG, UI_CONFIG
from core.screen_capture import list_screens, ScreenInfo
from core.omniparser_bridge import OmniParserBridge, UIElement
from core.parse_task import ParseOnceTask
from core.mapper import build_mapping
from ui.annotated_view import AnnotatedView
from ui.element_list import ElementList
from ui.mapping_editor import MappingEditorDialog

_BTN_CSS = """
QPushButton {{
    background:{bg};
    color:#f8fafc;
    border:1px solid {border};
    border-radius:8px;
    padding:7px 14px;
    font-size:12px;
    font-weight:600;
}}
QPushButton:hover    {{ background:{hover}; border-color:{glow}; }}
QPushButton:disabled {{ background:#2a2f3d; color:#677186; border-color:#323a4b; }}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TES Mapper — Parse & Mapping dynamique")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("background:#0f172a; color:#e2e8f0;")

        self._screens: List[ScreenInfo] = []
        self._screen_info: Optional[ScreenInfo] = None
        self._elements: List[UIElement] = []
        self._template: Optional[dict] = None
        self._template_path: Optional[str] = None
        self._last_mapped: Optional[dict] = None
        self._mapping_editor: Optional[MappingEditorDialog] = None

        self._omniparser = None
        self._parse_task: Optional[ParseOnceTask] = None

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._refresh_screens()
        self._load_models_async()

    def _build_toolbar(self) -> None:
        tb = QToolBar("Contrôles")
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar { background:#111827; border:none; padding:6px 10px; spacing:8px; border-bottom:1px solid #2f3a52; }"
            "QLabel   { color:#c6d0e1; font-size:11px; }"
            "QComboBox { background:#0b1220; color:#dbe4f0; border:1px solid #374151; border-radius:6px; padding:4px 8px; }"
            "QComboBox:focus { border-color:#c8a95a; }"
            "QComboBox QAbstractItemView { background:#0b1220; color:#dbe4f0; }"
        )
        self.addToolBar(tb)

        tb.addWidget(QLabel("Écran :"))
        self._screen_combo = QComboBox()
        self._screen_combo.setMinimumWidth(260)
        self._screen_combo.currentIndexChanged.connect(self._on_screen_changed)
        tb.addWidget(self._screen_combo)

        tb.addSeparator()

        self._btn_template = QPushButton("📄 Charger mapping JSON")
        self._btn_template.setStyleSheet(_BTN_CSS.format(bg="#334155", hover="#3f4e66", border="#4b5a73", glow="#c8a95a"))
        self._btn_template.clicked.connect(self._on_load_template)
        tb.addWidget(self._btn_template)

        self._btn_parse = QPushButton("📸 Parse")
        self._btn_parse.setStyleSheet(_BTN_CSS.format(bg="#1e3a8a", hover="#1d4ed8", border="#3b82f6", glow="#93c5fd"))
        self._btn_parse.clicked.connect(self._on_parse)
        self._btn_parse.setEnabled(False)
        tb.addWidget(self._btn_parse)

        self._btn_edit = QPushButton("🧩 Éditer mapping")
        self._btn_edit.setStyleSheet(_BTN_CSS.format(bg="#5b2a86", hover="#6d28d9", border="#8b5cf6", glow="#c4b5fd"))
        self._btn_edit.clicked.connect(self._on_open_mapping_editor)
        self._btn_edit.setEnabled(False)
        tb.addWidget(self._btn_edit)

        self._btn_export = QPushButton("🗺️ Export JSON final")
        self._btn_export.setStyleSheet(_BTN_CSS.format(bg="#14532d", hover="#166534", border="#22c55e", glow="#86efac"))
        self._btn_export.clicked.connect(self._on_export_mapping)
        self._btn_export.setEnabled(False)
        tb.addWidget(self._btn_export)

        self._template_label = QLabel("Template: non chargé")
        self._template_label.setStyleSheet("color:#c8a95a; padding-left:10px; font-weight:600;")
        tb.addWidget(self._template_label)

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view = AnnotatedView()
        self._view.coords_changed.connect(self._on_coords_changed)
        self._view.element_selected.connect(self._on_element_selected_from_view)

        self._elem_list = ElementList()
        self._elem_list.element_clicked.connect(self._on_element_selected_from_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background:#2f3a52; width:4px; }")
        splitter.addWidget(self._view)
        splitter.addWidget(self._elem_list)
        splitter.setSizes([1000, 600])
        layout.addWidget(splitter)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        sb.setStyleSheet("QStatusBar { background:#111827; color:#c6d0e1; font-size:10px; border-top:1px solid #2f3a52; }")
        self.setStatusBar(sb)
        self._status_state = QLabel("Chargement modèles…")
        self._status_norm = QLabel("🖱️  —")
        self.statusBar().addWidget(self._status_state)
        self.statusBar().addPermanentWidget(self._status_norm)

    def _refresh_screens(self) -> None:
        self._screens = list_screens()
        self._screen_combo.clear()
        for s in self._screens:
            self._screen_combo.addItem(f"#{s.index} — {s.width}x{s.height} @ ({s.left},{s.top})")
        if self._screens:
            self._screen_combo.setCurrentIndex(0)
            self._screen_info = self._screens[0]
            self._view.set_screen_info(self._screen_info)

    def _load_models_async(self) -> None:
        try:
            self._omniparser = OmniParserBridge(OMNIPARSER_CONFIG)
            self._btn_parse.setEnabled(True)
            self._status_state.setText("Prêt")
        except Exception as exc:
            self._status_state.setText("Erreur chargement modèles")
            QMessageBox.critical(self, "Erreur", f"Impossible de charger OmniParser: {exc}")

    @pyqtSlot(int)
    def _on_screen_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._screens):
            self._screen_info = self._screens[idx]
            self._view.set_screen_info(self._screen_info)

    @pyqtSlot()
    def _on_load_template(self) -> None:
        template_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner le mapping JSON initial", "", "JSON (*.json)")
        if not template_path:
            return
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                self._template = json.load(f)
            self._template_path = template_path
            self._template_label.setText(f"Template: {template_path.split('/')[-1]}")
            self._status_state.setText("Template chargé")
            if self._elements:
                self._run_auto_mapping_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Erreur template", f"Impossible de lire le JSON: {exc}")

    @pyqtSlot()
    def _on_parse(self) -> None:
        if not self._screen_info or not self._omniparser:
            return
        self._set_busy(True)
        self._status_state.setText("Analyse en cours…")

        self._parse_task = ParseOnceTask(self._omniparser, self._screen_info, self)
        self._parse_task.screenshot_ready.connect(self._on_screenshot_ready)
        self._parse_task.elements_ready.connect(self._on_elements_ready)
        self._parse_task.finished_ok.connect(lambda: self._set_busy(False))
        self._parse_task.finished_ok.connect(lambda: self._status_state.setText("Parse terminé"))
        self._parse_task.start()

    def _run_auto_mapping_preview(self) -> None:
        if not self._template:
            return
        self._last_mapped = build_mapping(self._template, self._elements)
        fields = self._last_mapped.get("fields", [])
        n_mapped = sum(1 for f in fields if f.get("mapped"))
        self._status_state.setText(f"Parse terminé — auto-mapped {n_mapped}/{len(fields)}")

    @pyqtSlot()
    def _on_open_mapping_editor(self) -> None:
        if not self._elements:
            QMessageBox.warning(self, "Aucun élément", "Faites d'abord un Parse.")
            return

        if self._mapping_editor is not None and self._mapping_editor.isVisible():
            self._mapping_editor.raise_()
            self._mapping_editor.activateWindow()
            return

        template = self._template or {"software": "unknown", "fields": []}
        editor = MappingEditorDialog(template, self._elements, self)
        editor.setModal(False)
        editor.setWindowModality(Qt.WindowModality.NonModal)
        editor.finished.connect(self._on_mapping_editor_finished)
        self._mapping_editor = editor

        if self._elements:
            editor.set_selected_element(self._elements[0])

        editor.show()
        editor.raise_()
        editor.activateWindow()

    @pyqtSlot(int)
    def _on_mapping_editor_finished(self, result: int) -> None:
        editor = self._mapping_editor
        if editor is None:
            return

        if result == editor.DialogCode.Accepted:
            self._last_mapped = editor.get_result()
            fields = self._last_mapped.get("fields", [])
            n_human = sum(1 for f in fields if f.get("human_validated"))
            self._status_state.setText(f"Édition terminée — {n_human} champs validés humain")
            self._btn_export.setEnabled(True)

        self._mapping_editor = None

    @pyqtSlot()
    def _on_export_mapping(self) -> None:
        if not self._last_mapped:
            # fallback auto
            if self._template:
                self._last_mapped = build_mapping(self._template, self._elements)
            else:
                QMessageBox.warning(self, "Aucun mapping", "Chargez un template et lancez un parse/édition.")
                return

        out_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le mapping final", "mapping_final.json", "JSON (*.json)")
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self._last_mapped, f, indent=2, ensure_ascii=False)
            self._status_state.setText("Mapping final exporté")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur export", f"Impossible d'écrire le JSON: {exc}")

    @pyqtSlot(object)
    def _on_screenshot_ready(self, img) -> None:
        self._view.update_image(img)

    @pyqtSlot(list)
    def _on_elements_ready(self, elements: list) -> None:
        self._elements = elements
        self._elem_list.update_elements(elements)
        self._view.set_elements(elements)
        self._btn_edit.setEnabled(True)
        self._btn_export.setEnabled(self._template is not None)
        if self._template:
            self._run_auto_mapping_preview()

    @pyqtSlot(float, float, int, int)
    def _on_coords_changed(self, nx: float, ny: float, px_x: int, px_y: int) -> None:
        self._status_norm.setText(f"🖱️  norm({nx:.3f}, {ny:.3f})  px({px_x}, {px_y})")

    @pyqtSlot(object)
    def _on_element_selected_from_view(self, elem) -> None:
        if elem is None:
            return
        self._elem_list.highlight_element(elem.id)
        if self._mapping_editor is not None:
            self._mapping_editor.set_selected_element(elem)

    @pyqtSlot(object)
    def _on_element_selected_from_list(self, elem) -> None:
        self._view.highlight_element(elem.id)
        if self._mapping_editor is not None:
            self._mapping_editor.set_selected_element(elem)

    def _set_busy(self, busy: bool) -> None:
        self._btn_parse.setEnabled(not busy and self._omniparser is not None)
        self._btn_template.setEnabled(not busy)
        self._btn_edit.setEnabled(not busy and bool(self._elements))
        self._screen_combo.setEnabled(not busy)

    def closeEvent(self, event) -> None:
        if self._parse_task and self._parse_task.isRunning():
            self._parse_task.terminate()
        super().closeEvent(event)
