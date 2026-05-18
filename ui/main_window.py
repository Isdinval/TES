"""
ui/main_window.py
Fenêtre principale PyQt6.

Layout :
  ┌─ Toolbar ──────────────────────────────────────────────────────────────┐
  │ Écran▼  Cycles[10]  [📸 Parse]  [▶ Lancer]  [⏹ Stop]  Modèle[…]      │
  ├─ Screenshot annoté ─┬─ Éléments ─┬─ Chat & Log ─────────────────────── │
  │                     │            │                                       │
  │  (interactif)       │  (tableau) │  (log + input)                       │
  │  hover=coords       │            │                                       │
  ├─ Status bar ───────────────────────────────────────────────────────────┤
  │ Cycle 2/10  │  État: …  │  Dernier: click(id=3)  │  norm(0.5, 0.3)    │
  └────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations
from typing import List, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QComboBox, QSpinBox,
    QPushButton, QLabel, QStatusBar, QSizePolicy,
    QMessageBox, QStackedWidget, QFileDialog,
)
from PyQt6.QtCore  import Qt, QThread, pyqtSlot
from PyQt6.QtGui   import QFont, QAction

from config import OMNIPARSER_CONFIG, OLLAMA_CONFIG, AGENT_CONFIG, UI_CONFIG
from core.screen_capture    import list_screens, ScreenInfo
from core.omniparser_bridge import OmniParserBridge, UIElement
from core.parse_task        import ParseOnceTask
from core.mapper            import build_mapping
from ui.annotated_view      import AnnotatedView
from ui.element_list        import ElementList
from ui.chat_panel          import ChatPanel

_BTN_CSS = """
QPushButton {{
    background:{bg}; color:#fff; border:none;
    border-radius:6px; padding:6px 14px; font-size:12px; font-weight:bold;
}}
QPushButton:hover    {{ background:{hover}; }}
QPushButton:disabled {{ background:#3a3a4e; color:#555; }}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_CONFIG['window_title'])
        self.resize(1600, 900)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("background:#1e1e2e; color:#e2e8f0;")

        # ── État ──────────────────────────────────────────────────────────
        self._screens:      List[ScreenInfo]  = []
        self._screen_info:  Optional[ScreenInfo] = None
        self._elements:     List[UIElement]   = []
        self._current_instruction: str        = ""

        # ── Chargement des modèles (bloquant, au démarrage) ───────────────
        self._omniparser = None
        self._planner    = None
        self._executor   = None
        self._parse_task: Optional[ParseOnceTask]  = None

        # ── Construction de l'UI ──────────────────────────────────────────
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        # Charge les écrans
        self._refresh_screens()

        # Charge les modèles en arrière-plan
        self._load_models_async()

    # ══════════════════════════════════════════════════════════════════════
    # Construction UI
    # ══════════════════════════════════════════════════════════════════════

    def _build_toolbar(self) -> None:
        tb = QToolBar("Contrôles")
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar { background:#16213e; border:none; padding:4px 8px; spacing:8px; }"
            "QLabel   { color:#94a3b8; font-size:11px; }"
        )
        self.addToolBar(tb)

        # ── Écran ─────────────────────────────────────────────────────────
        tb.addWidget(QLabel("Écran :"))
        self._screen_combo = QComboBox()
        self._screen_combo.setMinimumWidth(220)
        self._screen_combo.setStyleSheet(
            "QComboBox { background:#2a2a3e; color:#e2e8f0; border:1px solid #4a4a6e;"
            " border-radius:4px; padding:4px 8px; font-size:11px; }"
            "QComboBox QAbstractItemView { background:#2a2a3e; color:#e2e8f0; }"
        )
        self._screen_combo.currentIndexChanged.connect(self._on_screen_changed)
        tb.addWidget(self._screen_combo)

        tb.addSeparator()

        # ── Cycles ───────────────────────────────────────────────────────
        tb.addWidget(QLabel("Cycles max :"))
        self._cycles_spin = QSpinBox()
        self._cycles_spin.setRange(1, 50)
        self._cycles_spin.setValue(AGENT_CONFIG['max_cycles'])
        self._cycles_spin.setStyleSheet(
            "QSpinBox { background:#2a2a3e; color:#e2e8f0; border:1px solid #4a4a6e;"
            " border-radius:4px; padding:4px; width:60px; font-size:11px; }"
        )
        tb.addWidget(self._cycles_spin)

        tb.addSeparator()

        # ── Modèle ───────────────────────────────────────────────────────
        tb.addWidget(QLabel("Modèle :"))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(160)
        self._model_combo.addItems([
            OLLAMA_CONFIG['model'],
            "qwen2.5:7b",
            "llama3.1:8b",
            "mistral:7b",
        ])
        self._model_combo.setStyleSheet(
            "QComboBox { background:#2a2a3e; color:#e2e8f0; border:1px solid #4a4a6e;"
            " border-radius:4px; padding:4px 8px; font-size:11px; }"
        )
        tb.addWidget(self._model_combo)

        tb.addSeparator()

        # ── Boutons ───────────────────────────────────────────────────────
        self._btn_parse = QPushButton("📸  Parse")
        self._btn_parse.setStyleSheet(_BTN_CSS.format(bg="#1d4ed8", hover="#1e40af"))
        self._btn_parse.setToolTip("Capturer et analyser l'écran (sans lancer l'agent)")
        self._btn_parse.clicked.connect(self._on_parse)
        self._btn_parse.setEnabled(False)
        tb.addWidget(self._btn_parse)

        self._btn_map_export = QPushButton("🗺️  Export Mapping")
        self._btn_map_export.setStyleSheet(_BTN_CSS.format(bg="#7c3aed", hover="#6d28d9"))
        self._btn_map_export.clicked.connect(self._on_export_mapping)
        self._btn_map_export.setEnabled(False)
        tb.addWidget(self._btn_map_export)

        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_stop.setStyleSheet(_BTN_CSS.format(bg="#dc2626", hover="#b91c1c"))
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)
        tb.addWidget(self._btn_stop)
        
        # Bouton Configuration
        self._btn_config = QPushButton("⚙️  Config")
        self._btn_config.setStyleSheet(_BTN_CSS.format(bg="#475569", hover="#334155"))
        self._btn_config.setToolTip("Ouvrir les paramètres de l'agent")
        self._btn_config.clicked.connect(self.switch_to_config)
        tb.addWidget(self._btn_config)

        # ── Indicateur de chargement ──────────────────────────────────────
        self._loading_label = QLabel("⏳ Chargement des modèles…")
        self._loading_label.setStyleSheet("color:#f59e0b; font-size:11px; padding:0 12px;")
        tb.addWidget(self._loading_label)

    def _build_central(self) -> None:
        """Construit le layout central avec QStackedWidget (vue principale + config)"""
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # QStackedWidget pour switcher entre les vues
        self._stack = QStackedWidget()

        # === Création des widgets de la vue principale ===
        self._view = AnnotatedView()
        self._view.coords_changed.connect(self._on_coords_changed)
        self._view.element_selected.connect(self._on_element_selected_from_view)

        self._elem_list = ElementList()
        self._elem_list.element_clicked.connect(self._on_element_selected_from_list)

        self._chat = ChatPanel()
        self._chat.message_sent.connect(self._on_instruction_received)

        # Vue principale (splitter)
        self._main_view = self._create_main_view()

        # Vue Configuration (sera créée plus tard)
        self._config_view = None

        self._stack.addWidget(self._main_view)   # index 0 = principale

        self.main_layout.addWidget(self._stack)

    def _create_main_view(self):
        """Crée le splitter de la vue principale"""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            "QSplitter::handle { background:#3a3a5e; width:3px; }"
        )

        splitter.addWidget(self._view)
        splitter.addWidget(self._elem_list)
        splitter.addWidget(self._chat)
        splitter.setSizes([760, 340, 460])
        return splitter

    # ====================== Navigation entre vues ======================

    def switch_to_config(self):
        """Passe en mode configuration"""
        if self._config_view is None:
            from ui.config_panel import ConfigPanel
            self._config_view = ConfigPanel()
            self._config_view.config_saved.connect(self._on_config_saved)
            self._stack.addWidget(self._config_view)   # index 1

        self._stack.setCurrentIndex(1)
        self._set_busy(True)   # Désactive les contrôles pendant la config

    def switch_to_main_view(self):
        """Retour à la vue principale"""
        self._stack.setCurrentIndex(0)
        self._set_busy(False)

    def _on_config_saved(self, new_configs: dict):
        """Appelé quand l'utilisateur sauvegarde la config"""
        if "AGENT_CONFIG" in new_configs:
            AGENT_CONFIG.update(new_configs["AGENT_CONFIG"])
        if "OLLAMA_CONFIG" in new_configs:
            OLLAMA_CONFIG.update(new_configs["OLLAMA_CONFIG"])
        if "UI_CONFIG" in new_configs:
            UI_CONFIG.update(new_configs["UI_CONFIG"])

        # Synchronisation du combobox Modèle de la toolbar
        if "OLLAMA_CONFIG" in new_configs and "model" in new_configs["OLLAMA_CONFIG"]:
            self._model_combo.setCurrentText(new_configs["OLLAMA_CONFIG"]["model"])

        self.switch_to_main_view()
        self._chat.add_log("✅ Configuration mise à jour et sauvegardée", "ok")

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        sb.setStyleSheet(
            "QStatusBar { background:#16213e; color:#94a3b8; font-size:10px; }"
        )
        self.setStatusBar(sb)

        self._status_cycle = QLabel("Cycle : —")
        self._status_state = QLabel("En attente")
        self._status_last  = QLabel("Dernière action : —")
        self._status_norm  = QLabel("🖱️  —")

        for lbl in [self._status_cycle, self._status_state,
                    self._status_last, self._status_norm]:
            lbl.setStyleSheet("color:#94a3b8; padding:0 8px;")
            sb.addWidget(lbl)

        sb.addPermanentWidget(self._status_norm)

    # ══════════════════════════════════════════════════════════════════════
    # Chargement des modèles (thread dédié)
    # ══════════════════════════════════════════════════════════════════════

    def _load_models_async(self) -> None:
        class _Loader(QThread):
            def __init__(self_, parent):
                super().__init__(parent)
            def run(self_):
                try:
                    self._omniparser = OmniParserBridge(OMNIPARSER_CONFIG)
                    self._planner  = None
                    self._executor = None
                except Exception as e:
                    self._load_error = str(e)
                else:
                    self._load_error = None

        self._load_error = None
        self._loader = _Loader(self)
        self._loader.finished.connect(self._on_models_loaded)
        self._loader.start()

    @pyqtSlot()
    def _on_models_loaded(self) -> None:
        if self._load_error:
            self._loading_label.setText(f"❌ Erreur : {self._load_error}")
            self._loading_label.setStyleSheet("color:#ef4444; font-size:11px; padding:0 12px;")
            self._chat.add_log(f"Erreur chargement modèles : {self._load_error}", "error")
            return

        self._loading_label.setText("✓ Modèles chargés")
        self._loading_label.setStyleSheet("color:#22c55e; font-size:11px; padding:0 12px;")
        self._btn_parse.setEnabled(True)
        self._btn_map_export.setEnabled(True)
        self._chat.add_log("✓ OmniParser chargé — mode MAPPER (détection + export JSON)", "ok")
        self._status_state.setText("Prêt")

    # ══════════════════════════════════════════════════════════════════════
    # Gestion des écrans
    # ══════════════════════════════════════════════════════════════════════

    def _refresh_screens(self) -> None:
        self._screens = list_screens()
        self._screen_combo.blockSignals(True)
        self._screen_combo.clear()
        for s in self._screens:
            self._screen_combo.addItem(s.name)
        self._screen_combo.blockSignals(False)
        if self._screens:
            self._screen_info = self._screens[0]
            self._view.set_screen_info(self._screen_info)

    @pyqtSlot(int)
    def _on_screen_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._screens):
            self._screen_info = self._screens[idx]
            self._view.set_screen_info(self._screen_info)

    # ══════════════════════════════════════════════════════════════════════
    # Boutons toolbar
    # ══════════════════════════════════════════════════════════════════════

    @pyqtSlot(str)
    def _on_instruction_received(self, instruction: str) -> None:
        """
        Appelé quand l'utilisateur valide son message dans le chat (Entrée ou bouton).
        Stocke l'instruction et la lance immédiatement si les modèles sont prêts.
        """
        if not instruction:
            return
        self._current_instruction = instruction
        # Si les modèles sont chargés et aucun agent ne tourne → lancement auto
        self._chat.add_log(
            "✏️ Instruction reçue (mode mapper) : faites un Parse puis Export Mapping.", "info"
        )

    @pyqtSlot()
    def _on_parse(self) -> None:
        if not self._screen_info or not self._omniparser:
            return
        self._set_busy(True, parse_only=True)
        self._status_state.setText("Analyse en cours…")
        self._chat.add_log("📸 Lancement du parse ponctuel…", "info")

        self._parse_task = ParseOnceTask(self._omniparser, self._screen_info, self)
        self._parse_task.screenshot_ready.connect(self._on_screenshot_ready)
        self._parse_task.elements_ready.connect(self._on_elements_ready)
        self._parse_task.log_message.connect(self._chat.add_log)
        self._parse_task.finished_ok.connect(lambda: self._set_busy(False))
        self._parse_task.finished_ok.connect(
            lambda: self._status_state.setText("Prêt")
        )
        self._parse_task.start()

    @pyqtSlot()
    def _on_export_mapping(self) -> None:
        if not self._elements:
            QMessageBox.warning(self, "Aucun élément", "Faites d'abord un parse pour détecter les éléments UI.")
            return

        template_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner le template JSON", "", "JSON (*.json)")
        if not template_path:
            return

        import json
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Erreur template", f"Impossible de lire le template : {exc}")
            return

        mapped = build_mapping(template, self._elements)
        out_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le mapping généré", "mapping_output.json", "JSON (*.json)")
        if not out_path:
            return

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(mapped, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.critical(self, "Erreur export", f"Impossible d'écrire le JSON : {exc}")
            return

        n_mapped = sum(1 for field in mapped.get("fields", []) if field.get("mapped"))
        self._chat.add_log(f"🗺️ Mapping exporté : {n_mapped}/{len(mapped.get('fields', []))} champs mappés", "ok")
        self._status_state.setText("Mapping exporté")

    @pyqtSlot()
    def _on_stop(self) -> None:
        if self._parse_task and self._parse_task.isRunning():
            self._parse_task.terminate()
        self._set_busy(False)

    # ══════════════════════════════════════════════════════════════════════
    # Slots agent
    # ══════════════════════════════════════════════════════════════════════

    @pyqtSlot(object)
    def _on_screenshot_ready(self, img) -> None:
        self._view.update_image(img)

    @pyqtSlot(list)
    def _on_elements_ready(self, elements: list) -> None:
        self._elements = elements
        self._elem_list.update_elements(elements)
        self._view.set_elements(elements)

    @pyqtSlot(int, int)
    def _on_cycle_updated(self, current: int, total: int) -> None:
        self._status_cycle.setText(f"Cycle : {current}/{total}")

    # ══════════════════════════════════════════════════════════════════════
    # Interactions vue / liste éléments
    # ══════════════════════════════════════════════════════════════════════

    @pyqtSlot(float, float, int, int)
    def _on_coords_changed(self, nx: float, ny: float,
                            px_x: int, px_y: int) -> None:
        self._status_norm.setText(
            f"🖱️  norm({nx:.3f}, {ny:.3f})  px({px_x}, {px_y})"
        )

    @pyqtSlot(object)
    def _on_element_selected_from_view(self, elem) -> None:
        if elem is None:
            return
        self._elem_list.highlight_element(elem.id)
        self._chat.add_log(
            f"📍 Clic vue → id={elem.id} [{elem.elem_type}] {elem.label}", "info"
        )

    @pyqtSlot(object)
    def _on_element_selected_from_list(self, elem) -> None:
        self._view.highlight_element(elem.id)
        self._chat.add_log(
            f"📋 Sélection liste → id={elem.id} [{elem.elem_type}] {elem.label}", "info"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _set_busy(self, busy: bool, parse_only: bool = False) -> None:
        self._btn_parse.setEnabled(not busy and self._omniparser is not None)
        self._btn_map_export.setEnabled(not busy and self._omniparser is not None)
        self._btn_stop.setEnabled(busy)
        self._screen_combo.setEnabled(not busy)
        self._chat.set_input_enabled(not busy)
        if not busy:
            self._status_state.setText("Prêt")

    def closeEvent(self, event) -> None:
        self._on_stop()
        super().closeEvent(event)
