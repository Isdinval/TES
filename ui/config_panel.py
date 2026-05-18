"""
ui/config_panel.py
Panneau de configuration complet et persistant
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QGroupBox, QMessageBox, QTabWidget,
    QComboBox, QDialog
)
from PyQt6.QtCore import pyqtSignal
import json
from pathlib import Path

from config import AGENT_CONFIG, OLLAMA_CONFIG, UI_CONFIG, OMNIPARSER_CONFIG

CONFIG_FILE = Path(__file__).parent.parent / "config_overrides.json"


class ConfigPanel(QWidget):
    config_saved = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_from_file()          # charge les overrides persistants
        self._load_current_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("⚙️ Configuration de l'Agent")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #7c3aed;")
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._create_agent_tab(), "Agent")
        tabs.addTab(self._create_ollama_tab(), "Ollama")
        tabs.addTab(self._create_ui_tab(), "Interface")
        layout.addWidget(tabs)

        # Boutons
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 Sauvegarder & Appliquer")
        self.btn_save.clicked.connect(self._save_config)
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.clicked.connect(self._cancel)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

    # ====================== TABS ======================

    def _create_agent_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._max_cycles = QSpinBox(); self._max_cycles.setRange(1, 50)
        self._step_delay = QDoubleSpinBox(); self._step_delay.setRange(0.01, 5.0); self._step_delay.setSingleStep(0.05); self._step_delay.setSuffix(" s")
        self._post_action_delay = QDoubleSpinBox(); self._post_action_delay.setRange(0.1, 10.0); self._post_action_delay.setSingleStep(0.1); self._post_action_delay.setSuffix(" s")
        self._type_interval = QDoubleSpinBox(); self._type_interval.setRange(0.01, 0.5); self._type_interval.setSingleStep(0.01); self._type_interval.setSuffix(" s")
        self._scroll_amount = QSpinBox(); self._scroll_amount.setRange(1, 20)
        self._drag_duration = QDoubleSpinBox(); self._drag_duration.setRange(0.1, 2.0); self._drag_duration.setSingleStep(0.1); self._drag_duration.setSuffix(" s")

        form.addRow("Cycles maximum :", self._max_cycles)
        form.addRow("Délai entre actions :", self._step_delay)
        form.addRow("Pause après batch :", self._post_action_delay)
        form.addRow("Délai frappes clavier :", self._type_interval)
        form.addRow("Scroll amount :", self._scroll_amount)
        form.addRow("Drag duration :", self._drag_duration)

        group = QGroupBox("Configuration Agent (batch)")
        group.setLayout(form)
        return group

    def _create_ollama_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.addItems([
            "qwen3.5:397B-cloud", "qwen2.5:72b", "qwen2.5:32b",
            "llama3.1:70b", "llama3.1:8b", "mistral:7b", "gemma2:27b"
        ])

        self._base_url = QLineEdit()
        self._temperature = QDoubleSpinBox(); self._temperature.setRange(0.0, 1.0); self._temperature.setSingleStep(0.05)
        self._max_tokens = QSpinBox(); self._max_tokens.setRange(512, 16384); self._max_tokens.setSingleStep(512)
        self._timeout = QSpinBox(); self._timeout.setRange(30, 300); self._timeout.setSuffix(" s")

        form.addRow("Modèle :", self._model)
        form.addRow("Base URL :", self._base_url)
        form.addRow("Temperature :", self._temperature)
        form.addRow("Max tokens :", self._max_tokens)
        form.addRow("Timeout :", self._timeout)

        group = QGroupBox("Ollama")
        group.setLayout(form)
        return group

    def _create_ui_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._window_title = QLineEdit()
        self._screenshot_min_w = QSpinBox(); self._screenshot_min_w.setRange(800, 4000)
        self._elements_min_w = QSpinBox(); self._elements_min_w.setRange(600, 2000)
        self._chat_min_w = QSpinBox(); self._chat_min_w.setRange(300, 1000)

        form.addRow("Titre de la fenêtre :", self._window_title)
        form.addRow("Largeur min screenshot :", self._screenshot_min_w)
        form.addRow("Largeur min éléments :", self._elements_min_w)
        form.addRow("Largeur min chat :", self._chat_min_w)

        lbl = QLabel("🎨 Thème complet : éditez config.py pour les couleurs")
        lbl.setStyleSheet("color:#94a3b8; font-style:italic;")
        form.addRow(lbl)

        group = QGroupBox("Interface")
        group.setLayout(form)
        return group

    # ====================== LOAD / SAVE ======================

    def _load_current_values(self):
        # Agent
        self._max_cycles.setValue(AGENT_CONFIG.get('max_cycles', 20))
        self._step_delay.setValue(AGENT_CONFIG.get('step_delay', 1.4))
        self._post_action_delay.setValue(AGENT_CONFIG.get('post_action_delay', 1.2))
        self._type_interval.setValue(AGENT_CONFIG.get('type_interval', 0.04))
        self._scroll_amount.setValue(AGENT_CONFIG.get('scroll_amount', 5))
        self._drag_duration.setValue(AGENT_CONFIG.get('drag_duration', 0.5))

        # Ollama
        self._model.setCurrentText(OLLAMA_CONFIG.get('model', 'qwen3.5:397B-cloud'))
        self._base_url.setText(OLLAMA_CONFIG.get('base_url', 'http://localhost:11434'))
        self._temperature.setValue(OLLAMA_CONFIG.get('temperature', 0.1))
        self._max_tokens.setValue(OLLAMA_CONFIG.get('max_tokens', 4096))
        self._timeout.setValue(OLLAMA_CONFIG.get('timeout', 180))

        # UI
        self._window_title.setText(UI_CONFIG.get('window_title', 'GUI Agent'))
        self._screenshot_min_w.setValue(UI_CONFIG.get('screenshot_min_w', 1920))
        self._elements_min_w.setValue(UI_CONFIG.get('elements_min_w', 1080))
        self._chat_min_w.setValue(UI_CONFIG.get('chat_min_w', 420))

    def _load_from_file(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                AGENT_CONFIG.update(data.get("AGENT_CONFIG", {}))
                OLLAMA_CONFIG.update(data.get("OLLAMA_CONFIG", {}))
                UI_CONFIG.update(data.get("UI_CONFIG", {}))
            except Exception:
                pass

    def _save_config(self):
        new_config = {
            "AGENT_CONFIG": {
                "max_cycles": self._max_cycles.value(),
                "step_delay": self._step_delay.value(),
                "post_action_delay": self._post_action_delay.value(),
                "type_interval": self._type_interval.value(),
                "scroll_amount": self._scroll_amount.value(),
                "drag_duration": self._drag_duration.value(),
            },
            "OLLAMA_CONFIG": {
                "model": self._model.currentText().strip(),
                "base_url": self._base_url.text().strip(),
                "temperature": self._temperature.value(),
                "max_tokens": self._max_tokens.value(),
                "timeout": self._timeout.value(),
            },
            "UI_CONFIG": {
                "window_title": self._window_title.text().strip(),
                "screenshot_min_w": self._screenshot_min_w.value(),
                "elements_min_w": self._elements_min_w.value(),
                "chat_min_w": self._chat_min_w.value(),
            }
        }

        # Mise à jour des globals
        AGENT_CONFIG.update(new_config["AGENT_CONFIG"])
        OLLAMA_CONFIG.update(new_config["OLLAMA_CONFIG"])
        UI_CONFIG.update(new_config["UI_CONFIG"])

        # Persistance JSON
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de sauvegarder le fichier : {e}")

        self.config_saved.emit(new_config)
        QMessageBox.information(self, "Configuration", "✅ Configuration sauvegardée et appliquée !")

    def _cancel(self):
        if self.parent():
            self.parent().switch_to_main_view()

class OmniParserConfigDialog(QDialog):
    config_saved = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration OmniParser")
        self.resize(520, 280)
        self._build_ui()
        self._load_from_file()
        self._load_current_values()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)

        title = QLabel("⚙️ Paramètres principaux OmniParser")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #c8a95a;")
        layout.addWidget(title)

        form = QFormLayout()

        self._som_model_path = QLineEdit()
        self._caption_model_name = QLineEdit()
        self._caption_model_path = QLineEdit()
        self._box_threshold = QDoubleSpinBox()
        self._box_threshold.setRange(0.0001, 1.0)
        self._box_threshold.setDecimals(4)
        self._box_threshold.setSingleStep(0.001)

        form.addRow("SOM model path :", self._som_model_path)
        form.addRow("Caption model name :", self._caption_model_name)
        form.addRow("Caption model path :", self._caption_model_path)
        form.addRow("BOX_THRESHOLD :", self._box_threshold)
        layout.addLayout(form)

        hint = QLabel("Les changements sont sauvegardés dans config_overrides.json et appliqués au prochain parse.")
        hint.setStyleSheet("color:#94a3b8; font-size:11px;")
        layout.addWidget(hint)

        actions = QHBoxLayout()
        actions.addStretch(1)
        btn_close = QPushButton("Fermer")
        btn_close.clicked.connect(self.close)
        btn_save = QPushButton("💾 Sauvegarder")
        btn_save.clicked.connect(self._save_config)
        actions.addWidget(btn_close)
        actions.addWidget(btn_save)
        layout.addLayout(actions)

    def _load_current_values(self) -> None:
        self._som_model_path.setText(str(OMNIPARSER_CONFIG.get('som_model_path', '')))
        self._caption_model_name.setText(str(OMNIPARSER_CONFIG.get('caption_model_name', 'florence2')))
        self._caption_model_path.setText(str(OMNIPARSER_CONFIG.get('caption_model_path', '')))
        self._box_threshold.setValue(float(OMNIPARSER_CONFIG.get('BOX_TRESHOLD', 0.005)))

    def _load_from_file(self) -> None:
        if not CONFIG_FILE.exists():
            return
        try:
            with open(CONFIG_FILE, encoding='utf-8') as f:
                data = json.load(f)
            OMNIPARSER_CONFIG.update(data.get('OMNIPARSER_CONFIG', {}))
        except Exception:
            pass

    def _save_config(self) -> None:
        new_values = {
            'som_model_path': self._som_model_path.text().strip(),
            'caption_model_name': self._caption_model_name.text().strip() or 'florence2',
            'caption_model_path': self._caption_model_path.text().strip(),
            'BOX_TRESHOLD': float(self._box_threshold.value()),
        }

        OMNIPARSER_CONFIG.update(new_values)

        full = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, encoding='utf-8') as f:
                    full = json.load(f)
            except Exception:
                full = {}

        full['OMNIPARSER_CONFIG'] = new_values

        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(full, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.warning(self, 'Erreur', f'Impossible de sauvegarder: {exc}')
            return

        self.config_saved.emit(new_values)
        QMessageBox.information(self, 'Configuration', '✅ Paramètres OmniParser sauvegardés.')
