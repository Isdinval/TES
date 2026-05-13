"""
ui/chat_panel.py
Panneau droit : chat utilisateur + log des actions + raisonnement LLM.

Changement v2 : ajout de add_reflection() pour afficher le verdict
du nœud reflect (signal reflection_ready de AgentLoop).
"""
from __future__ import annotations
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel,
)
from PyQt6.QtCore  import Qt, pyqtSignal
from PyQt6.QtGui   import QFont, QTextCursor, QColor, QTextCharFormat

_COLORS = {
    "user":       "#7c3aed",
    "agent":      "#3b82f6",
    "ok":         "#22c55e",
    "warn":       "#f59e0b",
    "error":      "#ef4444",
    "info":       "#94a3b8",
    "think":      "#64748b",
    "action":     "#a855f7",
    "reflect_ok": "#10b981",   # vert sarcelle — plan approuvé
    "reflect_ko": "#f97316",   # orange     — plan rejeté / corrigé
}

_PREFIX = {
    "ok":         "✓",
    "warn":       "⚠",
    "error":      "✗",
    "info":       "·",
    "think":      "💭",
    "action":     "→",
    "reflect_ok": "✅",
    "reflect_ko": "🔄",
}

_CSS_BTN = """
QPushButton {{
    background:{bg}; color:#fff; border:none;
    border-radius:6px; padding:8px 18px; font-size:12px; font-weight:bold;
}}
QPushButton:hover {{ background:{hover}; }}
QPushButton:disabled {{ background:#3a3a4e; color:#666; }}
"""


class ChatPanel(QWidget):
    message_sent = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        title = QLabel("Chat & Journal")
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title.setStyleSheet("color:#e2e8f0; padding:4px;")
        layout.addWidget(title)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._log.setStyleSheet("""
            QTextEdit {
                background:#1a1a2e; color:#e2e8f0;
                border:1px solid #3a3a5e; border-radius:6px;
                font-size:11px; font-family:Consolas,monospace;
                padding:6px;
            }
        """)
        layout.addWidget(self._log, stretch=1)

        input_frame = QWidget()
        input_frame.setStyleSheet(
            "background:#2a2a3e; border-radius:8px; padding:4px;"
        )
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(6, 6, 6, 6)
        input_layout.setSpacing(6)

        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "Ex : Remplis nom=Jean, email=test@gmail.com puis clique Envoyer"
        )
        self._input.setStyleSheet("""
            QLineEdit {
                background:#1e1e2e; color:#e2e8f0;
                border:1px solid #4a4a6e; border-radius:5px;
                padding:8px; font-size:12px;
            }
            QLineEdit:focus { border-color:#7c3aed; }
        """)
        self._input.returnPressed.connect(self._send)
        input_layout.addWidget(self._input)

        btn_row = QHBoxLayout()
        self._btn_send = QPushButton("▶  Envoyer")
        self._btn_send.setStyleSheet(_CSS_BTN.format(bg="#7c3aed", hover="#6d28d9"))
        self._btn_send.clicked.connect(self._send)
        btn_row.addWidget(self._btn_send)

        self._btn_clear = QPushButton("Effacer log")
        self._btn_clear.setStyleSheet(_CSS_BTN.format(bg="#374151", hover="#4b5563"))
        self._btn_clear.clicked.connect(self._log.clear)
        btn_row.addWidget(self._btn_clear)

        input_layout.addLayout(btn_row)
        layout.addWidget(input_frame)

    # ── API publique ──────────────────────────────────────────────────────────

    def add_log(self, message: str, level: str = "info"):
        """Ajoute un message avec couleur selon le niveau - supporte le HTML pour les plans"""
        if level == "plan":
            # Affichage riche pour BatchPlan
            html = f"""
            <div style="background:#1f2937; padding:10px; border-radius:8px; margin:8px 0; border-left:4px solid #7c3aed;">
                {message}
            </div>
            """
            self._append_html(html)
            return

        # Cas normal (log simple)
        color = {
            "ok":    "#22c55e",
            "error": "#ef4444",
            "warn":  "#f59e0b",
            "info":  "#60a5fa",
            "think": "#64748b",
        }.get(level, "#e2e8f0")

        prefix = {
            "ok":    "✓",
            "warn":  "⚠",
            "error": "✗",
            "info":  "ℹ",
            "think": "💭",
        }.get(level, "•")

        html = f"<span style='color:{color}'>{prefix} {message}</span>"
        self._append_html(html)

    def add_user_message(self, text: str) -> None:
        self._append_bubble("Vous", text, _COLORS["user"])

    def add_agent_message(self, text: str) -> None:
        self._append_bubble("Agent", text, _COLORS["agent"])

    def add_llm_thinking(self, thinking: str) -> None:
        short = thinking[:300] + "…" if len(thinking) > 300 else thinking
        self.add_log(f"💭 {short}", "think")

    def add_action_result(self, msg: str, success: bool) -> None:
        self.add_log(f"Action : {msg}", "ok" if success else "error")

    def add_reflection(self, approved: bool, critique: str) -> None:
        """
        Affiche le résultat de l'auto-critique dans le journal.
        Appelé en réponse au signal reflection_ready de AgentLoop.
        """
        level   = "reflect_ok" if approved else "reflect_ko"
        verdict = "Plan approuvé" if approved else "Plan corrigé"
        short   = critique[:180] + "…" if len(critique) > 180 else critique
        self.add_log(f"[Réflexion] {verdict} — {short}", level)

    def set_input_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._btn_send.setEnabled(enabled)

    def get_instruction(self) -> str:
        return self._input.text().strip()

    def clear_input(self) -> None:
        self._input.clear()

    # ── Interne ───────────────────────────────────────────────────────────────

    def _append_html(self, html: str) -> None:
        """Ajoute du HTML brut dans le QTextEdit"""
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html + "<br>")
        self._log.setTextCursor(cursor)
        self._log.ensureCursorVisible()
        
    def _send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self.add_user_message(text)
        self.message_sent.emit(text)

    def _append_bubble(self, role: str, text: str, color: str) -> None:
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt_role = QTextCharFormat()
        fmt_role.setForeground(QColor(color))
        fmt_role.setFontWeight(QFont.Weight.Bold)
        cursor.setCharFormat(fmt_role)
        cursor.insertText(f"\n[{role}] ")

        fmt_text = QTextCharFormat()
        fmt_text.setForeground(QColor("#e2e8f0"))
        fmt_text.setFontWeight(QFont.Weight.Normal)
        cursor.setCharFormat(fmt_text)
        cursor.insertText(f"{text}\n")

        self._log.setTextCursor(cursor)
        self._log.ensureCursorVisible()