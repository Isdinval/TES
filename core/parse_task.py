from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.screen_capture import capture


class ParseOnceTask(QThread):
    screenshot_ready = pyqtSignal(object)
    elements_ready = pyqtSignal(list)
    log_message = pyqtSignal(str, str)
    finished_ok = pyqtSignal()

    def __init__(self, omniparser, screen_info, parent=None):
        super().__init__(parent)
        self.omniparser = omniparser
        self.screen_info = screen_info

    def run(self) -> None:
        try:
            self.log_message.emit("📸 Capture…", "info")
            pil_img = capture(self.screen_info)
            self.log_message.emit("🔍 Analyse OmniParser…", "info")
            annotated, elements = self.omniparser.parse(pil_img)
            self.screenshot_ready.emit(annotated)
            self.elements_ready.emit(elements)
            self.log_message.emit(f"✓ {len(elements)} éléments", "ok")
            self.finished_ok.emit()
        except Exception as exc:
            self.log_message.emit(f"Erreur : {exc}", "error")
