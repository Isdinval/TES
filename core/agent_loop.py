"""
core/agent_loop.py
Wrapper QThread autour du graphe LangGraph compilé — version batch.

Nouveaux signaux :
  • batch_progress (int, int, str) : progression dans le batch courant
"""
from __future__ import annotations
from typing import Optional, Any

from PyQt6.QtCore import QThread, pyqtSignal

from core.screen_capture    import ScreenInfo, capture
from core.omniparser_bridge import OmniParserBridge
from core.llm_planner       import LLMPlanner
from core.action_executor   import ActionExecutor
from core.graph             import build_agent_graph


class AgentLoop(QThread):

    # ── Signaux Qt ─────────────────────────────────────────────────────────
    screenshot_ready = pyqtSignal(object)       # PIL.Image annotée
    elements_ready   = pyqtSignal(list)
    plan_ready       = pyqtSignal(object)       # BatchPlan
    action_done      = pyqtSignal(str, bool)
    log_message      = pyqtSignal(str, str)
    cycle_updated    = pyqtSignal(int, int)
    task_completed   = pyqtSignal(str)
    llm_thinking     = pyqtSignal(str)
    reflection_ready = pyqtSignal(bool, str)
    batch_progress   = pyqtSignal(int, int, str)  # ← NOUVEAU : step, total, msg

    def __init__(
        self,
        omniparser: OmniParserBridge,
        planner:    LLMPlanner,
        executor:   ActionExecutor,
        agent_cfg:  dict,
        parent=None,
    ):
        super().__init__(parent)
        self.omniparser = omniparser
        self.planner    = planner
        self.executor   = executor
        self.cfg        = agent_cfg
        self._stop_flag = False

        self.instruction: str                  = ""
        self.screen_info: Optional[ScreenInfo] = None
        self.max_cycles:  int                  = agent_cfg.get("max_cycles", 10)
        self._graph:      Optional[Any]        = None

    def set_task(
        self,
        instruction: str,
        screen_info: ScreenInfo,
        max_cycles:  int,
    ) -> None:
        self.instruction = instruction
        self.screen_info = screen_info
        self.max_cycles  = max_cycles
        self._stop_flag  = False
        self._graph = build_agent_graph(
            omniparser = self.omniparser,
            planner    = self.planner,
            executor   = self.executor,
            agent_cfg  = self.cfg,
            signals    = self,
        )

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        if not self.screen_info or self._graph is None:
            self.task_completed.emit("Erreur : écran ou graphe non initialisé.")
            return

        self.log_message.emit(
            f"▶ Agent LangGraph (batch) — tâche : « {self.instruction} »", "info"
        )

        initial_state: dict = {
            "instruction":         self.instruction,
            "screen_info":         self.screen_info,
            "max_cycles":          self.max_cycles,
            "cycle":               0,
            "done":                False,
            "stop_requested":      False,
            "reflect_retries":     0,
            "raw_image":           None,
            "annotated_image":     None,
            "elements":            [],
            "plan":                None,
            "reflection":          None,
            "critique_for_replan": None,
            "history":             [],
            "last_batch_success":  True,
            "last_batch_msg":      "",
            "last_executed_actions": 0,
        }

        final_state: dict = initial_state

        try:
            for step_state in self._graph.stream(initial_state, stream_mode="values"):
                final_state = step_state
                if self._stop_flag:
                    self.log_message.emit("⏹ Arrêt manuel.", "warn")
                    self.task_completed.emit("Arrêt manuel (bouton Stop)")
                    return
        except Exception as exc:
            msg = f"Erreur graphe : {exc}"
            self.log_message.emit(msg, "error")
            self.task_completed.emit(msg)
            return

        reason = self._end_reason(final_state)
        self.log_message.emit(f"⏹ {reason}", "warn")
        self.task_completed.emit(reason)

    @staticmethod
    def _end_reason(state: dict) -> str:
        if state.get("done"):
            plan = state.get("plan")
            if plan and getattr(plan, "reasoning", ""):
                return plan.reasoning
            return state.get("last_batch_msg") or "Tâche accomplie"
        if state.get("cycle", 0) >= state.get("max_cycles", 10):
            return f"Max cycles atteint ({state.get('max_cycles')})"
        return state.get("last_batch_msg") or "Boucle terminée"


# ─── ParseOnceTask (inchangé) ─────────────────────────────────────────────────

class ParseOnceTask(QThread):
    screenshot_ready = pyqtSignal(object)
    elements_ready   = pyqtSignal(list)
    log_message      = pyqtSignal(str, str)
    finished_ok      = pyqtSignal()

    def __init__(self, omniparser, screen_info, parent=None):
        super().__init__(parent)
        self.omniparser  = omniparser
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