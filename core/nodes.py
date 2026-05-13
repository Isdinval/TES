"""
core/nodes.py
Nœuds du graphe LangGraph — version batch multi-actions.

Changement clé : execute_batch() parcourt TOUTES les ActionStep du BatchPlan
en une seule invocation du nœud. Le nœud ne rend la main au graphe qu'après :
  • avoir exécuté toutes les actions, OU
  • avoir rencontré un needs_screenshot=True (recapturer l'écran), OU
  • avoir rencontré un échec d'action.

Cela réduit drastiquement le nombre de cycles capture → parse.
"""
from __future__ import annotations
import time
from typing import Any, List

import pyautogui

from core.screen_capture import capture
from core.state          import AgentState
from core.llm_planner    import BatchPlan, ActionStep


# ─── Proxy signaux Qt ────────────────────────────────────────────────────────

class _Sig:
    __slots__ = ("_s",)

    def __init__(self, signals_obj: Any):
        self._s = signals_obj

    def _emit(self, name: str, *args) -> None:
        try:
            getattr(self._s, name).emit(*args)
        except Exception:
            pass

    def log(self, msg: str, level: str = "info") -> None:
        self._emit("log_message", msg, level)

    def screenshot(self, img: Any) -> None:
        self._emit("screenshot_ready", img)

    def elements(self, elems: list) -> None:
        self._emit("elements_ready", elems)

    def plan(self, p: Any) -> None:
        self._emit("plan_ready", p)

    def thinking(self, t: str) -> None:
        self._emit("llm_thinking", t)

    def action_done(self, msg: str, ok: bool) -> None:
        self._emit("action_done", msg, ok)

    def reflection(self, approved: bool, critique: str) -> None:
        self._emit("reflection_ready", approved, critique)

    def cycle(self, current: int, maxi: int) -> None:
        self._emit("cycle_updated", current, maxi)

    def batch_progress(self, step_idx: int, total: int, msg: str) -> None:
        self._emit("batch_progress", step_idx, total, msg)


# ─── node_capture ─────────────────────────────────────────────────────────────

def make_capture_node(agent_cfg: dict, signals: Any):
    sig = _Sig(signals)

    def capture_node(state: AgentState) -> dict:
        new_cycle = state["cycle"] + 1
        sig.cycle(new_cycle, state["max_cycles"])
        sig.log(f"── Cycle {new_cycle}/{state['max_cycles']} ──────────────", "info")

        if new_cycle == 1:
            si = state["screen_info"]
            cx = si.left + si.width  // 2
            cy = si.top  + si.height // 2
            sig.log("🖱️  Clic de focus initial…", "info")
            pyautogui.click(cx, cy)
            time.sleep(0.4)

        try:
            pil_img = capture(state["screen_info"])
            sig.log("📸 Screenshot capturé", "info")
        except Exception as exc:
            sig.log(f"Erreur screenshot : {exc}", "error")
            return {"cycle": new_cycle, "done": True,
                    "last_batch_msg": str(exc)}

        return {
            "cycle":               new_cycle,
            "raw_image":           pil_img,
            "reflect_retries":     0,
            "critique_for_replan": None,
        }

    return capture_node


# ─── node_parse ───────────────────────────────────────────────────────────────

def make_parse_node(omniparser: Any, signals: Any):
    sig = _Sig(signals)

    def parse_node(state: AgentState) -> dict:
        if state.get("done"):
            return {}

        sig.log("🔍 Analyse OmniParser…", "info")
        try:
            annotated, elements = omniparser.parse(state["raw_image"])
            sig.screenshot(annotated)
            sig.elements(elements)
            sig.log(f"✓ {len(elements)} éléments détectés", "ok")

            if not elements:
                sig.log("⚠ Aucun élément — arrêt.", "warn")
                return {"annotated_image": annotated, "elements": [],
                        "done": True, "last_batch_msg": "Aucun élément UI"}

            return {"annotated_image": annotated, "elements": elements}

        except Exception as exc:
            sig.log(f"Erreur OmniParser : {exc}", "error")
            return {"elements": [], "done": True, "last_batch_msg": str(exc)}

    return parse_node


# ─── node_plan ────────────────────────────────────────────────────────────────

def make_plan_node(planner: Any, signals: Any):
    sig = _Sig(signals)

    def plan_node(state: AgentState) -> dict:
        if state.get("done"):
            return {}

        critique = state.get("critique_for_replan")
        if critique:
            sig.log(f"↺ Re-plan (essai {state.get('reflect_retries',1)}) — {critique[:80]}…", "warn")
        else:
            sig.log("🧠 Planification LLM (batch)…", "info")

        try:
            batch = planner.plan(
                instruction = state["instruction"],
                elements    = state["elements"],
                history     = state["history"],
                critique    = critique,
            )
            sig.plan(batch)
            if batch.thinking:
                sig.thinking(batch.thinking)

            n = len(batch.actions)
            sig.log(
                f"→ Batch de {n} action(s) planifiée(s) — {batch.reasoning[:80]}",
                "ok",
            )
            for i, a in enumerate(batch.actions, 1):
                sig.log(
                    f"   [{i}/{n}] {a.action_type}"
                    f"(id={a.element_id}, val={a.value!r})"
                    f"{'  📸' if a.needs_screenshot else ''}",
                    "info",
                )

            return {"plan": batch, "critique_for_replan": None}

        except Exception as exc:
            sig.log(f"Erreur LLM : {exc}", "error")
            return {"done": True, "last_batch_msg": str(exc), "plan": None}

    return plan_node


# ─── node_reflect ─────────────────────────────────────────────────────────────

def make_reflect_node(reflector: Any, signals: Any):
    sig = _Sig(signals)

    def reflect_node(state: AgentState) -> dict:
        if state.get("done"):
            return {}

        plan: BatchPlan = state.get("plan")
        if plan is None or plan.is_empty:
            if plan and plan.done:
                return {
                    "reflection": {
                        "approved": True,
                        "critique": "done=true avec batch vide → tâche terminée.",
                        "suggested_actions": None,
                    }
                }
            return {
                "reflection": {
                    "approved": False,
                    "critique": "Batch vide ou absent.",
                    "suggested_actions": None,
                }
            }

        # Tâche déclarée terminée → skip critique
        if plan.done:
            return {
                "reflection": {
                    "approved":          True,
                    "critique":          "done=true — critique inutile.",
                    "suggested_actions": None,
                }
            }

        sig.log("🔎 Auto-critique de la séquence…", "info")
        result = reflector.reflect(
            instruction = state["instruction"],
            elements    = state["elements"],
            history     = state["history"],
            plan        = plan,
        )
        verdict = "✅" if result["approved"] else "❌"
        sig.log(f"Réflexion {verdict} — {result['critique'][:140]}",
                "ok" if result["approved"] else "warn")
        sig.reflection(result["approved"], result["critique"])

        return {"reflection": result}

    return reflect_node


# ─── node_execute_batch ───────────────────────────────────────────────────────

def make_execute_batch_node(executor: Any, agent_cfg: dict, signals: Any):
    """
    Nœud principal : exécute TOUTES les ActionStep du BatchPlan séquentiellement.

    Interruptions possibles :
      • step.needs_screenshot=True → arrête le batch, recapture l'écran
      • Échec d'une action         → arrête le batch, recapture
      • plan.done=True             → marque la tâche comme terminée
    """
    sig        = _Sig(signals)
    step_delay = agent_cfg.get("step_delay",        0.3)   # délai entre actions du batch
    post_delay = agent_cfg.get("post_action_delay",  1.0)   # délai après le batch complet

    def execute_batch_node(state: AgentState) -> dict:
        plan: BatchPlan = state.get("plan")

        if plan is None:
            return {"last_batch_success": False,
                    "last_batch_msg":     "Plan absent"}

        # ── Tâche déclarée terminée (batch vide ou done direct) ───────────
        if plan.done and not plan.actions:
            reason = plan.reasoning or "Tâche accomplie"
            sig.log(f"🏁 Tâche terminée : {reason}", "ok")
            return {"done": True, "last_batch_success": True,
                    "last_batch_msg": reason, "last_executed_actions": 0}

        elements    = state["elements"]
        screen_info = state["screen_info"]
        history_entries: List[dict] = []
        executed    = 0
        total       = len(plan.actions)

        sig.log(f"▶ Exécution du batch : {total} action(s)…", "info")

        for idx, step in enumerate(plan.actions):
            sig.batch_progress(idx + 1, total,
                               f"{step.action_type}(id={step.element_id})")

            # ── Exécution d'une action atomique ──────────────────────────
            success, msg = _execute_step(executor, step, elements, screen_info)
            executed += 1

            sig.action_done(msg, success)
            sig.log(
                f"  {'✓' if success else '✗'} [{idx+1}/{total}] {msg}",
                "ok" if success else "error",
            )

            hist: dict = {
                "step":       state["cycle"],
                "batch_idx":  idx + 1,
                "action":     step.action_type,
                "element_id": step.element_id,
                "value":      step.value,
                "reasoning":  step.reasoning,
                "success":    success,
            }
            if step.action_type == "type" and step.value:
                hist["expect_ocr"]  = step.value
                hist["verify_hint"] = (
                    f"Cycle suivant : cherche '{step.value}' dans ocr_text."
                )
            history_entries.append(hist)

            # ── Interruption du batch ─────────────────────────────────────
            if not success:
                sig.log(f"⚠ Batch interrompu à l'étape {idx+1}/{total} — recapture.", "warn")
                time.sleep(post_delay)
                return {
                    "last_batch_success":    False,
                    "last_batch_msg":        msg,
                    "last_executed_actions": executed,
                    "history":               history_entries,
                }

            if step.needs_screenshot and idx < total - 1:
                sig.log(f"📸 needs_screenshot — recapture après étape {idx+1}.", "info")
                time.sleep(post_delay)
                # On enregistre les actions exécutées jusqu'ici et on repart
                return {
                    "last_batch_success":    True,
                    "last_batch_msg":        f"Batch partiel ({executed}/{total}) — recapture demandée",
                    "last_executed_actions": executed,
                    "history":               history_entries,
                }

            # Pause inter-actions (plus courte que le post_delay)
            if idx < total - 1:
                time.sleep(step_delay)

        # ── Batch complet ────────────────────────────────────────────────
        if plan.done:
            sig.log(f"🏁 Tâche terminée après le batch : {plan.reasoning}", "ok")
            time.sleep(post_delay)
            return {
                "done":                  True,
                "last_batch_success":    True,
                "last_batch_msg":        plan.reasoning or "Tâche accomplie",
                "last_executed_actions": executed,
                "history":               history_entries,
            }

        sig.log(f"✓ Batch terminé ({executed} actions) — recapture.", "ok")
        time.sleep(post_delay)
        return {
            "last_batch_success":    True,
            "last_batch_msg":        f"{executed} action(s) exécutée(s)",
            "last_executed_actions": executed,
            "history":               history_entries,
        }

    return execute_batch_node


# ─── Exécution d'une ActionStep via ActionExecutor ───────────────────────────

def _execute_step(executor, step: ActionStep, elements, screen_info):
    """
    Adapte une ActionStep vers l'interface de ActionExecutor.
    ActionExecutor attend un ActionPlan-like (duck typing).
    On utilise un objet proxy léger.
    """
    proxy = _StepProxy(step)
    return executor.execute(proxy, elements, screen_info)


class _StepProxy:
    """
    Duck-type proxy : présente une ActionStep comme un ActionPlan
    pour que ActionExecutor n'ait pas besoin d'être modifié.
    """
    __slots__ = (
        "action_type", "element_id", "value",
        "scroll_direction", "scroll_amount",
        "drag_to_element_id", "done", "reasoning",
    )

    def __init__(self, step: ActionStep):
        self.action_type        = step.action_type
        self.element_id         = step.element_id
        self.value              = step.value
        self.scroll_direction   = step.scroll_direction
        self.scroll_amount      = step.scroll_amount
        self.drag_to_element_id = step.drag_to_element_id
        self.done               = False   # géré au niveau BatchPlan
        self.reasoning          = step.reasoning