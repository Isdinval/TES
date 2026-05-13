"""
core/action_executor.py
Exécution des actions pyautogui sur l'écran cible.
Remapping coordonnées normalisées [0,1] → pixels logiques écran.
"""
from __future__ import annotations
import time
import pyautogui
from typing import List, Optional, Tuple

from core.screen_capture import ScreenInfo
from core.omniparser_bridge import UIElement
from core.llm_planner import ActionPlan

# Sécurité pyautogui : désactive le failsafe coin supérieur gauche
# (tu peux le réactiver en retirant cette ligne)
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.0     # on gère nos propres délais


class ActionExecutor:
    def __init__(self, config: dict):
        self.type_interval  = config.get('type_interval',  0.04)
        self.drag_duration  = config.get('drag_duration',  0.5)
        self.scroll_amount  = config.get('scroll_amount',  3)

    # ── API publique ──────────────────────────────────────────────────────────

    def execute(
        self,
        plan:        ActionPlan,
        elements:    List[UIElement],
        screen_info: ScreenInfo,
    ) -> Tuple[bool, str]:
        """
        Exécute l'action décrite dans plan.
        Retourne (succès: bool, message: str).
        """
        if plan.action_type == "none" or plan.done:
            return True, "Aucune action (tâche terminée)"

        elem = self._find_element(plan.element_id, elements)

        try:
            match plan.action_type:
                case "click":
                    x, y = self._input_coords(elem, screen_info)
                    self._click(x, y)
                    return True, f"click({x}, {y})  [id={plan.element_id}]"

                case "double_click":
                    x, y = self._input_coords(elem, screen_info)
                    self._double_click(x, y)
                    return True, f"double_click({x}, {y})  [id={plan.element_id}]"

                case "right_click":
                    x, y = self._coords(elem, screen_info)
                    self._right_click(x, y)
                    return True, f"right_click({x}, {y})  [id={plan.element_id}]"

                case "type":
                    x, y = self._input_coords(elem, screen_info)
                    self._type_text(x, y, plan.value or "")
                    return True, f"type('{plan.value}')  [id={plan.element_id}]"

                case "press_key":
                    key = plan.value or "enter"
                    self._press_key(key)
                    return True, f"press_key('{key}')"

                case "scroll":
                    direction = plan.scroll_direction or "down"
                    if elem is None:
                        # Pas d'élément ciblé → Page Down / Page Up, bien plus efficace
                        key = "pagedown" if direction == "down" else "pageup"
                        repeat = max(1, plan.scroll_amount // 3)
                        for _ in range(repeat):
                            pyautogui.press(key)
                            time.sleep(0.05)
                        return True, f"press_key('{key}' x{repeat})  [navigation page]"
                    else:
                        # Élément ciblé → scroll molette classique sur cet élément
                        x, y  = self._coords(elem, screen_info)
                        amount = plan.scroll_amount or self.scroll_amount
                        self._scroll(x, y, direction, amount * 5)   # ×5 pour que ça bouge
                        return True, f"scroll({direction}, {amount*5})  [id={plan.element_id}]"

                case "drag_drop":
                    if not elem:
                        return False, "drag_drop : element source introuvable"
                    target = self._find_element(plan.drag_to_element_id, elements)
                    if not target:
                        return False, "drag_drop : element cible introuvable"
                    x1, y1 = self._coords(elem,   screen_info)
                    x2, y2 = self._coords(target, screen_info)
                    self._drag_drop(x1, y1, x2, y2)
                    return True, f"drag_drop({x1},{y1} → {x2},{y2})"

                case _:
                    return False, f"action inconnue : {plan.action_type}"

        except Exception as e:
            return False, f"Erreur exécution {plan.action_type} : {e}"

    # ── Actions pyautogui ─────────────────────────────────────────────────────

    def _click(self, x: int, y: int) -> None:
        pyautogui.click(x, y)

    def _double_click(self, x: int, y: int) -> None:
        pyautogui.doubleClick(x, y)

    def _right_click(self, x: int, y: int) -> None:
        pyautogui.rightClick(x, y)

    def _type_text(self, x: int, y: int, text: str) -> None:
        pyautogui.click(x, y)
        time.sleep(0.15)
        pyautogui.write(text, interval=self.type_interval)

    def _press_key(self, key_str: str) -> None:
        """
        Supporte les raccourcis : "ctrl+a", "ctrl+shift+t", "enter", "tab"…
        """
        parts = [k.strip().lower() for k in key_str.split('+')]
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)

    def _scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        clicks = amount if direction == "up" else -amount
        pyautogui.scroll(clicks, x=x, y=y)

    def _drag_drop(self, x1: int, y1: int, x2: int, y2: int) -> None:
        pyautogui.moveTo(x1, y1, duration=0.2)
        pyautogui.dragTo(x2, y2, duration=self.drag_duration, button='left')

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_element(
        element_id: Optional[int], elements: List[UIElement]
    ) -> Optional[UIElement]:
        if element_id is None:
            return None
        for e in elements:
            if e.id == element_id:
                return e
        return None

    @staticmethod
    def _coords(
        elem: Optional[UIElement], screen_info: ScreenInfo
    ) -> Tuple[int, int]:
        if elem is None:
            return ActionExecutor._screen_center(screen_info)
        return screen_info.norm_to_px(*elem.center_norm)

    @staticmethod
    def _input_coords(
        elem: Optional[UIElement], screen_info: ScreenInfo
    ) -> Tuple[int, int]:
        """
        Utilise click_target de l'élément (= input_center_norm si le classifier
        l'a localisé précisément via Florence-2 OVD, sinon center_norm classique).
        Zéro heuristique pixel ici — la logique est dans UIClassifier.
        """
        if elem is None:
            return ActionExecutor._screen_center(screen_info)
        return screen_info.norm_to_px(*elem.click_target)

    @staticmethod
    def _screen_center(screen_info: ScreenInfo) -> Tuple[int, int]:
        return (
            screen_info.left + screen_info.width  // 2,
            screen_info.top  + screen_info.height // 2,
        )