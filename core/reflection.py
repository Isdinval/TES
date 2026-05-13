"""
core/reflection.py
Auto-critique (Reflection) de la SÉQUENCE d'actions (BatchPlan).

Le critic Ollama évalue :
  1. Chaque element_id existe dans la liste et a le bon subtype
  2. La séquence est cohérente et fait progresser la tâche
  3. Pas de répétition stérile vs l'historique
  4. needs_screenshot est bien positionné (après les clics qui changent l'écran)
  5. Le nombre d'actions est raisonnable (ni trop peu, ni hallucination)

En cas d'erreur → fail-open (approuve silencieusement).
"""
from __future__ import annotations
import json
import re
import requests
from typing import List, Optional

from core.omniparser_bridge import UIElement
from core.llm_planner       import BatchPlan

_SYSTEM = """\
Tu es un validateur de séquences d'actions GUI.
Tu reçois une LISTE D'ACTIONS planifiée par un agent automatisé.

━━━ CRITÈRES DE VALIDATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R1. Chaque element_id référencé doit exister dans la liste d'éléments.
R2. action "type"  → l'élément doit avoir subtype "text_input".
R3. action "click" → l'élément doit être is_interactive=true.
R4. Pas de "scroll" sans element_id pour naviguer (utiliser "press_key" pagedown).
R5. La séquence doit progresser vers la tâche, sans répétition inutile.
R6. needs_screenshot doit être true après les clics qui changent l'écran
    (boutons de navigation, soumission de formulaire, ouverture de menu).
R7. done=true seulement si la tâche est visuellement confirmée dans les OCR.

━━━ FORMAT RÉPONSE (JSON strict) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "approved": true | false,
  "critique": "<explication concise>",
  "suggested_actions": [   // optionnel : séquence corrigée si approved=false
    {"action_type": "...", "element_id": ..., "value": "...",
     "reasoning": "...", "needs_screenshot": false}
  ]
}
Si approved=true, suggested_actions peut être null ou [].
"""

_USER = """\
TÂCHE : {instruction}

ÉLÉMENTS DÉTECTÉS ({n} éléments) :
{elements_json}

HISTORIQUE ({h} étapes) :
{history_json}

SÉQUENCE D'ACTIONS SOUMISE :
{actions_json}

done déclaré : {done}

→ Valide ou corrige cette séquence.
"""


class ReflectionEngine:

    def __init__(self, ollama_config: dict):
        self.base_url = ollama_config['base_url'].rstrip('/')
        self.model    = ollama_config['model']
        self.timeout  = ollama_config.get('timeout', 60)

    def reflect(
        self,
        instruction: str,
        elements:    List[UIElement],
        history:     List[dict],
        plan:        BatchPlan,
    ) -> dict:
        """Retourne toujours un dict ReflectionResult (jamais d'exception)."""
        try:
            prompt = self._build_prompt(instruction, elements, history, plan)
            raw    = self._call_ollama(prompt)
            return self._parse(raw)
        except Exception as exc:
            return {
                "approved":          True,
                "critique":          f"Réflexion indisponible ({exc}) — plan approuvé.",
                "suggested_actions": None,
            }

    def _build_prompt(
        self,
        instruction: str,
        elements:    List[UIElement],
        history:     List[dict],
        plan:        BatchPlan,
    ) -> str:
        actions_json = json.dumps(
            [
                {
                    "action_type":      s.action_type,
                    "element_id":       s.element_id,
                    "value":            s.value,
                    "reasoning":        s.reasoning,
                    "needs_screenshot": s.needs_screenshot,
                }
                for s in plan.actions
            ],
            ensure_ascii=False, indent=2,
        )
        elems_light = [
            {
                "id":             e.id,
                "subtype":        e.subtype,
                "is_interactive": e.is_interactive,
                "description":    e.description,
                "ocr_text":       e.ocr_text,
            }
            for e in elements
        ]
        return _USER.format(
            instruction   = instruction,
            n             = len(elements),
            elements_json = json.dumps(elems_light, ensure_ascii=False, indent=2),
            h             = len(history),
            history_json  = json.dumps(history[-6:], ensure_ascii=False, indent=2)
                            if history else "[]",
            actions_json  = actions_json,
            done          = plan.done,
        )

    def _call_ollama(self, user_prompt: str) -> str:
        payload = {
            "model":  self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1024},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    @staticmethod
    def _parse(raw: str) -> dict:
        text  = raw.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"approved": True, "critique": "Parse error — approuvé.", "suggested_actions": None}

        return {
            "approved":          bool(data.get("approved", True)),
            "critique":          str(data.get("critique", "")),
            "suggested_actions": data.get("suggested_actions") or None,
        }