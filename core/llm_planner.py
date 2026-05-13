"""
core/llm_planner.py
Planificateur LLM via Ollama — version batch multi-actions.

Le LLM produit désormais une SÉQUENCE d'actions (BatchPlan) à partir
d'un seul screenshot. L'objectif est de réduire drastiquement le nombre
de cycles capture → parse en groupant les actions logiquement liées.

Exemple : remplir un formulaire complet en un seul plan :
  [click champ_nom, type "Jean", tab, type "Dupont", tab,
   click champ_email, type "jean@test.fr", click bouton_envoyer]

Le LLM est guidé pour :
  • Grouper les actions séquentielles évidentes (saisie champ par champ)
  • Limiter à MAX_ACTIONS_PER_BATCH actions par plan (évite l'hallucination)
  • Indiquer done=true dans la DERNIÈRE action si la tâche est accomplie
  • Indiquer needs_screenshot=true si un screenshot intermédiaire est nécessaire
    (ex : après un clic qui ouvre un menu → on recapture avant de continuer)
"""
from __future__ import annotations
import json
import re
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Any

from core.omniparser_bridge import UIElement

MAX_ACTIONS_PER_BATCH = 8   # plafond de sécurité


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class ActionStep:
    """Une action atomique dans le plan batch."""
    action_type:        str           = "none"
    element_id:         Optional[int] = None
    value:              Optional[str] = None
    scroll_direction:   Optional[str] = None
    scroll_amount:      int           = 3
    drag_to_element_id: Optional[int] = None
    reasoning:          str           = ""
    # Si True → interrompt le batch APRÈS cette action pour recapturer l'écran
    needs_screenshot:   bool          = False


@dataclass
class BatchPlan:
    """Séquence d'actions produite par un seul appel LLM."""
    thinking:     str              = ""
    actions:      List[ActionStep] = field(default_factory=list)
    done:         bool             = False   # True → tâche globale terminée
    reasoning:    str              = ""
    raw_response: str              = field(default="", repr=False)

    @property
    def is_empty(self) -> bool:
        return not self.actions

    def __repr__(self) -> str:
        acts = [f"{a.action_type}(id={a.element_id}, val={a.value!r})"
                for a in self.actions]
        return f"BatchPlan(done={self.done}, actions={acts})"


VALID_ACTIONS = {
    "click", "double_click", "right_click",
    "type", "press_key",
    "scroll", "drag_drop",
    "none",
}


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
Tu es un agent d'automatisation GUI local. Tu analyses des éléments d'interface \
détectés sur un écran par OmniParser (YOLO + Florence-2 + OCR) et tu planifies \
une SÉQUENCE D'ACTIONS pour accomplir une tâche donnée.

━━━ RÈGLES ABSOLUES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Réponds UNIQUEMENT en JSON valide. Aucun texte avant ou après.
2. Tu peux planifier entre 1 et {MAX_ACTIONS_PER_BATCH} actions par réponse.
3. Utilise UNIQUEMENT des element_id présents dans la liste fournie.
4. Groupe les actions logiquement liées en une seule séquence.
5. Pour naviguer dans une page → press_key "pagedown"/"pageup" (JAMAIS scroll sans cible).
6. Si la tâche est terminée → mets "done": true dans le JSON racine.

- PRIORITÉ MAXIMALE : Ne clique QUE sur des éléments dont le subtype est "text_input", "button", "icon_button", "checkbox", ou dont le ocr_text/description contient clairement le label de la tâche (ex: "Nom", "XL", "Envoyer", "Commentaires", "Submit").
- Ignore complètement les éléments avec subtype "icon", "image", "unknown" ou dont le label ressemble à des icônes Windows (Explorateur, Chrome, barre des tâches).
- Pour les champs de formulaire, utilise toujours "click" + "type" sur le même element_id (text_input).
- Si un champ important n'est pas visible → utilise press_key "pagedown" ou "pageup" **une seule fois**, puis attends le nouveau screenshot. Ne fais jamais plus de 2 scrolls consécutifs sans nouvelle analyse.
- Vérifie l'historique : si un champ a déjà été rempli avec succès (via OCR dans l'historique), ne le refais pas.

- Si tu as la preuve visuelle (via OCR ou éléments) que le formulaire est soumis (message 'enregistrée', 'submitted', bouton 'Envoyer une autre réponse', etc.), mets 'done': true et renvoie un batch vide (actions: []).

━━━ STRATÉGIE DE GROUPAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Remplir plusieurs champs visibles → UNE séquence [click, type, tab, click, type…]
• Cocher plusieurs cases → UNE séquence [click, click, click]
• Clic + saisie dans le même champ → [click, type] groupés
• Clic qui ouvre un menu/modal inconnu → UNE seule action + "needs_screenshot": true
• Pagedown pour révéler des champs → autant de pagedown que nécessaire groupés

━━━ FORMAT DE RÉPONSE OBLIGATOIRE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "thinking": "<raisonnement interne : analyse l'écran, liste les champs/boutons visibles, décide la séquence optimale>",
  "actions": [
    {{
      "action_type": "<click|double_click|right_click|type|press_key|scroll|drag_drop|none>",
      "element_id": <entier ou null>,
      "value": "<texte ou touche ou null>",
      "scroll_direction": "<up|down|null>",
      "scroll_amount": <entier, défaut 3>,
      "drag_to_element_id": <entier ou null>,
      "reasoning": "<pourquoi cette action>",
      "needs_screenshot": <true si l'action change radicalement l'écran, false sinon>
    }}
    // ... jusqu'à {MAX_ACTIONS_PER_BATCH} actions
  ],
  "done": <true si la tâche est ENTIÈREMENT accomplie APRÈS toutes ces actions, false sinon>,
  "reasoning": "<synthèse du plan>"
}}

━━━ TYPES D'ACTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- click / double_click / right_click  → element_id requis
- type          → saisir 'value' dans element_id (subtype text_input attendu)
- press_key     → "pagedown", "pageup", "enter", "tab", "escape", "ctrl+a"…
- scroll        → molette sur un element_id précis (liste déroulante, etc.)
- drag_drop     → glisser element_id vers drag_to_element_id

━━━ EXEMPLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tâche "Remplis Prénom=Jean, Nom=Dupont et clique Valider"
Éléments visibles : id=1 text_input "Prénom", id=2 text_input "Nom", id=5 button "Valider"
→ {{
  "thinking": "Je vois les champs Prénom (1) et Nom (2) et le bouton Valider (5). Je peux tout faire en une seule séquence.",
  "actions": [
    {{"action_type":"click",     "element_id":1, "value":null, "reasoning":"focus champ Prénom", "needs_screenshot":false}},
    {{"action_type":"type",      "element_id":1, "value":"Jean", "reasoning":"saisie prénom", "needs_screenshot":false}},
    {{"action_type":"press_key", "element_id":null, "value":"tab", "reasoning":"passer au champ suivant", "needs_screenshot":false}},
    {{"action_type":"type",      "element_id":2, "value":"Dupont", "reasoning":"saisie nom", "needs_screenshot":false}},
    {{"action_type":"click",     "element_id":5, "value":null, "reasoning":"valider le formulaire", "needs_screenshot":true}}
  ],
  "done": false,
  "reasoning": "Remplissage complet du formulaire en une passe"
}}

Tâche "Va à la page suivante"
Éléments : id=3 button "Suivant"
→ {{
  "thinking": "Bouton Suivant visible, je clique dessus. Ça changera l'écran donc needs_screenshot.",
  "actions": [
    {{"action_type":"click","element_id":3,"value":null,"reasoning":"clic Suivant","needs_screenshot":true}}
  ],
  "done": false,
  "reasoning": "Navigation vers la page suivante"
}}

━━━ RÈGLE DE VÉRIFICATION done=true ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ne mets done=true QUE si tu as la PREUVE visuelle (dans les éléments OCR actuels)
que la tâche est accomplie. Après une action "type", attends la confirmation OCR
au cycle suivant avant de déclarer done.
"""

USER_TEMPLATE = """\
TÂCHE : {instruction}

ÉLÉMENTS DÉTECTÉS ({n_elements} éléments) :
{elements_json}

HISTORIQUE ({n_history} étapes exécutées) :
{history_json}

{critique_section}

Planifie la meilleure SÉQUENCE D'ACTIONS pour progresser vers la tâche.
Groupe un maximum d'actions visibles et logiquement enchaînables.
"""

_CRITIQUE_SECTION = """\
⚠️  RE-PLANIFICATION — séquence précédente rejetée par le validateur :
{critique}

Corrige ta séquence en tenant compte de ce retour.
"""


# ─── LLMPlanner ──────────────────────────────────────────────────────────────

class LLMPlanner:
    def __init__(self, config: dict):
        self.config   = config
        self.base_url = config['base_url'].rstrip('/')
        self.model    = config['model']
        self.timeout  = config['timeout']
        self.temp     = config['temperature']
        self.max_tok  = config['max_tokens']

    # ── API publique ──────────────────────────────────────────────────────────

    def plan(
        self,
        instruction: str,
        elements:    List[UIElement],
        history:     List[dict],
        critique:    Optional[str] = None,
    ) -> BatchPlan:
        """
        Appelle Ollama et retourne un BatchPlan (séquence d'actions).
        Lève une exception si l'appel réseau échoue.
        """
        prompt = self._build_prompt(instruction, elements, history, critique)
        raw    = self._call_ollama(prompt)
        plan   = self._parse(raw)
        plan.raw_response = raw
        return plan

    # ── Construction du prompt ────────────────────────────────────────────────

    def _build_prompt(
        self,
        instruction: str,
        elements:    List[UIElement],
        history:     List[dict],
        critique:    Optional[str],
    ) -> str:
        # Historique condensé : on garde les 10 dernières entrées max
        history_short = history[-10:] if len(history) > 10 else history
        return USER_TEMPLATE.format(
            instruction      = instruction,
            n_elements       = len(elements),
            elements_json    = json.dumps(
                [e.to_llm_dict() for e in elements],
                ensure_ascii=False, indent=2,
            ),
            n_history        = len(history),
            history_json     = json.dumps(history_short, ensure_ascii=False, indent=2)
                               if history_short else "[]",
            critique_section = _CRITIQUE_SECTION.format(critique=critique)
                               if critique else "",
        )

    # ── Appel Ollama ──────────────────────────────────────────────────────────

    def _call_ollama(self, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temp,
                "num_predict": self.max_tok,
            },
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, raw: str) -> BatchPlan:
        text  = raw.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return BatchPlan(
                thinking  = "Erreur parsing JSON",
                actions   = [],
                done      = False,
                reasoning = f"Réponse non parsable : {raw[:200]}",
            )

        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []

        # Plafond de sécurité
        raw_actions = raw_actions[:MAX_ACTIONS_PER_BATCH]

        steps: List[ActionStep] = []
        for a in raw_actions:
            if not isinstance(a, dict):
                continue
            atype = str(a.get("action_type", "none")).lower()
            if atype not in VALID_ACTIONS:
                atype = "none"
            steps.append(ActionStep(
                action_type        = atype,
                element_id         = _to_int(a.get("element_id")),
                value              = _to_str(a.get("value")),
                scroll_direction   = _to_str(a.get("scroll_direction")),
                scroll_amount      = int(a.get("scroll_amount") or 3),
                drag_to_element_id = _to_int(a.get("drag_to_element_id")),
                reasoning          = str(a.get("reasoning", "")),
                needs_screenshot   = bool(a.get("needs_screenshot", False)),
            ))

        return BatchPlan(
            thinking  = str(data.get("thinking", "")),
            actions   = steps,
            done      = bool(data.get("done", False)),
            reasoning = str(data.get("reasoning", "")),
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_str(v: Any) -> Optional[str]:
    if v is None or str(v).lower() in ("null", "none", ""):
        return None
    return str(v)
    
    
    
ActionPlan = BatchPlan   # backward compatibility for UI