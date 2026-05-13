"""
core/state.py
État partagé du graphe LangGraph — version batch multi-actions.

Le champ `plan` est désormais un BatchPlan contenant une Liste d'ActionStep.
Le nœud execute_batch les exécute TOUTES séquentiellement avant de
revenir capturer un nouveau screenshot → drastiquement moins de cycles.
"""
from __future__ import annotations
import operator
from typing import TypedDict, List, Optional, Any, Annotated


class ReflectionResult(TypedDict):
    """Verdict de l'auto-critique sur la séquence d'actions planifiée."""
    approved:          bool
    critique:          str
    suggested_actions: Optional[List[dict]]   # séquence corrigée si rejeté


class AgentState(TypedDict):
    """
    État global traversant tous les nœuds du graphe.

    `history` est annoté operator.add : chaque nœud renvoie
    {"history": [entries…]} et LangGraph les accumule automatiquement.
    """

    # ── Tâche ──────────────────────────────────────────────────────────────
    instruction:  str
    screen_info:  Any        # core.screen_capture.ScreenInfo
    max_cycles:   int

    # ── Contrôle de boucle ─────────────────────────────────────────────────
    cycle:           int
    done:            bool
    stop_requested:  bool
    reflect_retries: int     # re-plans tentés dans le cycle courant

    # ── Perception ─────────────────────────────────────────────────────────
    raw_image:       Optional[Any]   # PIL.Image brut
    annotated_image: Optional[Any]   # PIL.Image annoté OmniParser
    elements:        List[Any]       # List[UIElement]

    # ── Cognition ──────────────────────────────────────────────────────────
    plan:                Optional[Any]              # BatchPlan
    reflection:          Optional[ReflectionResult]
    critique_for_replan: Optional[str]

    # ── Mémoire ────────────────────────────────────────────────────────────
    history: Annotated[List[dict], operator.add]

    # ── Résultat du dernier batch ──────────────────────────────────────────
    last_batch_success:    bool
    last_batch_msg:        str
    last_executed_actions: int   # combien d'actions ont tourné dans le batch