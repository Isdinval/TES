"""
core/graph.py
Construction et compilation du graphe LangGraph — version batch multi-actions.

━━━ Topologie ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    START
      │
      ▼
   capture  ◄─────────────────────────────────────────────────────┐
      │                                                            │
      ▼                                                            │
    parse                                             (cycle suivant)
      │                                                            │
      ▼                                                            │
     plan  ◄──────────────────────────────────┐                   │
      │                                        │                   │
      ▼                                        │ replan            │
   reflect ──[❌ rejeté, retries<MAX]──► replan_node              │
      │                                                            │
      │ [✅ approuvé]                                              │
      ▼                                                            │
  execute_batch ──────────────────────────────────────────────────►┤
      │                                                            │
      ├──[done / stop / max_cycles]──► END                        │
      └──[continuer]────────────────────────────────────────────► ─┘

━━━ Gain vs v1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  V1 : 1 screenshot → 1 action → 1 screenshot  (N actions = N cycles)
  V2 : 1 screenshot → N actions → 1 screenshot (N actions = 1 cycle)
       Exemple : remplir un formulaire 5 champs → 1 seul cycle au lieu de 10+
"""
from __future__ import annotations
from typing import Any, Literal

from langgraph.graph import StateGraph, END

from core.state      import AgentState
from core.reflection import ReflectionEngine
from core.nodes      import (
    make_capture_node,
    make_parse_node,
    make_plan_node,
    make_reflect_node,
    make_execute_batch_node,
)

MAX_REFLECT_RETRIES: int = 2


# ─── Routeurs ─────────────────────────────────────────────────────────────────

def _route_after_reflect(state: AgentState) -> Literal["execute_batch", "replan"]:
    reflection = state.get("reflection") or {}
    approved   = reflection.get("approved", True)
    retries    = state.get("reflect_retries", 0)

    if approved or retries >= MAX_REFLECT_RETRIES:
        return "execute_batch"
    return "replan"


def _route_after_execute(state: AgentState) -> str:
    if state.get("done"):
        return END
    if state.get("stop_requested"):
        return END
    if state.get("cycle", 0) >= state.get("max_cycles", 10):
        return END
    if not state.get("elements"):
        return END
    return "capture"


# ─── Nœud replan ─────────────────────────────────────────────────────────────

def _replan_node(state: AgentState) -> dict:
    """
    Transforme le verdict de reflection en critique textuelle
    pour le prochain appel LLM. Incrémente le compteur de retries.
    """
    reflection = state.get("reflection") or {}
    critique   = reflection.get("critique", "Séquence invalide.")

    # Si le critic a proposé une séquence corrigée, on la mentionne
    suggested  = reflection.get("suggested_actions")
    if suggested:
        import json
        try:
            critique += f"\nSéquence suggérée : {json.dumps(suggested, ensure_ascii=False)}"
        except Exception:
            pass

    return {
        "critique_for_replan": critique,
        "reflect_retries":     state.get("reflect_retries", 0) + 1,
    }


# ─── Factory principale ───────────────────────────────────────────────────────

def build_agent_graph(
    omniparser: Any,
    planner:    Any,
    executor:   Any,
    agent_cfg:  dict,
    signals:    Any,
) -> Any:
    reflector = ReflectionEngine(planner.config)

    capture_node       = make_capture_node(agent_cfg, signals)
    parse_node         = make_parse_node(omniparser, signals)
    plan_node          = make_plan_node(planner, signals)
    reflect_node       = make_reflect_node(reflector, signals)
    execute_batch_node = make_execute_batch_node(executor, agent_cfg, signals)

    g = StateGraph(AgentState)

    g.add_node("capture",       capture_node)
    g.add_node("parse",         parse_node)
    g.add_node("plan",          plan_node)
    g.add_node("reflect",       reflect_node)
    g.add_node("replan",        _replan_node)
    g.add_node("execute_batch", execute_batch_node)

    g.set_entry_point("capture")

    g.add_edge("capture",  "parse")
    g.add_edge("parse",    "plan")
    g.add_edge("plan",     "reflect")
    g.add_edge("replan",   "plan")

    g.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"execute_batch": "execute_batch", "replan": "replan"},
    )
    g.add_conditional_edges(
        "execute_batch",
        _route_after_execute,
        {"capture": "capture", END: END},
    )

    return g.compile()