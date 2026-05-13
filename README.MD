# GUI Agent — OmniParser V2 + Ollama (Batch Mode)

**Agent d’automatisation GUI local 100 % offline** utilisant OmniParser V2 (YOLO + Florence-2 + OCR) + Ollama + LangGraph + PyQt6.

**Version actuelle : Batch Multi-Actions** (1 screenshot → N actions logiquement groupées).


---

## Architecture

```
gui_agent/
├── main.py                    ← Point d'entrée
├── config.py                  ← Tous les paramètres (chemins, modèles, délais)
├── requirements.txt
│
├── core/
│   ├── screen_capture.py      ← mss multi-écrans → PIL Image
│   ├── omniparser_bridge.py   ← YOLO + Florence-2 + OCR → List[UIElement]
│   ├── llm_planner.py         ← Ollama → ActionPlan JSON
│   ├── action_executor.py     ← pyautogui (click/type/scroll/drag…)
│   └── agent_loop.py          ← QThread : boucle autonome
│
├── ui/
│   ├── annotated_view.py      ← Screenshot interactif (hover=coords, clic=inspect)
│   ├── element_list.py        ← Tableau des éléments détectés
│   ├── chat_panel.py          ← Chat + log des actions
│   └── main_window.py         ← Fenêtre principale PyQt6
│
└── util/                      ← Tes fichiers OmniParser existants
    ├── utils.py
    ├── omniparser.py
    ├── box_annotator.py
    └── __init__.py
```


## ✨ Fonctionnalités phares

- **Vision ultra-précise** : OmniParser V2 + Florence-2 captions + EasyOCR
- **Classification intelligente** des éléments UI (règles + VQA Florence-2 + OVD pour localiser précisément les champs de saisie)
- **Batch planning** : le LLM planifie plusieurs actions en une seule passe
- **Auto-réflexion** (critique + replan) via un second prompt Ollama
- **Interface PyQt6 riche** : screenshot annoté interactif, liste des éléments, minimap, chat + log
- **Configuration persistante** via UI (config_overrides.json)
- **Multi-écrans** support

## Architecture (haut niveau)

```mermaid
graph TD
    A[Capture Screenshot] --> B[OmniParser Bridge]
    B --> C[UI Classifier]
    C --> D[LLM Planner (Batch)]
    D --> E[Reflection Engine]
    E -->|rejeté| D
    E --> F[Execute Batch]
    F -->|needs_screenshot ou fin batch| A
    F -->|done| END[Terminé]
