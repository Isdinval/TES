"""
config.py — Configuration centrale de l'agent GUI (version batch)
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── OmniParser V2 ─────────────────────────────────────────────────────────────
OMNIPARSER_CONFIG = {
    'som_model_path':      os.path.join(BASE_DIR, 'weights/icon_detect/model.pt'),
    'caption_model_name':  'florence2',
    'caption_model_path':  os.path.join(BASE_DIR, 'weights/icon_caption_florence'),
    'BOX_TRESHOLD':        0.005,
}

# ── Ollama LLM ────────────────────────────────────────────────────────────────
OLLAMA_CONFIG = {
    'model':       'qwen3.5:397B-cloud',
    'base_url':    'http://localhost:11434',
    'temperature': 0.1,
    'max_tokens':  4096,    # ↑ augmenté pour les réponses batch (listes d'actions)
    'timeout':     180,
}

# ── Boucle agent (batch multi-actions) ────────────────────────────────────────
AGENT_CONFIG = {
    'max_cycles':        20,      # ↓ réduit : chaque cycle = plusieurs actions
    'step_delay':        1.4,   # ← NOUVEAU : pause ENTRE les actions d'un batch (s)
    'post_action_delay': 1.2,    # pause après le batch complet (avant recapture)
    'type_interval':     0.04,   # délai entre frappes clavier
    'scroll_amount':     5,
    'drag_duration':     0.5,
}

# ── Interface ──────────────────────────────────────────────────────────────────
UI_CONFIG = {
    'window_title':     'GUI Agent — OmniParser + Ollama (batch)',
    'screenshot_min_w':  1920,
    'elements_min_w':    1080,
    'chat_min_w':        420,
    'theme': {
        'bg':           '#1e1e2e',
        'panel':        '#2a2a3e',
        'accent':       '#7c3aed',
        'accent_light': '#a855f7',
        'success':      '#22c55e',
        'warning':      '#f59e0b',
        'error':        '#ef4444',
        'text':         '#e2e8f0',
        'text_muted':   '#94a3b8',
    }
}