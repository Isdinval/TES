"""
core/ui_classifier.py
Classification des éléments UI détectés par OmniParser.

Stratégie en deux passes :
  1. Règles sémantiques rapides (regex sur description + ocr_text + bbox)
     → Couvre ~85% des cas sans appel modèle
  2. Florence-2 VQA sur les éléments ambigus restants
     → "What type of UI element is shown? Choose: text_input / button /
        checkbox / radio / dropdown / label / icon_button / other"

Pour chaque élément classifié "text_input" et dont la bbox est haute
(label + champ regroupés), Florence-2 localise le sous-rectangle du champ
via Open Vocabulary Detection → donne input_center_norm précis.
"""
from __future__ import annotations
import re
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

from core.omniparser_bridge import UIElement, INTERACTIVE_SUBTYPES


# ─── Tables de correspondance sémantique ────────────────────────────────────

# Mots-clés dans la description Florence-2 ou l'OCR → subtype
_KEYWORD_MAP: List[Tuple[str, str]] = [
    # text_input
    (r'(votre\s+r[ée]ponse|your\s+answer|réponse\s+ici|type\s+here|'
     r'placeholder|text\s*(field|box|input|area)|champ\s*(texte|de\s*saisie)|'
     r'saisir|saisissez|enter\s+(your|text)|tapez)', 'text_input'),
    # button
    (r'(^(envoyer|soumettre|valider|confirmer|annuler|fermer|suivant|précédent|'
     r'submit|send|cancel|ok\b|next|back|sign\s*in|log\s*in|register|'
     r'créer|sauvegarder|save|delete|supprimer|continuer|continue|'
     r'accepter|refuser|télécharger|upload|download)\b)', 'button'),
    # checkbox
    (r'(checkbox|case\s*à\s*cocher|☐|☑|✓|check\s*box)', 'checkbox'),
    # radio
    (r'(radio\s*(button)?|option\s*button|◉|○\s)', 'radio'),
    # dropdown
    (r'(dropdown|drop.?down|select|▼|liste\s+déroulante|combobox)', 'dropdown'),
    # link
    (r'(https?://|www\.|\.com|\.fr|lien\s+hypertexte|click\s+here|'
     r'cliquez\s+ici)', 'link'),
    # toggle
    (r'(toggle|switch|activer|désactiver)', 'toggle'),
    # slider
    (r'(slider|curseur|barre\s+de\s+défilement)', 'slider'),
    # icon_button  (petites icônes cliquables)
    (r'(icon|icône|search\s+icon|close\s+icon|menu\s+icon|settings\s+icon|'
     r'edit\s+icon|delete\s+icon|add\s+icon|plus\s+icon)', 'icon_button'),
    # label (non-interactif)
    (r'(^(titre|title|heading|label|description|question|section|'
     r'instructions?|note[sz]?)\b)', 'label'),
]

# OCR patterns pour text_input (placeholder text courants)
_OCR_INPUT_PATTERNS = re.compile(
    r'(votre\s+r[ée]ponse|your\s+answer|répondez\s+ici|type\s+here|'
    r'enter\s+text|saisir\.+|\.{3,})',
    re.IGNORECASE
)

# Sous-types qui sont interactifs
_INTERACTIVE = INTERACTIVE_SUBTYPES


# ─── UIClassifier ────────────────────────────────────────────────────────────

class UIClassifier:
    """
    Enrichit une liste de UIElement avec subtype, is_interactive,
    et input_center_norm pour les champs de saisie.
    """

    def __init__(
        self,
        caption_model_processor: Optional[Any] = None,
        use_florence_vqa: bool = True,
    ):
        """
        caption_model_processor : tuple (model, processor) Florence-2
            déjà chargé dans OmniParserBridge — on le réutilise.
        use_florence_vqa : si True, lance VQA sur les éléments ambigus.
        """
        self.caption_model_processor = caption_model_processor
        self.use_florence_vqa = use_florence_vqa and caption_model_processor is not None

    # ── API publique ──────────────────────────────────────────────────────────

    def classify(
        self,
        elements:   List[UIElement],
        screenshot: Image.Image,
    ) -> List[UIElement]:
        """
        Classifie tous les éléments en place (modifie les objets).
        Retourne la même liste (pour chaîner).
        """
        ambiguous: List[UIElement] = []

        for elem in elements:
            subtype = self._classify_by_keywords(elem)
            if subtype:
                elem.subtype       = subtype
                elem.is_interactive = subtype in _INTERACTIVE
            else:
                # Règles géométriques
                subtype = self._classify_by_geometry(elem)
                if subtype:
                    elem.subtype       = subtype
                    elem.is_interactive = subtype in _INTERACTIVE
                else:
                    ambiguous.append(elem)

        # Passe VQA Florence-2 sur les ambigus
        if self.use_florence_vqa and ambiguous:
            self._classify_by_vqa(ambiguous, screenshot)
        else:
            # Fallback : marque "label" les non résolus (non-interactif par défaut)
            for elem in ambiguous:
                if elem.subtype == "unknown":
                    elem.subtype       = "label"
                    elem.is_interactive = False

        # Localisation précise du champ de saisie dans les text_input ambigus
        for elem in elements:
            if elem.subtype == "text_input":
                elem.input_center_norm = self._locate_input_area(elem, screenshot)

        return elements

    # ── Passe 1 : règles sémantiques ─────────────────────────────────────────

    def _classify_by_keywords(self, elem: UIElement) -> Optional[str]:
        text = (elem.description + " " + elem.ocr_text).lower().strip()

        # OCR placeholder → text_input direct
        if _OCR_INPUT_PATTERNS.search(text):
            return "text_input"

        for pattern, subtype in _KEYWORD_MAP:
            if re.search(pattern, text, re.IGNORECASE):
                return subtype

        return None

    # ── Passe 2 : règles géométriques ─────────────────────────────────────────

    def _classify_by_geometry(self, elem: UIElement) -> Optional[str]:
        """
        Heuristiques sur la forme de la bbox.
        - Carré petit (< 4% × 4%) → icon_button
        - Large et plat (ratio w/h > 4) → button ou text_input selon le type
        - Très haut (ratio h/w > 2) → text_input (zone de texte multi-ligne)
        """
        bbox = elem.bbox_norm
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return None

        # Petite icône carrée
        if w < 0.05 and h < 0.05 and abs(w - h) < 0.02:
            return "icon_button"

        # Zone de texte multi-ligne (haute)
        if h > 0.08 and h / w > 1.2:
            return "text_input"

        # Bouton large et peu haut (type Google Forms)
        if w > 0.15 and h < 0.06 and elem.elem_type == "text":
            return "button"

        return None

    # ── Passe 3 : Florence-2 VQA ─────────────────────────────────────────────

    _VQA_PROMPT = (
        "What type of UI element is this? "
        "Answer with exactly one word from: "
        "text_input, button, checkbox, radio, dropdown, label, icon_button, other"
    )

    _VQA_LABEL_MAP = {
        "text_input":   "text_input",
        "button":       "button",
        "checkbox":     "checkbox",
        "radio":        "radio",
        "dropdown":     "dropdown",
        "label":        "label",
        "icon_button":  "icon_button",
        "input":        "text_input",
        "text":         "text_input",
        "field":        "text_input",
        "link":         "link",
        "other":        "label",
    }

    def _classify_by_vqa(
        self,
        elements:   List[UIElement],
        screenshot: Image.Image,
    ) -> None:
        """
        Pour chaque élément ambigu, crop la bbox et demande à Florence-2
        de quel type d'élément UI il s'agit.
        """
        try:
            import torch
            model, processor = self.caption_model_processor
            device = next(model.parameters()).device

            for elem in elements:
                crop = self._crop_element(elem, screenshot)
                if crop is None:
                    continue

                inputs = processor(
                    text    = self._VQA_PROMPT,
                    images  = crop,
                    return_tensors = "pt",
                ).to(device)

                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens = 10,
                        do_sample      = False,
                    )

                answer = processor.batch_decode(
                    generated, skip_special_tokens=True
                )[0].strip().lower()

                # Cherche un mot-clé connu dans la réponse
                subtype = None
                for keyword, stype in self._VQA_LABEL_MAP.items():
                    if keyword in answer:
                        subtype = stype
                        break
                if subtype is None:
                    subtype = "label"

                elem.subtype       = subtype
                elem.is_interactive = subtype in _INTERACTIVE

        except Exception as e:
            print(f"[UIClassifier] VQA Florence-2 erreur : {e}")
            # Fallback silencieux
            for elem in elements:
                if elem.subtype == "unknown":
                    elem.subtype       = "label"
                    elem.is_interactive = False

    # ── Localisation précise zone input ──────────────────────────────────────

    _OVD_QUERIES = [
        "text input field",
        "input box",
        "text area",
        "answer field",
    ]

    def _locate_input_area(
        self, elem: UIElement, screenshot: Image.Image
    ) -> Optional[Tuple[float, float]]:
        """
        Pour un text_input dont la bbox est grande (label + champ groupés),
        utilise Florence-2 Open Vocabulary Detection pour localiser précisément
        la zone de saisie à l'intérieur de la bbox.

        Retourne les coordonnées normalisées du centre de la zone input
        (dans le repère de l'écran entier), ou None si la bbox est déjà petite.
        """
        bbox = elem.bbox_norm
        h    = bbox[3] - bbox[1]

        # Si la bbox est petite → le centre est correct, pas besoin de localiser
        if h < 0.06:
            return None

        if not self.use_florence_vqa:
            # Fallback géométrique : zone basse de la bbox (last 40%)
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[1] + h * 0.72
            return (cx, cy)

        try:
            import torch
            model, processor = self.caption_model_processor
            device = next(model.parameters()).device

            crop = self._crop_element(elem, screenshot)
            if crop is None:
                return None

            # Open Vocabulary Detection dans le crop
            best_box   = None
            best_score = 0.0

            for query in self._OVD_QUERIES:
                prompt = f"<OPEN_VOCABULARY_DETECTION>{query}"
                inputs = processor(
                    text=prompt, images=crop, return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                    )
                result = processor.post_process_generation(
                    processor.batch_decode(out, skip_special_tokens=False)[0],
                    task="<OPEN_VOCABULARY_DETECTION>",
                    image_size=(crop.width, crop.height),
                )

                bboxes = result.get("<OPEN_VOCABULARY_DETECTION>", {}).get("bboxes", [])
                scores = result.get("<OPEN_VOCABULARY_DETECTION>", {}).get("bboxes_scores", [1.0] * len(bboxes))

                for box, score in zip(bboxes, scores):
                    if score > best_score:
                        best_score = float(score)
                        best_box   = box

            if best_box and best_score > 0.2:
                # best_box en pixels du crop → normalise dans le repère écran
                cw, ch = crop.width, crop.height
                # centre dans le crop (normalisé [0,1])
                cx_crop = (best_box[0] + best_box[2]) / 2 / cw
                cy_crop = (best_box[1] + best_box[3]) / 2 / ch

                # Remappe dans le repère écran complet
                bx1, by1 = bbox[0], bbox[1]
                bw  = bbox[2] - bbox[0]
                bh  = bbox[3] - bbox[1]
                cx_screen = bx1 + cx_crop * bw
                cy_screen = by1 + cy_crop * bh

                # Sanity : doit rester dans la bbox
                if bbox[0] <= cx_screen <= bbox[2] and bbox[1] <= cy_screen <= bbox[3]:
                    return (cx_screen, cy_screen)

        except Exception as e:
            print(f"[UIClassifier] OVD localisation erreur : {e}")

        # Fallback : zone basse de la bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[1] + h * 0.72
        return (cx, cy)

    # ── Utilitaires ──────────────────────────────────────────────────────────

    @staticmethod
    def _crop_element(
        elem: UIElement, screenshot: Image.Image
    ) -> Optional[Image.Image]:
        """Extrait le crop PIL de la bbox de l'élément."""
        try:
            iw, ih = screenshot.size
            x1 = int(elem.bbox_norm[0] * iw)
            y1 = int(elem.bbox_norm[1] * ih)
            x2 = int(elem.bbox_norm[2] * iw)
            y2 = int(elem.bbox_norm[3] * ih)
            # Marge minimale de 1px
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            return screenshot.crop((x1, y1, x2, y2))
        except Exception:
            return None