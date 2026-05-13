"""
core/omniparser_bridge.py
Pont vers OmniParser V2 : YOLO + Florence-2 + EasyOCR.

Retourne une liste de UIElement avec :
  - id, type (text/icon), description, texte OCR
  - bbox normalisée [0,1] et centre normalisé
"""
from __future__ import annotations
import re
import sys
import os
import base64
import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

# ── Patch NumPy 2.0 : np.sctypes supprimé, nécessaire pour imgaug/paddleocr ──
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int':     [np.int8,    np.int16,    np.int32,    np.int64],
        'uint':    [np.uint8,   np.uint16,   np.uint32,   np.uint64],
        'float':   [np.float32, np.float64,  np.longdouble],
        'complex': [np.complex64, np.complex128],
        'others':  [bool, object, bytes, str, np.void],
    }
# ─────────────────────────────────────────────────────────────────────────────

# Ajoute le répertoire racine du projet au PYTHONPATH pour trouver util/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from util.utils import (
        get_som_labeled_img,
        get_caption_model_processor,
        get_yolo_model,
        check_ocr_box,
    )
except ImportError as e:
    raise ImportError(
        f"Impossible d'importer util.utils : {e}\n"
        "Vérifie que le dossier 'util/' est bien à la racine du projet."
    )


# ─── Dataclass UIElement ──────────────────────────────────────────────────────

# Sous-types interactifs reconnus
INTERACTIVE_SUBTYPES = {
    "text_input", "button", "checkbox", "radio",
    "dropdown", "link", "icon_button", "toggle", "slider",
}

@dataclass
class UIElement:
    id:           int
    elem_type:    str           # 'text' | 'icon'
    description:  str           # caption Florence-2 ou texte OCR
    ocr_text:     str           # texte OCR brut
    bbox_norm:    List[float]   # [x1, y1, x2, y2] normalisé [0,1]
    center_norm:  Tuple[float, float]

    # ── Champs de classification (remplis par UIClassifier) ──
    subtype:            str   = "unknown"  # text_input | button | checkbox | ...
    is_interactive:     bool  = False
    input_center_norm:  Optional[Tuple[float, float]] = None  # zone cliquable précise

    @property
    def click_target(self) -> Tuple[float, float]:
        """Coordonnée normalisée à utiliser pour cliquer sur cet élément."""
        return self.input_center_norm or self.center_norm

    @property
    def label(self) -> str:
        txt = self.description or self.ocr_text
        return txt[:60] + "…" if len(txt) > 60 else txt

    def to_llm_dict(self) -> dict:
        return {
            "id":           self.id,
            "subtype":      self.subtype,
            "interactive":  self.is_interactive,
            "description":  self.description,
            "ocr_text":     self.ocr_text,
            "click_target": [round(self.click_target[0], 3),
                             round(self.click_target[1], 3)],
        }


# ─── Bridge principal ────────────────────────────────────────────────────────

class OmniParserBridge:
    """
    Wrapper autour des fonctions OmniParser V2.
    Charge les modèles une seule fois, puis expose parse().
    """

    def __init__(self, config: dict, device: Optional[str] = None):
        import torch
        from core.ui_classifier import UIClassifier
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[OmniParser] Chargement des modèles sur {self.device}…")
        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(
            model_name=config['caption_model_name'],
            model_name_or_path=config['caption_model_path'],
            device=self.device,
        )
        # Classifier UI : réutilise Florence-2 déjà chargé, pas de surcoût mémoire
        self.classifier = UIClassifier(
            caption_model_processor=self.caption_model_processor,
            use_florence_vqa=True,
        )
        print("[OmniParser] Modèles chargés ✓")

    # ── API publique ──────────────────────────────────────────────────────────

    def parse(self, pil_image: Image.Image) -> Tuple[Image.Image, List[UIElement]]:
        """
        Analyse une image PIL.
        Retourne (image_annotée PIL, liste de UIElement).
        """
        img_w, img_h = pil_image.size
        box_overlay_ratio = max(img_w, img_h) / 3200

        draw_cfg = {
            'text_scale':     0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding':   max(int(3 * box_overlay_ratio), 1),
            'thickness':      max(int(3 * box_overlay_ratio), 1),
        }

        # OCR
        (ocr_text, ocr_bbox), _ = check_ocr_box(
            pil_image,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'text_threshold': 0.8},
            use_paddleocr=False,
        )

        # YOLO + Florence-2
        labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            pil_image,
            self.som_model,
            BOX_TRESHOLD=self.config['BOX_TRESHOLD'],
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_cfg,
            caption_model_processor=self.caption_model_processor,
            ocr_text=ocr_text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128,
        )

        # Convertit labeled_img en PIL si besoin
        if isinstance(labeled_img, str):             # base64
            labeled_img = Image.open(io.BytesIO(base64.b64decode(labeled_img)))
        elif isinstance(labeled_img, np.ndarray):
            labeled_img = Image.fromarray(labeled_img)

        elements = self._build_elements(parsed_content_list, label_coordinates)

        # ── Classification interactive / non-interactive ──────────────────────
        if elements:
            self.classifier.classify(elements, pil_image)
            n_interactive = sum(1 for e in elements if e.is_interactive)
            print(f"[UIClassifier] {n_interactive}/{len(elements)} éléments interactifs")

        # ── Debug : affiche toujours les 3 premiers raw coords ───────────────
        print(f"[OmniParser] {len(elements)} éléments construits | "
              f"{len(label_coordinates)} coords brutes | "
              f"{len(parsed_content_list)} items parsed_content")
        print("  [DEBUG] 3 premiers label_coordinates bruts :")
        for i, (k, v) in enumerate(list(label_coordinates.items())[:3]):
            print(f"    id={k}  bbox_raw={v}")
        if elements:
            e0 = elements[0]
            print(f"  [DEBUG] Element[0] → bbox_norm={e0.bbox_norm}  center_norm={e0.center_norm}")

        return labeled_img, elements

    # ── Parsing interne ───────────────────────────────────────────────────────

    def _build_elements(
        self,
        parsed_content_list: List[Any],
        label_coordinates: Dict[Any, List[float]],
    ) -> List[UIElement]:
        """
        Construit la liste de UIElement depuis les sorties brutes d'OmniParser V2.

        OmniParser V2 retourne parsed_content_list sous plusieurs formats selon
        la version / config :

        Format A — liste de strings (le plus courant) :
            "Text Box ID: 0; content: Submit"
            "Icon Box ID: 1; content: search"
            "0: Submit"  (fallback sans préfixe)

        Format B — liste de dicts :
            {"id": 0, "type": "text", "content": "Submit", "bbox": [...]}

        Format C — liste indexée par position (index = id implicite) :
            ["Submit", "Cancel", ...]   → id = index dans la liste

        Fallback — si aucun item ne parse, construit depuis label_coordinates
        directement (garantit toujours des éléments si YOLO a détecté qqch).
        """
        # ── Normalise label_coordinates en {int → [x1,y1,x2,y2]} ─────────────
        coords: Dict[int, List[float]] = {}
        for k, v in label_coordinates.items():
            try:
                coords[int(k)] = v
            except (TypeError, ValueError):
                pass

        if not coords:
            return []

        # ── Détection du format bbox (une seule fois) ─────────────────────────
        if OmniParserBridge._bbox_format is None:
            OmniParserBridge._bbox_format = self._detect_format_from_coords(coords)
            print(f"  [BBox fmt] Format retenu : '{OmniParserBridge._bbox_format}'")

        elements: List[UIElement] = []

        # ── Tente de parser parsed_content_list ───────────────────────────────
        for idx, item in enumerate(parsed_content_list):
            elem = None
            if isinstance(item, str):
                elem = self._parse_string_item(item, idx, coords)
            elif isinstance(item, dict):
                elem = self._parse_dict_item(item, idx, coords)
            if elem is not None:
                elements.append(elem)

        # ── Fallback : construire depuis label_coordinates si tout a raté ──────
        # (se produit quand parsed_content_list est vide ou dans un format inconnu)
        if not elements and coords:
            print("[OmniParser] Fallback → construction depuis label_coordinates")
            for eid, bbox in sorted(coords.items()):
                elements.append(self._make_element_fallback(eid, bbox))

        return elements

    # -- Format string ---------------------------------------------------------

    # Patterns couvrant les variantes connues d'OmniParser V2
    _PATTERNS = [
        # "Text Box ID: 3; content: Submit"
        (re.compile(r'[Tt]ext\s+[Bb]ox\s+ID\s*[:\s]+(\d+)\s*[;,]\s*content\s*[:\s]+(.*)', re.IGNORECASE), 'text'),
        # "Icon Box ID: 5; content: search icon"
        (re.compile(r'[Ii]con\s+[Bb]ox\s+ID\s*[:\s]+(\d+)\s*[;,]\s*content\s*[:\s]+(.*)', re.IGNORECASE), 'icon'),
        # "type: text, id: 3, content: Submit"  (ordre variable)
        (re.compile(r'.*?id\s*[:\s]+(\d+).*?content\s*[:\s]+(.*)', re.IGNORECASE), None),
        # "3: Submit"  ou  "3 - Submit"
        (re.compile(r'^(\d+)\s*[:\-]\s*(.*)', re.IGNORECASE), 'icon'),
    ]

    def _parse_string_item(
        self, text: str, idx: int, coords: Dict[int, List[float]]
    ) -> Optional[UIElement]:
        t = text.strip()
        for pattern, forced_type in self._PATTERNS:
            m = pattern.match(t)
            if m:
                try:
                    eid = int(m.group(1))
                except (ValueError, IndexError):
                    continue
                content = m.group(2).strip() if len(m.groups()) >= 2 else ''
                # Détermine le type si non forcé
                etype = forced_type
                if etype is None:
                    etype = 'text' if 'text' in t.lower() else 'icon'
                return self._make_element(eid, etype, content, coords)

        # Dernier recours : l'index dans la liste correspond à l'id
        if idx in coords:
            return self._make_element(idx, 'icon', t, coords)
        return None

    # -- Format dict -----------------------------------------------------------

    def _parse_dict_item(
        self, d: dict, idx: int, coords: Dict[int, List[float]]
    ) -> Optional[UIElement]:
        # Cherche l'id sous toutes ses formes possibles
        eid = None
        for key in ('id', 'box_id', 'element_id', 'index'):
            if key in d:
                try:
                    eid = int(d[key])
                    break
                except (TypeError, ValueError):
                    pass
        if eid is None:
            eid = idx  # fallback sur la position

        etype = str(d.get('type', d.get('element_type', 'icon'))).lower()
        if etype not in ('text', 'icon'):
            etype = 'text' if 'text' in etype else 'icon'

        content = str(d.get('content',
                    d.get('description',
                    d.get('label',
                    d.get('caption', '')))))

        # Certaines versions embarquent la bbox dans le dict lui-même
        if eid not in coords and 'bbox' in d:
            try:
                coords[eid] = [float(x) for x in d['bbox']]
            except (TypeError, ValueError):
                pass

        return self._make_element(eid, etype, content, coords)

    # -- Constructeurs --------------------------------------------------------

    # Format détecté une fois pour toutes lors du premier appel
    _bbox_format: Optional[str] = None   # 'xyxy' | 'xywh'

    @classmethod
    def _detect_format_from_coords(cls, coords: Dict[int, List[float]]) -> str:
        """
        Détecte automatiquement le format des bbox depuis label_coordinates.

        OmniParser V2 peut retourner deux formats selon la version :
          • 'xyxy' : [x1, y1, x2, y2]  — x2 > x1, y2 > y1
          • 'xywh' : [x1, y1, w, h]    — w et h sont des dimensions (pas des coords absolues)

        Heuristique :
          Dans le format xywh, bbox[0] + bbox[2] donne le bord droit.
          Dans le format xyxy, bbox[2] est directement le bord droit.
          Si bbox[2] < bbox[0], c'est impossible en xyxy → c'est xywh.
          Sinon, compare si bbox[2]-bbox[0] ressemble à une largeur (<0.5)
          ou à une coordonnée absolue (> bbox[0]+0.05).
        """
        samples = list(coords.values())[:20]
        xywh_votes = 0
        xyxy_votes = 0

        for b in samples:
            if len(b) < 4:
                continue
            x1, y1, v2, v3 = b[:4]

            # Cas certain xywh : la "x2" serait inférieure à x1 (impossible en xyxy)
            if v2 < x1 or v3 < y1:
                xywh_votes += 2
                continue

            # Si x1+v2 et y1+v3 restent bien dans [0,1] ET v2 < x1+0.6
            # c'est probablement xywh (width/height petits)
            if (0 < x1 + v2 <= 1.01) and (0 < y1 + v3 <= 1.01):
                if v2 < 0.6 and v3 < 0.6:
                    # Centre xywh = (x1+v2/2, y1+v3/2) — doit être > x1 et < 1
                    cx_xywh = x1 + v2 / 2
                    cy_xywh = y1 + v3 / 2
                    if 0 < cx_xywh < 1 and 0 < cy_xywh < 1:
                        xywh_votes += 1

            # Centre xyxy = ((x1+v2)/2, (y1+v3)/2) — doit être entre x1 et v2
            cx_xyxy = (x1 + v2) / 2
            cy_xyxy = (y1 + v3) / 2
            if x1 < cx_xyxy < v2 and y1 < cy_xyxy < v3:
                xyxy_votes += 1

        fmt = 'xywh' if xywh_votes > xyxy_votes else 'xyxy'
        print(f"  [BBox fmt] votes xyxy={xyxy_votes} xywh={xywh_votes} → format='{fmt}'")
        return fmt

    @classmethod
    def _bbox_to_xyxy(
        cls, raw: List[float], fmt: str
    ) -> Optional[List[float]]:
        """
        Convertit une bbox brute en [x1, y1, x2, y2] normalisé.
        Retourne None si les valeurs sont incohérentes.
        """
        if len(raw) < 4:
            return None
        x1, y1, v2, v3 = raw[:4]

        if fmt == 'xywh':
            # [x1, y1, width, height] → [x1, y1, x2, y2]
            x2, y2 = x1 + v2, y1 + v3
        else:
            # [x1, y1, x2, y2] — déjà dans le bon format
            x2, y2 = v2, v3

        # Sanity check
        if x2 <= x1 or y2 <= y1:
            # Essai de l'autre format en fallback
            if fmt == 'xyxy':
                x2, y2 = x1 + v2, y1 + v3
            else:
                x2, y2 = v2, v3
            if x2 <= x1 or y2 <= y1:
                return None

        # Clamp [0, 1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        return [x1, y1, x2, y2]

    def _make_element(
        self, eid: int, etype: str, content: str,
        coords: Dict[int, List[float]],
    ) -> Optional[UIElement]:
        if eid not in coords:
            return None
        raw  = coords[eid]
        bbox = self._bbox_to_xyxy(raw, self._bbox_format or 'xyxy')
        if bbox is None:
            return None
        cx  = (bbox[0] + bbox[2]) / 2
        cy  = (bbox[1] + bbox[3]) / 2
        ocr = content if etype == 'text' else ''
        return UIElement(
            id          = eid,
            elem_type   = etype,
            description = content,
            ocr_text    = ocr,
            bbox_norm   = bbox,
            center_norm = (cx, cy),
        )

    def _make_element_fallback(self, eid: int, raw_bbox: List[float]) -> UIElement:
        """Crée un UIElement minimal quand parsed_content_list ne parse pas."""
        bbox = self._bbox_to_xyxy(raw_bbox, self._bbox_format or 'xyxy') or raw_bbox[:4]
        cx   = (bbox[0] + bbox[2]) / 2
        cy   = (bbox[1] + bbox[3]) / 2
        return UIElement(
            id          = eid,
            elem_type   = 'icon',
            description = f'élément {eid}',
            ocr_text    = '',
            bbox_norm   = list(bbox[:4]),
            center_norm = (cx, cy),
        )