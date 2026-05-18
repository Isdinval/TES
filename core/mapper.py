"""
core/mapper.py
Helpers pour générer un JSON de mapping UI orienté "agent local".
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


def _to_relative_bbox_xywh(bbox_norm: List[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox_norm
    return {
        "x": round(float(x1), 6),
        "y": round(float(y1), 6),
        "w": round(float(max(0.0, x2 - x1)), 6),
        "h": round(float(max(0.0, y2 - y1)), 6),
    }


def build_mapping(template: Dict[str, Any], elements: List[Any]) -> Dict[str, Any]:
    """
    Complète un template JSON métier avec les bboxes détectées.

    Règle de matching : template.fields[*].logical_key <-> elem.description/ocr_text
    (match exact insensible à la casse, puis fallback inclusion de sous-chaîne).
    """
    result = deepcopy(template)
    fields = result.get("fields", [])

    normalized = []
    for e in elements:
        haystack = f"{getattr(e, 'description', '')} {getattr(e, 'ocr_text', '')}".lower()
        normalized.append((e, haystack))

    for field in fields:
        key = str(field.get("logical_key", "")).strip().lower()
        match = None
        if key:
            for e, hay in normalized:
                if key == hay.strip():
                    match = e
                    break
            if match is None:
                for e, hay in normalized:
                    if key in hay:
                        match = e
                        break

        if match is None:
            field["mapped"] = False
            field["mapping_confidence"] = 0.0
            continue

        field["mapped"] = True
        field["mapping_confidence"] = 0.6
        field["detected_label"] = getattr(match, "label", "")
        field["bbox_relative"] = _to_relative_bbox_xywh(getattr(match, "bbox_norm", [0, 0, 0, 0]))
        field["click_target"] = {
            "x": round(float(match.click_target[0]), 6),
            "y": round(float(match.click_target[1]), 6),
        }

    result["mapper_mode"] = True
    return result
