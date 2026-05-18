# TES — Agent Mapper UI (OmniParser V2)

## Vue d'ensemble
TES est maintenant un **agent mapper visuel** orienté détection et cartographie d'interfaces, sans boucle d'action/rétroaction.  
Le projet sert à construire un **JSON de mapping UI robuste** (positions relatives, type de champ, chemin fonctionnel, action attendue) à partir d'une capture d'écran analysée par OmniParser V2.

Le but opérationnel est d'alimenter un agent local côté utilisateur (ex. automation desktop) avec un mapping stable et portable entre résolutions d'écran.

---

## Objectif produit
Dans le contexte SAYDO → logiciels métiers d'orthodontie (Orthalis, Arakis/Orthokis, OrthoADVANCE…), l'objectif est de :

- réduire la double saisie,
- limiter les erreurs administratives,
- accélérer les flux post-consultation,
- fiabiliser le transfert des informations vers les logiciels métiers sans dépendre d'API natives.

TES ne pilote pas directement les actions métier : il prépare le terrain avec un mapping précis et exploitable.

---

## Ce que fait TES aujourd'hui

### 1) Détection UI
- Capture écran multi-moniteurs.
- Analyse visuelle via OmniParser V2 (YOLO + Florence-2 + OCR).
- Production d'une liste d'éléments UI détectés (bbox, texte OCR, description, type probable).

### 2) Mapping JSON
À partir d'un template JSON métier (défini manuellement), TES enrichit chaque champ avec :
- `bbox_relative` (coordonnées relatives 0..1),
- `click_target` (point de clic utile),
- `mapped` (booléen),
- `mapping_confidence` (score de confiance).

### 3) Export
- Export d'un JSON final prêt à être consommé par un agent local d'automatisation.

---

## Ce que TES ne fait plus
- Plus de planification LLM multi-étapes.
- Plus d'exécution d'actions GUI.
- Plus de boucle autonome action/réflexion/replan.

Le projet est volontairement focalisé sur le rôle de **mapper visuel**.

---

## Architecture actuelle

```text
TES/
├── main.py
├── config.py
├── requirements.txt
│
├── core/
│   ├── screen_capture.py      # capture écran + infos moniteurs
│   ├── omniparser_bridge.py   # pipeline YOLO / Florence-2 / OCR
│   ├── ui_classifier.py       # enrichissement typologique des éléments
│   ├── parse_task.py          # tâche parse one-shot (QThread)
│   └── mapper.py              # construction du mapping JSON final
│
├── ui/
│   ├── main_window.py         # UI principale (parse + export mapping)
│   ├── annotated_view.py      # vue screenshot annotée
│   ├── element_list.py        # liste des éléments détectés
│   ├── chat_panel.py          # journal + interactions UI
│   └── config_panel.py        # configuration runtime
│
└── util/
    ├── omniparser.py
    ├── utils.py
    ├── box_annotator.py
    └── __init__.py
```

---

## Pipeline de traitement

1. Sélection de l'écran cible.
2. Capture image.
3. Détection + OCR + description des éléments.
4. Visualisation annotée dans l'UI.
5. Chargement d'un template JSON métier.
6. Matching des `logical_key` avec les éléments détectés.
7. Conversion en coordonnées relatives.
8. Export du JSON enrichi.

---

## Modèle de données de mapping (résumé)

Exemple de structure attendue par champ :

```json
{
  "logical_key": "diagnostic_squelettique",
  "ui_type": "text_input",
  "path": "Fiche Patient > Diagnostic",
  "action": "click_then_type",
  "bbox_relative": {"x": 0.015, "y": 0.243, "w": 0.185, "h": 0.032},
  "click_target": {"x": 0.107, "y": 0.259},
  "mapped": true,
  "mapping_confidence": 0.88
}
```

Principe clé : les coordonnées relatives rendent le mapping réutilisable sur des résolutions différentes.

---

## Interface utilisateur

La fenêtre principale est centrée sur deux actions :
- **📸 Parse** : analyser l'écran courant.
- **🗺️ Export Mapping** : charger un template JSON puis exporter le mapping enrichi.

Le panneau de droite conserve l'historique/log pour suivre les opérations de parsing et d'export.

---

## Pré-requis techniques

- Python 3.10+
- Poids OmniParser présents dans `weights/`
- Environnement GPU recommandé (CPU possible mais plus lent)

---

## Lancement

```bash
python main.py
```

---

## Notes de qualité

- Le mapping dépend de la qualité visuelle de la capture (zoom OS, thème, DPI, contraste).
- Les labels métier (`logical_key`) doivent être explicites pour améliorer le matching.
- Un contrôle manuel du JSON exporté reste recommandé pour les champs critiques.
