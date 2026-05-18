# Contexte projet — TES

## 1) Résumé exécutif
TES est une application desktop Python (PyQt6) dédiée à la création d’un mapping JSON entre un référentiel de champs métier et des éléments d’interface détectés visuellement.

Le projet est orienté “**mapper visuel**” : il capte l’écran, détecte les composants UI (YOLO + Florence + OCR), puis assiste un humain pour valider/corriger le mapping avant export.

## 2) Problème adressé
Dans des contextes métiers sans API natives robustes, les équipes doivent souvent ressaisir des informations dans plusieurs outils. TES vise à réduire cette friction en préparant un mapping UI fiable, réutilisable et traçable.

## 3) Positionnement produit
- Ce que TES fait : détection UI, pré-mapping, édition humaine, export JSON.
- Ce que TES ne fait pas : orchestration autonome d’actions GUI de bout en bout.

## 4) Contexte métier
Le projet a été conçu pour des logiciels métiers (ex. orthodontie), où les écrans sont parfois hétérogènes et peu interopérables. L’approche visuelle permet d’éviter une dépendance stricte aux API applicatives.

## 5) Architecture applicative (vue d’ensemble)
- `main.py` : point d’entrée GUI.
- `ui/main_window.py` : shell principal (parse, édition, export).
- `ui/annotated_view.py` : visualisation annotée.
- `ui/element_list.py` : liste des éléments détectés + filtrage.
- `ui/mapping_editor.py` : validation/édition humaine du mapping.
- `core/parse_task.py` : tâche asynchrone de parse.
- `core/omniparser_bridge.py` : intégration OmniParser + structuration `UIElement`.
- `core/mapper.py` : génération/enrichissement du mapping JSON.
- `core/screen_capture.py` : gestion des écrans et capture.

## 6) Flux principal
1. L’utilisateur choisit l’écran.
2. TES capture puis parse la vue.
3. Les éléments détectés sont affichés en vue annotée + tableau.
4. Un template JSON est chargé.
5. TES auto-mappe les champs.
6. L’utilisateur affine dans l’éditeur (mode assignation, validation).
7. TES exporte le mapping final.

## 7) Donnée centrale
Le contrat de sortie est un JSON de mapping contenant :
- clés métier (`logical_key`, `path`, `action`, `ui_type`),
- coordonnées relatives (`bbox_relative`),
- point de clic (`click_target`),
- métadonnées de qualité (`mapped`, `mapping_confidence`, `mapping_source`, `human_validated`),
- journal d’édition (`mapping_editor`).

## 8) Principes de conception observés
- Human-in-the-loop : l’humain reste arbitre final.
- Coordonnées normalisées : meilleure portabilité inter-résolutions.
- Non-modalité de l’éditeur : interaction simultanée avec la vue principale.
- Traçabilité des modifications par historique.

## 9) Dépendances et environnement
- Python 3.10+
- PyQt6
- Stack OmniParser (modèles/poids locaux)
- OCR et composants ML associés
- GPU recommandé (CPU possible)

## 10) Risques et points d’attention
- Stabilité des détections selon qualité visuelle.
- Variabilité UI (thèmes, zoom OS, densité pixel).
- Risque de faux positifs sur auto-mapping nécessitant validation humaine.

## 11) État actuel et trajectoire
État actuel : socle fonctionnel de mapping visuel opérationnel avec édition humaine assistée.

Trajectoire naturelle :
- contrôles qualité pré-export,
- détection d’incohérences,
- amélioration UX de production de mapping en volume,
- outillage de maintenance des templates métier.
