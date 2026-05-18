# Cahier des charges fonctionnel — TES (état actuel)

## 1) Objet du document
Ce document décrit le périmètre fonctionnel **actuel** de TES, les besoins couverts, les parcours utilisateurs, les règles de gestion et les critères d’acceptation.

## 2) Contexte et finalité
TES est un outil de **mapping visuel d’interface utilisateur**. Son rôle est de produire un JSON de mapping robuste entre des champs métier (`logical_key`) et des éléments UI détectés à l’écran (bbox relative, point de clic, type d’action).

Le produit cible des environnements logiciels métiers (notamment orthodontie) afin de préparer un mapping exploitable par un agent d’automatisation local.

## 3) Périmètre fonctionnel (in-scope)

### 3.1 Capture et détection UI
- Sélection d’un écran/moniteur.
- Capture de l’écran sélectionné.
- Analyse de la capture via pipeline OmniParser (détection + OCR + description).
- Affichage d’une image annotée avec les éléments détectés.
- Affichage d’une liste filtrable des éléments détectés.

### 3.2 Mapping automatique
- Chargement d’un template JSON contenant des champs métier.
- Tentative d’association automatique `logical_key` ↔ élément détecté (matching sur description/OCR).
- Enrichissement des champs avec :
  - `mapped`
  - `mapping_source`
  - `human_validated`
  - `mapping_confidence`
  - `detected_label`
  - `bbox_relative` (coordonnées normalisées)
  - `click_target`

### 3.3 Édition humaine du mapping
- Ouverture d’un éditeur de mapping en fenêtre non bloquante.
- Visualisation tabulaire des champs.
- Édition des propriétés de champ (`logical_key`, `ui_type`, `path`, `action`).
- Ajout et suppression de champ.
- Toggle de validation humaine.
- Réassignation manuelle d’un champ vers un élément UI sélectionné.
- Historisation des actions d’édition.

### 3.4 Export
- Export du mapping final en JSON.
- Inclusion d’un bloc `mapping_editor` avec résumé et historique.

## 4) Hors périmètre (out-of-scope actuel)
- Exécution d’actions GUI métier en boucle autonome.
- Pilotage LLM multi-étapes de type planification/réflexion.
- Garantie de robustesse cross-version applicative sans recalibrage visuel.

## 5) Utilisateurs cibles
- Opérateur métier / intégrateur fonctionnel.
- Équipe technique configurant un mapping pour un agent local.

## 6) Parcours utilisateur nominal
1. Lancer TES.
2. Charger un template JSON métier.
3. Sélectionner un écran cible.
4. Lancer un parse.
5. Vérifier l’auto-mapping.
6. Ouvrir l’éditeur de mapping.
7. Activer le mode assignation.
8. Sélectionner un champ, puis un élément UI détecté.
9. Assigner et valider.
10. Exporter le JSON final.

## 7) Exigences fonctionnelles détaillées

### EF-01 — Sélection écran
Le système doit lister les écrans disponibles et permettre d’en sélectionner un.

### EF-02 — Parse one-shot
Le système doit exécuter une analyse ponctuelle de la capture et restituer :
- image annotée,
- liste d’éléments structurés.

### EF-03 — Chargement template
Le système doit charger un template JSON valide depuis le disque local.

### EF-04 — Auto-mapping
Le système doit exécuter un mapping automatique des champs du template vers les éléments détectés.

### EF-05 — Éditeur non modal
Le système doit permettre d’interagir avec l’UI principale pendant l’édition du mapping.

### EF-06 — Assignation explicite
Le système doit proposer un mode assignation ON/OFF dans l’éditeur.

### EF-07 — Garde-fous assignation
Le bouton d’assignation manuelle doit être activé uniquement si :
- mode assignation actif,
- un champ est sélectionné,
- un élément UI est sélectionné.

### EF-08 — Aperçu de cible
Le système doit afficher les métadonnées de l’élément actuellement sélectionné (id, subtype, interactif, click target).

### EF-09 — Auto-avancement
Après assignation manuelle, le système doit sélectionner le prochain champ non validé humain.

### EF-10 — Export final
Le système doit exporter le mapping enrichi au format JSON lisible.

## 8) Règles de gestion
- RG-01 : Les coordonnées stockées dans le mapping final sont normalisées (0..1).
- RG-02 : Une assignation manuelle positionne `mapping_source="human"`, `human_validated=true`, `mapping_confidence=1.0`.
- RG-03 : Toute édition significative doit être historisée dans `mapping_editor.history`.
- RG-04 : En absence de correspondance auto, `mapped=false`.

## 9) Exigences non fonctionnelles
- ENF-01 : Interface desktop PyQt6 fluide sur usage standard.
- ENF-02 : Tolérance à des temps de parse variables selon CPU/GPU.
- ENF-03 : Lisibilité des états (prêt, en cours, terminé, erreur) dans l’interface.
- ENF-04 : Compatibilité Python 3.10+.

## 10) Entrées / sorties

### Entrées
- Capture écran active.
- Template JSON métier.
- Interactions utilisateur (sélections, assignations, validations).

### Sorties
- JSON final de mapping.
- Historique d’édition humain intégré.

## 11) Critères d’acceptation (recette)
- CA-01 : L’éditeur ouvert n’empêche pas la sélection d’éléments dans la fenêtre principale.
- CA-02 : Le mode assignation est visible et pilotable (ON/OFF).
- CA-03 : Le bouton “Assigner” est correctement activé/désactivé selon les prérequis.
- CA-04 : L’assignation manuelle met à jour les champs attendus (`bbox_relative`, `click_target`, etc.).
- CA-05 : Après assignation, le champ suivant non validé est sélectionné automatiquement.
- CA-06 : L’export produit un JSON valide contenant le bloc `mapping_editor`.

## 12) Contraintes et dépendances
- Dépendance aux modèles et poids OmniParser.
- Qualité dépendante de la capture (DPI, zoom, contraste, thème).
- Performance liée à la disponibilité GPU.

## 13) Limites connues
- Le matching auto est simple (description/OCR) et peut nécessiter validation manuelle.
- Les variations UI fortes peuvent dégrader la précision du mapping.
- L’outil prépare le mapping mais n’exécute pas les actions métier.
