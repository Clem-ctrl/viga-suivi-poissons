# Étude de cas VIGA : suivi individuel de poissons par IA

Étude de cas préparée pour la candidature à la thèse VIGA (LabISEN – ANSES). Sur un jeu de données public, elle
mesure le comportement de chaque poisson par détection de pose et suivi multi-objets. Elle répond à deux questions
de la thèse :

- **Q1.** Quels indicateurs comportementaux distinguent les poissons atteints des autres, et à partir de quand ?
- **Q2.** Que gagne-t-on à suivre chaque poisson par IA, par rapport à une mesure globale du bassin ou à une mesure
  simple par poisson de type ZebraLab ?

Le rapport complet (méthode, résultats, limites, apports pour VIGA) est joint à la candidature.

## Résultats en bref

- Les poissons injectés (*Aeromonas hydrophila*) se distinguent dès les premières minutes : immobiles 87 % du temps
  contre 48 %. Sans témoin injecté ni ligne de base, ce jeu de données ne permet pas de dater les premiers signes.
- Par poisson, la vitesse seule repère 80 % des injectés en 30 s, pour 5 % de fausses alertes ; un score à
  7 indicateurs ne fait pas mieux.
- Dans des bassins virtuels de 8 poissons, le suivi individuel signale un seul injecté dans 60 à 76 % des cas ; la
  mesure globale du bassin, dans moins de 10 %.

## Données

FHA-MTT : Chen et al., 2026, *Smart Agricultural Technology* 13:101830,
[doi:10.1016/j.atech.2026.101830](https://doi.org/10.1016/j.atech.2026.101830). Données sur Zenodo,
[doi:10.5281/zenodo.17518455](https://doi.org/10.5281/zenodo.17518455), licence CC BY 4.0.

Les vidéos et les trajectoires sont trop lourdes pour ce dépôt (dossier `data/` exclu). Les étapes 1 à 7
ci-dessous les recréent.

## Étapes

| # | Étape | Fichier | Où |
|---|---|---|---|
| 1 | Conversion des vidéos `.video` en MP4 (sans réencodage) | `convertir_videos.py` | PC, FFmpeg |
| 2 | Annotations LabelMe → format YOLO-pose | `labelme_vers_yolo_pose.py` | PC |
| 3 | Contrôle de fuite (découpage par blocs de temps et témoin aléatoire) | `decoupage_par_blocs.py` | PC |
| 4 | 45 extraits de 3 min, un toutes les 10 min | `extraire_fenetres.py` | PC |
| 5 | Entraînement de YOLOv8n-pose | `kaggle/01_entrainement_pose.ipynb` | Kaggle, GPU |
| 6 | Comparaison de 7 réglages de suivi | `kaggle/02a_reglage_suivi.ipynb` | Kaggle, GPU |
| 7 | Suivi des 45 extraits (TrackTrack) | `kaggle/02_suivi.ipynb` | Kaggle, GPU |
| 8 | Q1 : indicateurs et écart entre groupes | `analyse_q1.py` | PC |
| 9 | Q2 : alerte par poisson et pour le bassin | `analyse_q2.py` | PC, environ 2 min |
| 10 | Vérification de la mesure globale par les pixels | `verifier_n0_pixels.py` | PC, 10 à 15 min |

Chaque script explique en tête ce qu'il lit, ce qu'il écrit et ses choix de méthode. Les résultats arrivent dans
`results/q1/`, `results/q2/` et `results/n0_pixels/`. La graine est fixée (42) : deux exécutions donnent les
mêmes chiffres.

## Installation (PC)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
winget install -e --id Gyan.FFmpeg
```

Les notebooks Kaggle installent eux-mêmes Ultralytics et `lap`. Ils demandent l'accélérateur GPU et l'accès à
Internet.

## Outils

Tout est gratuit : Python 3.12, Ultralytics 8.4 (YOLOv8-pose, BoT-SORT, ByteTrack, TrackTrack ; licence AGPL-3.0),
OpenCV, FFmpeg, pandas, SciPy, statsmodels, matplotlib, et les GPU gratuits de Kaggle.

## Auteur

Clément Cornière, 2026.
