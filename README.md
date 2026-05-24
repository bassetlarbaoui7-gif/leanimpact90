# LI90 — Lean Impact 90

> Le système nerveux industriel : de l'incident à l'excellence automatiquement.

Logiciel d'amélioration continue pour usines, conçu pour réduire la charge
mentale des opérateurs, accélérer les décisions des responsables, et rendre
l'usine fluide.

## Démarrer en local

```bash
pip install -r requirements.txt
python -m streamlit run landing.py
```

L'application s'ouvre sur `http://localhost:8501`.

## Stack

- **Streamlit** — interface
- **SQLite** — persistance locale
- **LightGBM + SHAP + ONNX** — moteur Ishikawa cause racine IA
- **Plotly / Three.js** — visualisations

## Modules principaux

- `landing.py` — page d'accueil 3D
- `pages/mission_control.py` — nouvelle interface unifiée (beta)
- `pages/app.py` — ancienne interface deux vues (Stabiliser / Projet AC)
- `core/db.py` — base SQLite et helpers CRUD
- `core/workflow.py` — machine à états du projet d'amélioration continue
- `vue_b/` — les 6 fonctionnalités du workflow d'AC

## Statut

MVP V1 — livraison prévue août 2026.
