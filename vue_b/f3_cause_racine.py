"""
F3 - Cause racine IA.

Le moteur IA F3 V2 (LightGBM + SHAP + ONNX) propose les top causes par
branche M (5M Ishikawa). Reutilise integralement le code existant de
ishikawa_ui.render_ishikawa_page() qui contient :
  - chargement / entrainement du moteur SupervisedRootCauseEngine
  - contexte defaut (parametres / capteurs)
  - top-3 causes par M avec contributions SHAP
  - validation humaine + persistance en base
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ishikawa_ui import render_ishikawa_page

DB_PATH = Path(__file__).resolve().parent.parent / "li90.db"


def render() -> None:
    df = st.session_state.get("df")
    if df is None:
        st.info(
            "Aucun fichier de donnees charge en memoire. "
            "Bascule un instant sur Vue A → Import & analyse SPC pour "
            "charger un fichier, puis reviens ici. Tes resultats Vue B "
            "(projets, actions...) restent intacts."
        )
        return

    render_ishikawa_page(df=df, db_path=DB_PATH)
