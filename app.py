"""
LI90 MVP - Application Streamlit.
Pipeline complet : import -> selection -> nettoyage -> analyse -> affichage -> PDF.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis import analyze_all, clean_data
from data_loader import list_excel_sheets, load_file
from drift import detect_drift
from losses import analyze_losses
from report import generate_pdf
from shift_analysis import analyze_shifts

# Traduction FR des patterns Nelson pour affichage
PATTERN_LABELS = {
    "none": "Aucun",
    "hors_controle": "Hors controle (point extreme)",
    "shift": "Decalage de moyenne",
    "shift_partiel": "Decalage partiel",
    "trend": "Tendance (derive)",
    "instabilite": "Instabilite",
    "stratification": "Stratification (sigma sous-estime)",
}

st.set_page_config(page_title="LI90 - MVP", layout="wide")
st.title("LI90 — Analyse de derives machine")
st.caption(
    "MVP v0.2 — Shewhart + Nelson + drift (EWMA/CUSUM) + shift + PDF"
)

# -----------------------------------------------------------------------------
# 1. Import
# -----------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Importer un fichier Excel ou CSV",
    type=["xlsx", "xls", "csv"],
)

if uploaded is None:
    st.info("Depose un fichier pour commencer.")
    st.stop()

# Selection de feuille pour les fichiers Excel multi-feuilles
selected_sheet = None
if uploaded.name.lower().endswith((".xlsx", ".xls")):
    try:
        sheets = list_excel_sheets(uploaded)
        uploaded.seek(0)  # rewind apres lecture
        if len(sheets) > 1:
            selected_sheet = st.selectbox(
                "Feuille Excel a analyser",
                options=sheets,
                index=0,
            )
    except Exception as e:
        st.warning(f"Impossible de lister les feuilles : {e}")

try:
    df, load_report = load_file(uploaded, uploaded.name, sheet=selected_sheet)
except ValueError as e:
    st.error(f"Erreur de lecture : {e}")
    st.stop()
except Exception as e:
    st.error(f"Erreur inattendue de lecture : {e}")
    st.stop()

# Rapport de chargement (tracabilite audit qualite)
with st.expander("Rapport de chargement", expanded=False):
    st.caption(load_report.summary())
    if load_report.warnings:
        for warn in load_report.warnings:
            st.warning(warn)

st.subheader("Apercu des donnees")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"{len(df)} lignes · {len(df.columns)} colonnes")

# -----------------------------------------------------------------------------
# 2. Selection de colonnes
# -----------------------------------------------------------------------------
st.subheader("Selection des parametres machine")

# On propose uniquement les colonnes numeriques reelles :
# - exclut les dtypes datetime (timestamp converti en nanosecondes = bruit)
# - exclut les colonnes 100% NaN apres conversion numerique
candidate_cols = [
    c for c in df.columns
    if not pd.api.types.is_datetime64_any_dtype(df[c])
    and pd.to_numeric(df[c], errors="coerce").notna().any()
]

selected = st.multiselect(
    "Colonnes a analyser",
    options=candidate_cols,
    default=candidate_cols[:5],
)

if not selected:
    st.warning("Selectionne au moins une colonne.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Nettoyage + analyse
# -----------------------------------------------------------------------------
cleaned = clean_data(df, selected)
st.caption(f"Apres nettoyage : {len(cleaned)} lignes valides "
           f"(sur {len(df)} lignes importees).")

if cleaned.empty:
    st.error("Aucune donnee exploitable apres nettoyage.")
    st.stop()

results = analyze_all(cleaned)

# -----------------------------------------------------------------------------
# 4. Affichage des resultats
# -----------------------------------------------------------------------------
st.subheader("Resultats — classes par criticite")

# Vue principale compacte : Shewhart + pattern Nelson dominant traduit en FR
display_df = results[["parameter", "n", "mean", "std", "lcl", "ucl",
                      "violations", "criticality", "nelson_total",
                      "dominant_pattern"]].copy()
display_df["dominant_pattern"] = display_df["dominant_pattern"].map(
    PATTERN_LABELS).fillna(display_df["dominant_pattern"])
display_df = display_df.rename(columns={
    "nelson_total": "nelson_signaux",
    "dominant_pattern": "pattern_dominant",
})

styled = display_df.style.format({
    "mean": "{:.3f}",
    "std": "{:.3f}",
    "lcl": "{:.3f}",
    "ucl": "{:.3f}",
    "criticality": "{:.2f} %",
}).background_gradient(subset=["criticality"], cmap="Reds")

st.dataframe(styled, use_container_width=True)

# Detail Nelson (8 regles) dans un expander pour ne pas surcharger la vue
with st.expander("Detail des 8 regles de Nelson", expanded=False):
    nelson_cols = ["parameter"] + [f"nelson_{i}" for i in range(1, 9)] + \
                  ["nelson_total"]
    st.dataframe(results[nelson_cols], use_container_width=True)
    st.caption(
        "Regle 1 : point > 3 sigma · Regle 2 : 9 points meme cote · "
        "Regle 3 : 6 points monotones · Regle 4 : 14 points alternants · "
        "Regle 5 : 2/3 > 2 sigma · Regle 6 : 4/5 > 1 sigma · "
        "Regle 7 : 15 points dans +/-1 sigma · Regle 8 : 8 points > 1 sigma"
    )

# -----------------------------------------------------------------------------
# 5. Graphiques (cartes de controle par parametre)
# -----------------------------------------------------------------------------
st.subheader("Cartes de controle")

for _, row in results.iterrows():
    param = row["parameter"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(cleaned[param].values, marker="o", linewidth=0.8, markersize=3)
    ax.axhline(row["mean"], color="green", linestyle="--",
               linewidth=1, label="Moyenne")
    ax.axhline(row["ucl"], color="red", linestyle="--",
               linewidth=1, label="UCL")
    ax.axhline(row["lcl"], color="red", linestyle="--",
               linewidth=1, label="LCL")
    ax.set_title(
        f"{param} — {int(row['violations'])} violation(s) / {int(row['n'])} "
        f"({row['criticality']:.1f} %)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------------------------
# 5bis. Detection avancee (drift EWMA/CUSUM + shift)
# -----------------------------------------------------------------------------
st.subheader("Detection avancee")

with st.expander("Derives lentes (EWMA) et cumulatives (CUSUM)", expanded=False):
    try:
        drift_df = detect_drift(cleaned, columns=selected)
        drifted = drift_df[drift_df["drift_detected"]]
        if drifted.empty:
            st.info(
                "Aucune derive detectee par EWMA ou CUSUM "
                "sur les parametres selectionnes."
            )
        else:
            st.warning(
                f"Derive(s) detectee(s) sur {len(drifted)} parametre(s) : "
                f"{', '.join(drifted['parameter'].tolist())}"
            )
        st.dataframe(drift_df, use_container_width=True)
        st.caption(
            "EWMA (lambda=0.2, L=3) detecte les derives lentes de moyenne. "
            "CUSUM (k=0.5, h=4) detecte les sauts cumulatifs."
        )
    except Exception as e:
        st.error(f"Calcul de drift impossible : {e}")
        drift_df = None

with st.expander("Analyse par shift (equipes A/B/C)", expanded=False):
    shift_result = analyze_shifts(df, param_cols=selected)
    if not shift_result["detected"]:
        st.info(f"Analyse shift non disponible : {shift_result['reason']}")
    else:
        st.success(
            f"Colonne shift detectee : '{shift_result['shift_column']}' "
            f"({shift_result['n_shifts']} modalites)"
        )
        st.markdown("**Stabilite par shift (CV = ecart-type / moyenne)**")
        stability = shift_result["stability"]
        if not stability.empty:
            styled_stab = stability.style.format({
                "mean": "{:.3f}", "std": "{:.3f}", "cv": "{:.4f}",
            }).background_gradient(subset=["cv"], cmap="Oranges")
            st.dataframe(styled_stab, use_container_width=True)

        st.markdown("**Comparaison inter-shift (Kruskal-Wallis)**")
        comparison = shift_result["comparison"]
        if not comparison.empty:
            significant = comparison[comparison["significatif_5pct"]]
            if not significant.empty:
                st.warning(
                    f"Ecart significatif entre shifts sur {len(significant)} "
                    f"parametre(s) (p < 0.05) : "
                    f"{', '.join(significant['parameter'].tolist())}"
                )
            st.dataframe(comparison, use_container_width=True)

# Persist pour export PDF et vague 3/4
st.session_state["drift_results"] = drift_df
st.session_state["shift_results"] = shift_result

# -----------------------------------------------------------------------------
# 6. Analyse des pertes (optionnelle)
# -----------------------------------------------------------------------------
st.subheader("Analyse des pertes — correlation parametres / defauts")

losses_results = None
analyser_pertes = st.checkbox(
    "Activer l'analyse des pertes",
    help="Necessite une colonne de defauts (comptage par ligne).",
)

if analyser_pertes:
    candidate_defaut = [
        c for c in df.columns
        if c not in selected
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    if not candidate_defaut:
        st.warning(
            "Aucune colonne numerique disponible comme colonne de defauts. "
            "Assure-toi qu'une colonne distincte des parametres contient des comptages."
        )
    else:
        defaut_col = st.selectbox("Colonne defaut", options=candidate_defaut)
        volume_input = st.number_input(
            "Volume total produit (0 = ne pas calculer le PPM)",
            min_value=0, value=0, step=1000,
        )
        volume_total = float(volume_input) if volume_input > 0 else None

        if st.button("Lancer l'analyse des pertes"):
            try:
                losses_results = analyze_losses(
                    df,
                    param_columns=selected,
                    defaut_column=defaut_col,
                    volume_total=volume_total,
                )
            except ValueError as e:
                st.error(f"Erreur : {e}")
                losses_results = None

        if losses_results is not None:
            st.markdown("**Correlations (triees par |r| decroissant)**")
            corr_display = losses_results["correlations"].copy()
            st.dataframe(corr_display, use_container_width=True)

            if losses_results["ppm"] is not None:
                st.metric("PPM global", f"{losses_results['ppm']:.0f}")
            else:
                st.caption("PPM non calcule (volume total non renseigne).")

            # Persist pour export PDF
            st.session_state["losses_results"] = losses_results

# -----------------------------------------------------------------------------
# 7. Export PDF
# -----------------------------------------------------------------------------
st.subheader("Export")

if st.button("Generer le rapport PDF"):
    tmp_path = Path(tempfile.gettempdir()) / "rapport_li90.pdf"
    generate_pdf(
        results,
        str(tmp_path),
        source_file=uploaded.name,
        losses_results=st.session_state.get("losses_results"),
    )
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
    st.download_button(
        label="Telecharger le rapport PDF",
        data=pdf_bytes,
        file_name="rapport_li90.pdf",
        mime="application/pdf",
    )
