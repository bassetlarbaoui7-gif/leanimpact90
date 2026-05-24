"""
LI90 - Application desktop (point d'entree moderne).
Lance avec : python -m streamlit run home.py
Empaquetable en .exe avec PyInstaller.
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from analysis import analyze_all, clean_data
from data_loader import list_excel_sheets, load_file
from data_quality import audit_dataframe, summary_stats
from dashboard import (
    render_kpi_header, control_chart, pareto_chart,
    correlation_heatmap, render_active_alerts,
)
from drift import detect_drift
from ishikawa_ui import render_ishikawa_page, render_history_panel
from losses import analyze_losses
from report import generate_pdf
from shift_analysis import analyze_shifts
from ui_theme import (
    inject_theme, page_header, section, pill,
    COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_DANGER, COLOR_OK,
    COLOR_BORDER,
)


# ---------------------------------------------------------------------------
# Setup global
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LI90 - Amelioration continue",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

DB_PATH = Path("li90.db")


# ---------------------------------------------------------------------------
# Helpers state
# ---------------------------------------------------------------------------
def _ensure_state() -> None:
    defaults = {
        "df": None,
        "source_name": "",
        "selected_params": [],
        "results_spc": None,        # DataFrame
        "results_drift": None,       # DataFrame
        "results_losses": None,      # dict
        "results_shifts": None,      # dict
        "defaut_col": "",
        "shift_col": "",
        "volume_total": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ensure_state()


def _safe_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _alerts_from_results(results_spc: pd.DataFrame | None) -> list[dict]:
    """Construit la liste d'alertes a partir du DataFrame d'analyse SPC."""
    if results_spc is None or results_spc.empty:
        return []
    alerts: list[dict] = []
    for _, row in results_spc.iterrows():
        viol = _safe_int(row.get("violations", 0))
        crit = float(row.get("criticality", 0.0))
        if viol > 0 or crit > 0:
            sev = "danger" if crit >= 1.0 or viol >= 3 else "warn"
            alerts.append({
                "parameter": str(row["parameter"]),
                "message": (
                    f"{viol} violation(s) hors limite "
                    f"&middot; criticite {crit:.2f}%"
                ),
                "severity": sev,
                "timestamp": "",
            })
    # Tri par criticite decroissante
    return alerts


def _file_uploader_block() -> pd.DataFrame | None:
    uploaded = st.file_uploader(
        "Glisser-deposer un fichier Excel ou CSV",
        type=["xlsx", "xls", "csv"],
        key="main_uploader",
    )
    if uploaded is None:
        return None
    selected_sheet = None
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        try:
            sheets = list_excel_sheets(uploaded)
            uploaded.seek(0)
            if len(sheets) > 1:
                selected_sheet = st.selectbox("Feuille", options=sheets, index=0)
        except Exception as e:
            st.warning(f"Impossible de lister les feuilles : {e}")
    try:
        df, load_report = load_file(uploaded, uploaded.name, sheet=selected_sheet)
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        return None

    with st.expander("Rapport de chargement", expanded=False):
        st.caption(load_report.summary())
        for warn in load_report.warnings:
            st.warning(warn)

    st.session_state["df"] = df
    st.session_state["source_name"] = uploaded.name
    # Reset des analyses precedentes
    st.session_state["results_spc"] = None
    st.session_state["results_drift"] = None
    st.session_state["results_losses"] = None
    st.session_state["results_shifts"] = None
    return df


# ---------------------------------------------------------------------------
# Sidebar - navigation (refonte v0 : Logo LI90 dark+orange, role, Vue A/B)
# ---------------------------------------------------------------------------
ROLE_LABELS = {
    "operator":    "Operateur",
    "technician":  "Technicien N+1",
    "maintenance": "Resp. Maintenance",
    "ac_manager":  "Responsable AC",
    "production":  "Resp. Production",
    "ceo":         "Direction",
}

NAV_A = [
    "Dashboard",
    "Import & analyse SPC",
    "Analyse des pertes",
    "Analyse inter-shift",
    "Ishikawa IA (F3)",
    "Historique RCA",
    "Reglages",
]

# Filtre de navigation Vue A selon le role.
# Resp. AC = referent qualite -> voit tout.
# Direction = vue strategique -> KPIs + analyses transverses.
NAV_A_BY_ROLE = {
    "operator": [
        "Dashboard", "Ishikawa IA (F3)",
    ],
    "technician": [
        "Dashboard", "Import & analyse SPC", "Analyse des pertes",
        "Ishikawa IA (F3)", "Historique RCA",
    ],
    "maintenance": [
        "Dashboard", "Import & analyse SPC", "Analyse des pertes",
        "Ishikawa IA (F3)", "Historique RCA",
    ],
    "ac_manager": NAV_A,  # tout
    "production": [
        "Dashboard", "Analyse des pertes", "Analyse inter-shift",
        "Ishikawa IA (F3)", "Historique RCA",
    ],
    "ceo": [
        "Dashboard", "Analyse inter-shift", "Ishikawa IA (F3)",
        "Historique RCA",
    ],
}

NAV_B = [
    "1. Collecte terrain",
    "2. Cadrage du probleme",
    "3. Cause racine IA",
    "4. Validation distribuee",
    "5. Solution & faisabilite",
    "6. Suivi d'action",
]

# Defaults session
if "view" not in st.session_state:
    st.session_state["view"] = "A"
if "role" not in st.session_state:
    st.session_state["role"] = "ac_manager"

with st.sidebar:
    # Logo LI90 - dark + orange (AlignPro edition)
    st.markdown(
        f"""
        <div style="padding:0.6rem 0 1rem 0;
                    border-bottom:1px solid {COLOR_BORDER};
                    margin-bottom:0.8rem;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:36px; height:36px;
                            background:linear-gradient(135deg,#f97316,#ea580c);
                            border-radius:9px; display:flex;
                            align-items:center; justify-content:center;
                            color:white; font-weight:800; font-size:12px;
                            box-shadow:0 0 18px rgba(249,115,22,0.35);">L90</div>
                <div>
                    <div style="font-weight:800; font-size:15px;
                                color:{COLOR_TEXT};">LI90</div>
                    <div style="font-size:11px; color:{COLOR_TEXT_MUTED};">
                        AlignPro edition
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Selecteur de role
    role_keys = list(ROLE_LABELS.keys())
    role_lbls = list(ROLE_LABELS.values())
    cur_role  = st.session_state.get("role", "ac_manager")
    cur_idx   = role_keys.index(cur_role) if cur_role in role_keys else 3
    sel_lbl   = st.selectbox(
        "Role",
        options=role_lbls,
        index=cur_idx,
        label_visibility="collapsed",
        help="Filtre la lecture du logiciel selon ton role",
    )
    st.session_state["role"] = role_keys[role_lbls.index(sel_lbl)]

    # Switch Vue A / Vue B (radio horizontal)
    view = st.radio(
        "Vue",
        options=["A", "B"],
        format_func=lambda v: (
            "Stabiliser une ligne" if v == "A" else "Projet AC"
        ),
        horizontal=True,
        index=0 if st.session_state.get("view", "A") == "A" else 1,
        label_visibility="collapsed",
    )
    st.session_state["view"] = view

    st.markdown("---")

    # Navigation conditionnelle selon la vue ET le role (Vue A filtree)
    if view == "A":
        cur_role_key = st.session_state.get("role", "ac_manager")
        nav_opts = NAV_A_BY_ROLE.get(cur_role_key, NAV_A)
    else:
        nav_opts = NAV_B
    page = st.radio(
        "Navigation",
        options=nav_opts,
        label_visibility="collapsed",
    )

    st.markdown("---")
    df_loaded = st.session_state.get("df") is not None
    src = st.session_state.get("source_name") or "—"
    st.markdown(
        f"<div style='font-size:11px; color:{COLOR_TEXT_MUTED};'>"
        f"<b style='color:{COLOR_TEXT};'>Fichier :</b> {src[:32]}<br>"
        f"<b style='color:{COLOR_TEXT};'>Statut :</b>"
        f" {'Charge' if df_loaded else 'Aucun'}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("LI90 · Desktop v0.4")


# ---------------------------------------------------------------------------
# Header global
# ---------------------------------------------------------------------------
TITLES = {
    "Dashboard": ("Vue d'ensemble", "Synthese temps reel des indicateurs cles"),
    "Import & analyse SPC": ("Analyse statistique (SPC + Nelson + drift)",
                              "Detection automatique des derives parametre"),
    "Analyse des pertes": ("Analyse des pertes",
                            "Correlation parametres / defauts &middot; Cpk &middot; PPM"),
    "Analyse inter-shift": ("Analyse inter-shift",
                             "Comparaison des equipes A / B / C (Kruskal-Wallis)"),
    "Ishikawa IA (F3)": ("Causes racines (5M + IA)",
                          "Detection automatique + validation humaine"),
    "Historique RCA": ("Base de connaissance",
                        "Toutes les analyses validees - corpus IA"),
    "Reglages": ("Reglages", "Configuration moteur, modele IA, base"),
}

TITLES_B = {
    "1. Collecte terrain":
        ("Collecte terrain",
         "Operateur saisit l'incident, audio + requete SQL automatique"),
    "2. Cadrage du probleme":
        ("Cadrage du probleme",
         "QQOQCP - le Resp. AC cadre avec precision"),
    "3. Cause racine IA":
        ("Cause racine IA",
         "Moteur Ishikawa 5M avec LightGBM + SHAP + ONNX"),
    "4. Validation distribuee":
        ("Validation distribuee",
         "Resp. Prod + Tech N+1 valident sans reunion"),
    "5. Solution & faisabilite":
        ("Solution & faisabilite",
         "Etude automatique + gains traduits par role"),
    "6. Suivi d'action":
        ("Suivi d'action",
         "Plan d'action priorise, statuts, ROI reel"),
}

# ---------------------------------------------------------------------------
# Routage Vue B : placeholders pour les 6 fonctionnalites (etape 2 du build)
# ---------------------------------------------------------------------------
if st.session_state.get("view") == "B":
    title_b, subtitle_b = TITLES_B.get(page, (page, ""))
    page_header(title_b, subtitle_b)
    role_lbl = ROLE_LABELS.get(st.session_state.get("role", "ac_manager"), "—")
    st.markdown(
        f"""
        <div class="empty-state" style="margin-top: 32px;">
            <div class="empty-state-icon">🚧</div>
            <div style="font-size:18px; color:{COLOR_TEXT};">
                <b>Etape 2 du build en construction</b>
            </div>
            <div style="font-size:13px; margin-top:10px;
                        color:{COLOR_TEXT_MUTED}; max-width:540px;
                        margin-left:auto; margin-right:auto; line-height:1.6;">
                Cette fonctionnalite arrive avec la livraison du squelette
                Vue B (base SQLite + workflow + 6 pages connectees).<br><br>
                Vue actuelle :
                <b style="color:{COLOR_PRIMARY};">Projet AC</b> &middot;
                Role :
                <b style="color:{COLOR_PRIMARY};">{role_lbl}</b><br><br>
                Pour tester ce qui est deja construit, bascule sur
                <b style="color:{COLOR_TEXT};">Stabiliser une ligne</b>
                dans la sidebar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Routage Vue A : header + cascade if/elif existante
title, subtitle = TITLES[page]
page_header(title, subtitle)


# ===========================================================================
# PAGE : DASHBOARD
# ===========================================================================
if page == "Dashboard":
    df = st.session_state.get("df")
    results_spc = st.session_state.get("results_spc")
    losses_res = st.session_state.get("results_losses")

    if df is None:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div><b>Aucune donnee chargee</b></div>
                <div style="font-size:13px; margin-top:6px;">
                    Va dans <b>Import & analyse SPC</b> pour charger un fichier.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        n_points = len(df)
        num_df = df.select_dtypes(include="number")
        n_params = len(num_df.columns)

        # KPIs
        n_alerts = 0
        if results_spc is not None and not results_spc.empty:
            n_alerts = int(results_spc["violations"].sum())
        ppm = None
        if losses_res and losses_res.get("ppm") is not None:
            ppm = float(losses_res["ppm"])
        cpk_mean = None
        if losses_res and isinstance(losses_res.get("cpk"), pd.DataFrame) \
                and not losses_res["cpk"].empty:
            cpk_mean = float(losses_res["cpk"]["cpk"].mean())

        render_kpi_header(n_points, n_params, n_alerts, ppm=ppm, cpk=cpk_mean)

        col_charts, col_alerts = st.columns([2, 1], gap="large")

        # === Graphiques ===
        with col_charts:
            with st.container(border=True):
                section("Cartes de controle",
                        "Top 3 parametres avec le plus de variabilite")
                # Top 3 par std normalisee
                if not num_df.empty:
                    scores = []
                    for col in num_df.columns:
                        s = num_df[col].dropna()
                        if len(s) >= 2 and s.mean() != 0:
                            scores.append((col, abs(s.std() / s.mean())))
                    scores.sort(key=lambda x: -x[1])
                    top3 = [c for c, _ in scores[:3]] or list(num_df.columns[:3])
                    if top3:
                        tabs = st.tabs(top3)
                        for i, col in enumerate(top3):
                            with tabs[i]:
                                s = num_df[col].dropna()
                                if len(s) >= 2:
                                    st.plotly_chart(
                                        control_chart(s, col),
                                        use_container_width=True,
                                    )
                                else:
                                    st.info(
                                        "Pas assez de donnees pour la "
                                        "carte de controle.")

            with st.container(border=True):
                section("Pareto des parametres critiques",
                        "Score base sur le coefficient de variation")
                scores_d = {}
                for col in num_df.columns:
                    s = num_df[col].dropna()
                    if len(s) >= 2 and s.mean() != 0:
                        scores_d[col] = float(abs(s.std() / s.mean()) * 100)
                st.plotly_chart(pareto_chart(scores_d, top_n=10),
                                use_container_width=True)

            if len(num_df.columns) >= 2:
                with st.container(border=True):
                    section("Correlations", "Matrice Pearson")
                    try:
                        corr = num_df.corr().round(2)
                        st.plotly_chart(correlation_heatmap(corr),
                                        use_container_width=True)
                    except Exception as e:
                        st.warning(f"Correlation non calculable : {e}")

        # === Alertes ===
        with col_alerts:
            with st.container(border=True):
                section("Alertes actives", "Triees par criticite")
                alerts = _alerts_from_results(results_spc)
                render_active_alerts(alerts[:12])

            # Mini-bloc recommandations
            with st.container(border=True):
                section("Prochaines actions",
                        "Recommandations basees sur l'analyse")
                if not alerts:
                    st.markdown(
                        f"<div style='font-size:13px; color:{COLOR_TEXT};'>"
                        f"Tout est sous controle. Continue le suivi quotidien."
                        f"</div>", unsafe_allow_html=True)
                else:
                    top_alert = alerts[0]
                    st.markdown(
                        f"<div style='font-size:13px;'>"
                        f"1. <b>Investiguer {top_alert['parameter']}</b> "
                        f"({top_alert['message']})<br>"
                        f"2. Ouvrir une analyse Ishikawa (F3) sur ce defaut<br>"
                        f"3. Lancer l'analyse des pertes pour confirmer la "
                        f"correlation"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ===========================================================================
# PAGE : IMPORT & ANALYSE SPC
# ===========================================================================
elif page == "Import & analyse SPC":
    df = _file_uploader_block()
    if df is None:
        df = st.session_state.get("df")

    if df is None:
        st.info("Importe un fichier pour demarrer l'analyse.")
    else:
        with st.container(border=True):
            section("Apercu des donnees",
                    f"{len(df)} lignes &middot; {len(df.columns)} colonnes")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        # Panneau qualite donnees (reflexe industriel : audit avant analyse)
        with st.container(border=True):
            qsum = summary_stats(df)
            section(
                "Audit qualite des donnees",
                f"{qsum['n_ok']} OK &middot; {qsum['n_warn']} warn "
                f"&middot; {qsum['n_danger']} danger",
            )
            audit = audit_dataframe(df)
            if not audit.empty:
                # Colorisation verdict
                def _style_row(row):
                    color = {
                        "ok": "background-color:#ECFDF5",
                        "warn": "background-color:#FFFBEB",
                        "danger": "background-color:#FEF2F2",
                    }.get(row["verdict"], "")
                    return [color] * len(row)
                try:
                    styled = audit.style.apply(_style_row, axis=1)
                    st.dataframe(styled, use_container_width=True,
                                 hide_index=True)
                except Exception:
                    st.dataframe(audit, use_container_width=True,
                                 hide_index=True)
                if qsum["n_danger"] > 0:
                    st.warning(
                        "Des colonnes sont en rouge (>50% de valeurs "
                        "manquantes). Evite de les inclure dans l'analyse."
                    )

        candidate_cols = [
            c for c in df.columns
            if not pd.api.types.is_datetime64_any_dtype(df[c])
            and pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        with st.container(border=True):
            section("Parametres a analyser")
            selected = st.multiselect(
                "Colonnes",
                options=candidate_cols,
                default=st.session_state.get("selected_params") or
                        candidate_cols[: min(5, len(candidate_cols))],
                key="spc_selected",
                help=(
                    "Selectionne les parametres machine a surveiller. "
                    "Seules les colonnes numeriques non-dates sont proposees."
                ),
            )
            st.session_state["selected_params"] = selected

            launch = st.button("Lancer l'analyse", type="primary")

        if launch:
            if not selected:
                st.error("Selectionne au moins un parametre.")
            else:
                try:
                    cleaned = clean_data(df, selected)
                    dropped = cleaned.attrs.get("dropped_columns", [])
                    if dropped:
                        st.warning(
                            f"Colonnes ecartees (trop de valeurs manquantes) : "
                            f"{', '.join(dropped)}"
                        )
                    if cleaned.empty:
                        st.error(
                            "Aucune donnee exploitable apres nettoyage. "
                            "Verifie la qualite des colonnes selectionnees."
                        )
                        st.stop()
                    results = analyze_all(cleaned)
                    drift = detect_drift(cleaned)
                    st.session_state["results_spc"] = results
                    st.session_state["results_drift"] = drift
                    st.success(
                        f"Analyse terminee : {len(cleaned.columns)} parametre(s), "
                        f"{int(results['violations'].sum())} violation(s) "
                        f"Shewhart, "
                        f"{int(drift['drift_detected'].sum())} derive(s) "
                        f"detectee(s)."
                    )
                except Exception as e:
                    st.error(f"Erreur d'analyse : {e}")
                    with st.expander("Trace technique"):
                        st.code(traceback.format_exc())

        # Affichage resultats
        results = st.session_state.get("results_spc")
        drift = st.session_state.get("results_drift")
        if results is not None and not results.empty:
            with st.container(border=True):
                section("Resultats SPC",
                        "Triés par criticite decroissante")
                # Tableau enrichi
                display_cols = ["parameter", "n", "mean", "std", "lcl", "ucl",
                                "violations", "criticality", "dominant_pattern"]
                display_cols = [c for c in display_cols if c in results.columns]
                st.dataframe(
                    results[display_cols].round(3),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.container(border=True):
                section("Detail par parametre")
                cleaned = clean_data(df, selected)
                for _, row in results.iterrows():
                    param = row["parameter"]
                    viol = _safe_int(row["violations"])
                    badge = pill(f"{viol} violation(s)",
                                 "danger" if viol >= 3 else
                                 "warn" if viol > 0 else "ok")
                    with st.expander(
                        f"{param} — {viol} violation(s) "
                        f"&middot; criticite {row['criticality']:.2f}%",
                    ):
                        s = cleaned[param].dropna()
                        if len(s) >= 2:
                            st.plotly_chart(control_chart(s, param),
                                            use_container_width=True)
                        st.markdown(badge, unsafe_allow_html=True)

        if drift is not None and not drift.empty:
            with st.container(border=True):
                section("Detection de derive (EWMA + CUSUM)",
                        "Derives lentes et sauts cumulatifs")
                st.dataframe(drift, use_container_width=True, hide_index=True)

        # Export PDF
        if results is not None and not results.empty:
            with st.container(border=True):
                section("Export PDF", "Rapport pour diffusion direction")
                if st.button("Generer le rapport PDF"):
                    try:
                        tmp_path = Path(tempfile.gettempdir()) / "rapport_li90.pdf"
                        generate_pdf(
                            results, str(tmp_path),
                            source_file=st.session_state.get("source_name", ""),
                            losses_results=st.session_state.get("results_losses"),
                            shifts_results=st.session_state.get("results_shifts"),
                        )
                        with open(tmp_path, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            "Telecharger le PDF",
                            data=pdf_bytes,
                            file_name="rapport_li90.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.error(f"Erreur export PDF : {e}")


# ===========================================================================
# PAGE : ANALYSE DES PERTES
# ===========================================================================
elif page == "Analyse des pertes":
    df = st.session_state.get("df")
    if df is None:
        st.info(
            "Importe un fichier dans 'Import & analyse SPC' "
            "avant de lancer l'analyse des pertes."
        )
    else:
        num_cols = [c for c in df.columns
                    if pd.to_numeric(df[c], errors="coerce").notna().any()]

        with st.container(border=True):
            section("Configuration",
                    "Choisis la colonne defauts et les parametres a tester")
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                defaut_col = st.selectbox(
                    "Colonne defauts (variable a expliquer)",
                    options=num_cols,
                    index=(num_cols.index(st.session_state["defaut_col"])
                           if st.session_state["defaut_col"] in num_cols else 0)
                          if num_cols else 0,
                    help=(
                        "Variable cible : nombre de defauts, taux de rebut, "
                        "ou toute autre mesure de non-qualite a expliquer."
                    ),
                )
            with c2:
                volume_total = st.number_input(
                    "Volume total produit (optionnel, pour PPM)",
                    min_value=0.0, value=float(st.session_state["volume_total"]),
                    step=1000.0,
                    help=(
                        "PPM = defauts / volume produit x 1 000 000. "
                        "Laisse 0 si tu ne connais pas le volume."
                    ),
                )
            with c3:
                st.write("")
                st.write("")
                launch_loss = st.button("Lancer l'analyse",
                                        type="primary",
                                        use_container_width=True)
            param_cols = st.multiselect(
                "Parametres a tester (cause potentielle)",
                options=[c for c in num_cols if c != defaut_col],
                default=[c for c in num_cols if c != defaut_col][:6],
            )
            st.session_state["defaut_col"] = defaut_col
            st.session_state["volume_total"] = volume_total

        if launch_loss:
            if not param_cols:
                st.error("Selectionne au moins un parametre.")
            else:
                try:
                    losses = analyze_losses(
                        df,
                        param_columns=param_cols,
                        defaut_column=defaut_col,
                        volume_total=volume_total if volume_total > 0 else None,
                    )
                    st.session_state["results_losses"] = losses
                    st.success("Analyse des pertes terminee.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    with st.expander("Trace"):
                        st.code(traceback.format_exc())

        losses = st.session_state.get("results_losses")
        if losses:
            # KPIs PPM
            ppm = losses.get("ppm")
            if ppm is not None:
                with st.container(border=True):
                    section(f"PPM global : {ppm:,.0f}".replace(",", " "),
                            "Defauts par million de pieces produites")

            # Correlations
            corr_df = losses.get("correlations")
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                with st.container(border=True):
                    section("Correlation parametres ↔ defauts",
                            "Triées par |r| decroissant - test retenu : "
                            "Pearson ou Spearman selon normalite")
                    show = corr_df[["param", "n", "test_retenu",
                                     "r", "p_value"]].copy()
                    show["r"] = show["r"].round(3)
                    show["p_value"] = show["p_value"].apply(
                        lambda x: f"{x:.2e}" if pd.notna(x) else "-")
                    st.dataframe(show, use_container_width=True,
                                 hide_index=True)
                    st.caption(
                        "💡 Une correlation forte (|r| ≥ 0.7) avec p-value "
                        "< 0.05 indique un parametre suspect. "
                        "Correlation ≠ causalite : valider terrain."
                    )

            # Cpk
            cpk_df = losses.get("cpk")
            if isinstance(cpk_df, pd.DataFrame) and not cpk_df.empty:
                with st.container(border=True):
                    section(
                        "Capabilite du processus (Cpk)",
                        "Cpk >= 1.33 = capable  |  1.0-1.33 = limite  |  "
                        "< 1 = insuffisant (pertes certaines)",
                    )
                    st.dataframe(cpk_df.round(3),
                                 use_container_width=True, hide_index=True)
                    st.caption(
                        "Cpk mesure la marge entre la variabilite reelle du "
                        "procede et les tolerances. C'est la reference "
                        "automobile (AIAG) et aerospatiale (AS9100)."
                    )


# ===========================================================================
# PAGE : ANALYSE INTER-SHIFT
# ===========================================================================
elif page == "Analyse inter-shift":
    df = st.session_state.get("df")
    if df is None:
        st.info("Importe un fichier au prealable.")
    else:
        cat_cols = [c for c in df.columns
                    if df[c].dtype == "object" or
                    df[c].nunique(dropna=True) <= 10]
        num_cols = [c for c in df.columns
                    if pd.to_numeric(df[c], errors="coerce").notna().any()]

        with st.container(border=True):
            section("Configuration", "Colonne shift + parametres a comparer")
            c1, c2 = st.columns([1, 1])
            with c1:
                shift_col = st.selectbox(
                    "Colonne shift (auto-detection si vide)",
                    options=[""] + cat_cols,
                    index=0,
                )
            with c2:
                params = st.multiselect(
                    "Parametres",
                    options=num_cols,
                    default=num_cols[:5],
                )
            launch_s = st.button("Comparer les shifts", type="primary")

        if launch_s:
            try:
                res = analyze_shifts(
                    df, param_cols=params,
                    shift_col=shift_col if shift_col else None,
                )
                st.session_state["results_shifts"] = res
                if res.get("detected"):
                    st.success(
                        f"Shift detecte : {res['shift_column']} "
                        f"({res['n_shifts']} equipe(s))"
                    )
                else:
                    st.warning(f"Aucun shift detecte ({res.get('reason', '')})")
            except Exception as e:
                st.error(f"Erreur : {e}")
                with st.expander("Trace"):
                    st.code(traceback.format_exc())

        res = st.session_state.get("results_shifts")
        if res and res.get("detected"):
            stab = res.get("stability")
            if isinstance(stab, pd.DataFrame) and not stab.empty:
                with st.container(border=True):
                    section("Stabilite par shift", "Moyenne, ecart-type, n")
                    st.dataframe(stab.round(3), use_container_width=True,
                                 hide_index=True)
            comp = res.get("comparison")
            if isinstance(comp, pd.DataFrame) and not comp.empty:
                with st.container(border=True):
                    section(
                        "Comparaison Kruskal-Wallis",
                        "p-value < 0.05 = au moins un shift differe "
                        "significativement des autres",
                    )
                    show = comp.copy()
                    if "p_value" in show.columns:
                        show["p_value"] = show["p_value"].apply(
                            lambda x: f"{x:.2e}" if pd.notna(x) else "-")
                    st.dataframe(show, use_container_width=True,
                                 hide_index=True)


# ===========================================================================
# PAGE : ISHIKAWA F3
# ===========================================================================
elif page == "Ishikawa IA (F3)":
    df = st.session_state.get("df")
    render_ishikawa_page(df=df, db_path=DB_PATH)


# ===========================================================================
# PAGE : HISTORIQUE
# ===========================================================================
elif page == "Historique RCA":
    with st.container(border=True):
        render_history_panel(db_path=DB_PATH)


# ===========================================================================
# PAGE : REGLAGES
# ===========================================================================
elif page == "Reglages":
    with st.container(border=True):
        section("Base de donnees locale", f"Fichier : {DB_PATH.absolute()}")
        if DB_PATH.exists():
            size_kb = DB_PATH.stat().st_size / 1024
            st.markdown(pill(f"Active &middot; {size_kb:.1f} KB", "ok"),
                        unsafe_allow_html=True)
        else:
            st.markdown(
                pill("Sera cree au premier enregistrement RCA", "info"),
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        section("Moteur IA F3",
                "Bascule auto TF-IDF -> LightGBM a 200+ cas")
        st.markdown(
            f"""
            <div style="font-size:13px; color:{COLOR_TEXT};">
                <b>Mode actuel :</b> TF-IDF + cosine similarity (V1)<br/>
                <b>Bascule prevue :</b> LightGBM + SHAP + ONNX (V2)<br/>
                <b>Donnees requises pour V2 :</b> 200 cas labellises minimum
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        section("A propos", "LI90 v0.3 — Logiciel desktop on-premise")
        st.markdown(
            f"""
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">
                Stack : Streamlit + PyInstaller + SQLite + Plotly + scikit-learn.<br/>
                Donnees : 100% locales, aucune transmission externe.<br/>
                Tests : 83 tests pytest, ruff clean.
            </div>
            """,
            unsafe_allow_html=True,
        )
