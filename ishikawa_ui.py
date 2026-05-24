"""
Interface F3 - Ishikawa 5M + IA causale.
Design pro : diagramme arete de poisson Plotly + panneau suggestions IA.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ishikawa import (
    BRANCHES_5M, AUTO_BRANCHES, SEMI_AUTO_BRANCHES, MANUAL_BRANCHES,
    RcaCase, RootCauseEngine,
    auto_fill_machine_branch, auto_fill_matiere_branch,
    init_db, load_all_rca, save_rca, empty_5m_structure, branch_mode,
)
from inference import (
    SupervisedRootCauseEngine, MPrediction, ShapContribution,
)
from training import (
    M_BRANCHES, generate_synthetic_global_history, train_all_models,
)
from ui_theme import (
    COLOR_PRIMARY, COLOR_PRIMARY_LIGHT, COLOR_ACCENT,
    COLOR_OK, COLOR_WARN, COLOR_DANGER,
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_BORDER, COLOR_BG,
    ai_suggestion_card, section, pill, kpi_card, kpi_grid,
)


# ---------------------------------------------------------------------------
# Helpers V2 - moteur supervise
# ---------------------------------------------------------------------------
DEFAULT_MODELS_DIR = Path("./models")


def try_load_supervised_engine(
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> SupervisedRootCauseEngine | None:
    """
    Tente de charger les ONNX + metadata.json depuis models_dir.
    Retourne None silencieusement si absent ou corrompu (la V1 prendra le
    relais).
    """
    try:
        if not (models_dir / "metadata.json").exists():
            return None
        return SupervisedRootCauseEngine(models_dir=models_dir)
    except Exception:
        return None


def v2_status_html(engine: SupervisedRootCauseEngine | None) -> str:
    """Renvoie le HTML des 3 pills de statut par M (Machine, Matiere, Methode)."""
    if engine is None:
        return pill("Modele V2 non charge", kind="warn")
    avail = engine.is_available()
    reasons = engine.reasons_unavailable()
    parts = []
    for m in M_BRANCHES:
        if avail.get(m):
            parts.append(pill(f"IA {m} prete", kind="ok"))
        else:
            why = reasons.get(m, "indisponible")
            parts.append(pill(f"IA {m} : {why[:40]}", kind="warn"))
    return " ".join(parts)


def shap_explanation_html(contributions: list[ShapContribution]) -> str:
    """
    Tableau lisible des top-N contributions SHAP.
    Vert = pousse vers la cause / Rouge = pousse contre.
    """
    rows_html = []
    for c in contributions:
        sign_color = COLOR_OK if c.contribution >= 0 else COLOR_DANGER
        sign_arrow = "+" if c.contribution >= 0 else "-"
        delta = c.value - c.median
        delta_pct = (delta / c.median * 100) if c.median else 0
        rows_html.append(
            f"""
            <tr>
                <td style="padding:4px 8px;">{c.feature}</td>
                <td style="padding:4px 8px; text-align:right;">
                    <code>{c.value:.2f}</code>
                </td>
                <td style="padding:4px 8px; text-align:right; color:{COLOR_TEXT_MUTED};">
                    <code>{c.median:.2f}</code>
                </td>
                <td style="padding:4px 8px; text-align:right; color:{sign_color};">
                    <b>{sign_arrow}{abs(c.contribution):.3f}</b>
                </td>
                <td style="padding:4px 8px; text-align:right; color:{COLOR_TEXT_MUTED};">
                    {delta_pct:+.1f}%
                </td>
            </tr>
            """
        )
    return f"""
    <table style="width:100%; font-size:12px; border-collapse:collapse;">
      <thead>
        <tr style="background:{COLOR_BORDER}; color:{COLOR_TEXT};">
          <th style="padding:6px 8px; text-align:left;">Feature</th>
          <th style="padding:6px 8px; text-align:right;">Observe</th>
          <th style="padding:6px 8px; text-align:right;">Median historique</th>
          <th style="padding:6px 8px; text-align:right;">Contribution SHAP</th>
          <th style="padding:6px 8px; text-align:right;">Ecart</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
    """


def render_v2_models_panel(
    engine: SupervisedRootCauseEngine | None,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> SupervisedRootCauseEngine | None:
    """
    Panneau de gestion des modeles V2 :
      - statut par M (3 pills)
      - bouton 'Recharger' si modele present
      - bouton 'Entrainer sur historique synthetique' sinon (demo client)
    Retourne le moteur (mis a jour) ou None.
    """
    with st.expander(
        "Modele IA supervise V2 (LightGBM + SHAP + ONNX)",
        expanded=engine is None,
    ):
        st.markdown(v2_status_html(engine), unsafe_allow_html=True)
        st.markdown("")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Recharger les modeles", use_container_width=True):
                new_engine = try_load_supervised_engine(models_dir)
                if new_engine and new_engine.is_trained():
                    st.success(f"Modeles charges depuis {models_dir.resolve()}")
                    return new_engine
                else:
                    st.warning(
                        "Aucun modele exploitable trouve dans "
                        f"`{models_dir}`. Entraine un modele synthetique pour "
                        "demarrer."
                    )
        with col_b:
            if st.button(
                "Entrainer sur historique synthetique (demo)",
                use_container_width=True,
            ):
                with st.spinner("Generation + entrainement en cours..."):
                    df_synth = generate_synthetic_global_history(
                        n=400, seed=42,
                    )
                    report = train_all_models(
                        df_synth, output_dir=models_dir, min_cases=80,
                    )
                st.success(
                    f"Entrainement termine : "
                    f"{len(report.models_trained)}/{len(M_BRANCHES)} modeles. "
                    f"{len(report.models_skipped)} skip(s)."
                )
                with st.expander("Rapport d'entrainement"):
                    st.code(report.summary())
                return try_load_supervised_engine(models_dir)
    return engine


def render_v2_suggestions(
    engine: SupervisedRootCauseEngine,
    defect_context: dict[str, float],
) -> tuple[dict[str, MPrediction], dict[str, list[ShapContribution]]]:
    """
    Affiche le top-3 par M (Machine, Matiere, Methode) + expander SHAP.
    Retourne (predictions, explanations) pour pouvoir les enregistrer en SQLite.
    """
    multi = engine.predict_multi(defect_context, top_k=3)
    explanations = engine.explain_multi(defect_context, top_features=5)
    for m in M_BRANCHES:
        pred = multi.predictions[m]
        if not pred.available:
            st.markdown(
                ai_suggestion_card(
                    cause="(modele indisponible)",
                    branch=f"{m} (V2)",
                    confidence=0.0,
                    explanation=pred.reason_unavailable,
                ),
                unsafe_allow_html=True,
            )
            continue
        if pred.is_none and not pred.top_causes:
            st.markdown(
                ai_suggestion_card(
                    cause="Aucune cause detectee sur cette branche",
                    branch=f"{m} (V2)",
                    confidence=0.0,
                    explanation="Le modele ne voit pas de signal sur ce M. "
                                "A valider manuellement.",
                ),
                unsafe_allow_html=True,
            )
            continue
        # Top-3 cause par cause
        for rank, cp in enumerate(pred.top_causes, start=1):
            st.markdown(
                ai_suggestion_card(
                    cause=f"#{rank} - {cp.cause}",
                    branch=f"{m} (V2)",
                    confidence=cp.confidence,
                    explanation=(
                        "Modele supervise LightGBM. Ouvre le 'Pourquoi' "
                        "pour les contributions SHAP."
                        if rank == 1 else ""
                    ),
                ),
                unsafe_allow_html=True,
            )
        # Expander SHAP (top-1 seulement)
        contribs = explanations.get(m, [])
        if contribs:
            with st.expander(f"Pourquoi - top features SHAP {m}", expanded=False):
                st.markdown(
                    shap_explanation_html(contribs),
                    unsafe_allow_html=True,
                )
    return multi.predictions, explanations


# ---------------------------------------------------------------------------
# Diagramme arete de poisson
# ---------------------------------------------------------------------------
def fishbone_diagram(
    defect: str,
    causes_5m: dict[str, list[str]],
    height: int = 480,
) -> go.Figure:
    """
    Dessine un diagramme Ishikawa 5M avec Plotly.
    Axe central horizontal + 5 aretes diagonales + causes le long des aretes.
    """
    fig = go.Figure()

    # Parametres geometriques
    spine_start, spine_end = 0.05, 0.92
    y_center = 0.5
    # 3 branches au-dessus, 2 en-dessous
    branches_pos = [
        ("Machine", 0.25, 0.95),      # x, y_tip
        ("Methode", 0.45, 0.95),
        ("Matiere", 0.65, 0.95),
        ("Main-d'oeuvre", 0.30, 0.05),
        ("Milieu", 0.60, 0.05),
    ]

    # Ligne centrale (colonne vertebrale)
    fig.add_shape(
        type="line",
        x0=spine_start, y0=y_center, x1=spine_end, y1=y_center,
        line=dict(color=COLOR_PRIMARY, width=3),
    )
    # Tete (probleme)
    fig.add_shape(
        type="rect",
        x0=spine_end, y0=y_center - 0.08, x1=spine_end + 0.08, y1=y_center + 0.08,
        fillcolor=COLOR_PRIMARY, line=dict(color=COLOR_PRIMARY),
    )
    fig.add_annotation(
        x=spine_end + 0.04, y=y_center,
        text=f"<b>{defect or 'Defaut'}</b>",
        showarrow=False,
        font=dict(color="white", size=11, family="Inter, sans-serif"),
        xanchor="center", yanchor="middle",
    )

    # Branches principales
    for branch, x_base, y_tip in branches_pos:
        # Ligne diagonale du spine vers la tete de branche
        fig.add_shape(
            type="line",
            x0=x_base, y0=y_center, x1=x_base + 0.06, y1=y_tip,
            line=dict(color=COLOR_PRIMARY_LIGHT, width=2),
        )
        # Label de branche
        mode = branch_mode(branch)
        badge_color = {
            "auto": COLOR_OK,
            "semi-auto": COLOR_ACCENT,
            "manuel": COLOR_TEXT_MUTED,
        }[mode]
        fig.add_annotation(
            x=x_base + 0.06, y=y_tip,
            text=f"<b>{branch}</b><br><span style='font-size:9px; "
                 f"color:{badge_color}'>● {mode}</span>",
            showarrow=False,
            font=dict(color=COLOR_PRIMARY, size=12, family="Inter, sans-serif"),
            xanchor="center",
            yanchor="bottom" if y_tip > 0.5 else "top",
            align="center",
        )
        # Causes listees le long de la branche
        causes = causes_5m.get(branch, [])
        is_upper = y_tip > 0.5
        for i, cause in enumerate(causes[:5]):
            dy = (0.08 + i * 0.08) * (1 if is_upper else -1)
            # Trait court
            x_spine = x_base + (0.04 * (i + 1) / 5)
            y_line = y_center + dy
            fig.add_shape(
                type="line",
                x0=x_spine, y0=y_line,
                x1=x_spine - 0.05, y1=y_line,
                line=dict(color=COLOR_TEXT_MUTED, width=1),
            )
            fig.add_annotation(
                x=x_spine - 0.06, y=y_line,
                text=cause,
                showarrow=False,
                font=dict(size=9, color=COLOR_TEXT, family="Inter, sans-serif"),
                xanchor="right", yanchor="middle",
            )

    fig.update_xaxes(range=[0, 1.05], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# CSS local de la page (cards M, step headers, M-card)
# ---------------------------------------------------------------------------
_LOCAL_CSS = f"""
<style>
/* ----- design tokens locaux ------------------------------------- */
:root {{
    --f3-radius: 14px;
    --f3-shadow-1: 0 1px 2px rgba(15,23,42,.04), 0 1px 1px rgba(15,23,42,.03);
    --f3-shadow-2: 0 4px 12px rgba(15,23,42,.06), 0 2px 4px rgba(15,23,42,.04);
    --f3-shadow-card-hover: 0 8px 24px rgba(15,23,42,.08), 0 2px 6px rgba(15,23,42,.04);
    --f3-ease: cubic-bezier(0.2, 0.8, 0.2, 1);
}}
/* ----- step card (workflow numerote) ---------------------------- */
.f3-step {{
    background: white;
    border: 1px solid {COLOR_BORDER};
    border-radius: var(--f3-radius);
    padding: 22px 24px 24px 24px;
    margin-bottom: 16px;
    box-shadow: var(--f3-shadow-1);
}}
.f3-step-head {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}}
.f3-step-num {{
    width: 28px; height: 28px;
    background: {COLOR_PRIMARY};
    color: white;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0;
    font-variant-numeric: tabular-nums;
}}
.f3-step-title {{
    font-size: 15px;
    font-weight: 600;
    color: {COLOR_TEXT};
    letter-spacing: -0.01em;
}}
.f3-step-sub {{
    font-size: 12.5px;
    line-height: 1.55;
    color: {COLOR_TEXT_MUTED};
    margin: 4px 0 18px 40px;
    max-width: 70ch;
}}
/* ----- M card (Machine, Matiere, Methode) ----------------------- */
.m-card {{
    background: white;
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 18px 18px 14px 18px;
    height: 100%;
    transition: box-shadow 200ms var(--f3-ease),
                transform 200ms var(--f3-ease);
}}
.m-card:hover {{
    box-shadow: var(--f3-shadow-card-hover);
    transform: translateY(-1px);
}}
.m-card-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid {COLOR_BORDER};
}}
.m-card-title {{
    font-weight: 600;
    color: {COLOR_TEXT};
    font-size: 14px;
    letter-spacing: -0.005em;
}}
.m-card-status {{
    font-size: 10.5px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}
.m-card-status::before {{
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}}
.m-card-status.ok    {{ background: #ECFDF5; color: #047857; }}
.m-card-status.ok::before {{ background: #10B981; box-shadow: 0 0 0 3px #D1FAE5; }}
.m-card-status.warn  {{ background: #FFFBEB; color: #B45309; }}
.m-card-status.warn::before {{ background: #F59E0B; box-shadow: 0 0 0 3px #FEF3C7; }}
.m-card-status.off   {{ background: #F8FAFC; color: #64748B; }}
.m-card-status.off::before {{ background: #94A3B8; box-shadow: 0 0 0 3px #E2E8F0; }}
/* ----- liste de causes ranked ----------------------------------- */
.cause-rank {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 0;
    border-bottom: 1px solid #F1F5F9;
}}
.cause-rank:last-child {{ border-bottom: none; padding-bottom: 4px; }}
.cause-rank-num {{
    width: 22px; height: 22px;
    background: #F1F5F9;
    color: {COLOR_TEXT_MUTED};
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 11px;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
}}
.cause-rank-num.top {{
    background: {COLOR_PRIMARY};
    color: white;
    box-shadow: 0 2px 6px rgba(11,61,145,.20);
}}
.cause-rank-text {{
    flex: 1;
    font-size: 13px;
    color: {COLOR_TEXT};
    line-height: 1.35;
    letter-spacing: -0.005em;
}}
.cause-rank-pct {{
    font-size: 12px;
    color: {COLOR_PRIMARY};
    font-weight: 700;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
    min-width: 32px;
    text-align: right;
}}
.cause-rank-bar {{
    flex: 0 0 56px;
    height: 5px;
    background: #F1F5F9;
    border-radius: 999px;
    overflow: hidden;
}}
.cause-rank-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, {COLOR_PRIMARY}, {COLOR_PRIMARY_LIGHT});
    border-radius: 999px;
    transition: width 400ms var(--f3-ease);
}}
/* ----- empty state ---------------------------------------------- */
.m-card-empty {{
    padding: 36px 14px 22px 14px;
    text-align: center;
    color: {COLOR_TEXT_MUTED};
    font-size: 12.5px;
    line-height: 1.5;
}}
.m-card-empty svg {{
    width: 36px; height: 36px;
    stroke: #CBD5E1;
    margin-bottom: 8px;
}}
/* ----- micro respect des focus / hover natifs ------------------- */
.f3-step button:focus-visible,
.f3-step [role="button"]:focus-visible {{
    outline: 2px solid {COLOR_PRIMARY};
    outline-offset: 2px;
    border-radius: 6px;
}}
</style>
"""

_EMPTY_SVG = """
<svg viewBox="0 0 24 24" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
  <path d="M3 7l9-4 9 4-9 4-9-4zM3 7v10l9 4M21 7v10l-9 4"/>
</svg>
"""


def _step(num: int, title: str, subtitle: str = "") -> None:
    """Header de step uniforme (numerotation visible)."""
    sub_html = (f'<div class="f3-step-sub">{subtitle}</div>' if subtitle else "")
    st.markdown(
        f"""
        <div class="f3-step-head">
            <div class="f3-step-num">{num}</div>
            <div class="f3-step-title">{title}</div>
        </div>
        {sub_html}
        """,
        unsafe_allow_html=True,
    )


def _build_defect_context_from_files(
    files_by_m: dict[str, "object"],
) -> dict[str, float]:
    """
    Lit chaque fichier (Excel ou CSV), prend la derniere ligne numerique,
    fusionne les colonnes des 3 M en un seul defect_context.
    Tolerant : si un fichier manque, on retourne ce qu'on a.
    """
    from data_loader import load_file
    context: dict[str, float] = {}
    for m, fobj in files_by_m.items():
        if fobj is None:
            continue
        try:
            # load_file(file, filename) -> (df, report)
            filename = getattr(fobj, "name", str(fobj))
            df_m, _report = load_file(fobj, filename)
            if df_m is None or df_m.empty:
                continue
            num_df = df_m.select_dtypes(include="number")
            if num_df.empty:
                continue
            last = num_df.iloc[-1]
            for col in num_df.columns:
                val = last[col]
                if pd.notna(val):
                    context[col] = float(val)
        except Exception as e:
            st.warning(f"Lecture {m} : {e}")
    return context


def _build_defect_context_from_global_df(df: pd.DataFrame) -> dict[str, float]:
    """Mode 'un seul fichier global' (cf. home.py) : derniere ligne numerique."""
    if df is None or df.empty:
        return {}
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        return {}
    last = num_df.iloc[-1]
    return {c: float(last[c]) for c in num_df.columns if pd.notna(last[c])}


def _render_m_card(
    m: str,
    pred: MPrediction | None,
    contributions: list[ShapContribution],
) -> None:
    """
    Carte d'un M (Machine, Matiere, Methode) avec top-3 + bouton SHAP.
    Visuel uniforme et compact.
    """
    # Header carte
    if pred is None:
        status_class, status_text = "off", "en attente"
    elif not pred.available:
        status_class, status_text = "off", "indisponible"
    elif pred.is_none and not pred.top_causes:
        status_class, status_text = "warn", "pas de cause"
    else:
        status_class, status_text = "ok", "IA prete"

    st.markdown(
        f"""
        <div class="m-card">
          <div class="m-card-head">
            <div class="m-card-title">{m}</div>
            <div class="m-card-status {status_class}">{status_text}</div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    # Corps : top-3 ou message
    if pred is None:
        st.markdown(
            f'<div class="m-card-empty">{_EMPTY_SVG}'
            'Charge un fichier de donnees pour declencher l\'analyse IA.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return
    if not pred.available:
        st.markdown(
            f'<div class="m-card-empty">{_EMPTY_SVG}'
            f'{pred.reason_unavailable}<br>'
            f'A saisir manuellement sur cette branche.</div></div>',
            unsafe_allow_html=True,
        )
        return
    if not pred.top_causes:
        st.markdown(
            f'<div class="m-card-empty">{_EMPTY_SVG}'
            'Le modele ne detecte pas de cause sur cette branche.<br>'
            'A valider manuellement.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Top-3
    rows_html = []
    for rank, cp in enumerate(pred.top_causes, start=1):
        pct = int(round(cp.confidence * 100))
        num_class = "top" if rank == 1 else ""
        rows_html.append(
            f"""
            <div class="cause-rank">
              <div class="cause-rank-num {num_class}">{rank}</div>
              <div class="cause-rank-text">{cp.cause}</div>
              <div class="cause-rank-bar">
                <div class="cause-rank-bar-fill" style="width:{pct}%"></div>
              </div>
              <div class="cause-rank-pct">{pct}%</div>
            </div>
            """
        )
    st.markdown("".join(rows_html) + "</div>", unsafe_allow_html=True)

    # Bouton SHAP
    if contributions:
        with st.expander(f"Pourquoi ? - top features SHAP", expanded=False):
            st.markdown(
                shap_explanation_html(contributions),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page F3 complete
# ---------------------------------------------------------------------------
def render_ishikawa_page(
    df: pd.DataFrame | None = None,
    db_path: Path = Path("li90.db"),
) -> None:
    """
    Page F3 redesignee : workflow lineaire 4 etapes.
      1. Charger les donnees au moment du defaut (3 fichiers M ou df global)
      2. Decrire le defaut observe
      3. Voir les causes proposees par l'IA (3 cartes Machine / Matiere / Methode)
      4. Valider la cause racine et enregistrer
    """
    init_db(db_path)
    st.markdown(_LOCAL_CSS, unsafe_allow_html=True)

    # ---------- Header simple ---------------------------------------------
    st.markdown(
        f"""
        <div style="margin: 4px 0 22px 0;">
            <div style="font-size:11px; font-weight:600; color:{COLOR_PRIMARY};
                        letter-spacing:0.08em; text-transform:uppercase;
                        margin-bottom:6px;">
                Module F2 - Ishikawa 5M assiste par IA
            </div>
            <div style="font-size:24px; font-weight:600; color:{COLOR_TEXT};
                        letter-spacing:-0.02em; line-height:1.2;">
                Analyse cause racine
            </div>
            <div style="font-size:13px; color:{COLOR_TEXT_MUTED};
                        margin-top:8px; max-width:62ch; line-height:1.55;">
                Charge les donnees machine, matiere et methode au moment du
                defaut. L'IA propose les trois causes les plus probables par
                M, avec leur explication SHAP. Tu retiens la cause finale et
                tu valides.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Pre-load engine V2 (lazy) ----------------------------------
    if "f3_v2_engine" not in st.session_state:
        st.session_state["f3_v2_engine"] = try_load_supervised_engine()
    v2_engine: SupervisedRootCauseEngine | None = st.session_state["f3_v2_engine"]
    all_cases = load_all_rca(db_path)

    # ---------- STEP 1 : Donnees au moment du defaut -----------------------
    st.markdown('<div class="f3-step">', unsafe_allow_html=True)
    _step(
        1, "Donnees au moment du defaut",
        "Un fichier Excel/CSV par M (Machine, Matiere, Methode). "
        "Le logiciel prend la derniere ligne de chaque fichier.",
    )

    parameters: dict[str, float] = {}

    # Si on arrive depuis la page Import (df global dispo), on prefill
    if df is not None and not df.empty:
        parameters = _build_defect_context_from_global_df(df)
        st.success(
            f"Contexte recupere depuis le dataset SPC ({len(parameters)} "
            f"parametres, derniere ligne)."
        )
        with st.expander("Voir les valeurs reprises"):
            st.json(parameters)
    else:
        # Mode 3 fichiers : un uploader par M cote-a-cote
        c_machine, c_matiere, c_methode = st.columns(3, gap="medium")
        files: dict[str, "object"] = {}
        with c_machine:
            files["Machine"] = st.file_uploader(
                "Fichier Machine (capteurs)",
                type=["xlsx", "xls", "csv"],
                key="f3_file_machine",
                help="Colonnes attendues : M_machine_temperature, "
                     "M_machine_pression, ...",
            )
        with c_matiere:
            files["Matiere"] = st.file_uploader(
                "Fichier Matiere (lots, fournisseurs)",
                type=["xlsx", "xls", "csv"],
                key="f3_file_matiere",
            )
        with c_methode:
            files["Methode"] = st.file_uploader(
                "Fichier Methode (recettes, consignes)",
                type=["xlsx", "xls", "csv"],
                key="f3_file_methode",
            )
        if any(f is not None for f in files.values()):
            parameters = _build_defect_context_from_files(files)
            st.success(
                f"Contexte construit : {len(parameters)} parametres extraits."
            )
        else:
            st.info(
                "Charge au moins un fichier pour declencher l'analyse IA."
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- STEP 2 : Decrire le defeut ---------------------------------
    st.markdown('<div class="f3-step">', unsafe_allow_html=True)
    _step(
        2, "Defaut observe",
        "Type, description courte et facteurs humains/environnement "
        "(qui ne sont pas dans les donnees machine).",
    )
    c_def, c_ctx = st.columns([1, 1.4], gap="medium")
    with c_def:
        defect_type = st.text_input(
            "Type de defaut",
            placeholder="Ex. Soudure non conforme",
            key="f3_defect_type",
        )
    with c_ctx:
        context = st.text_input(
            "Description courte (optionnelle)",
            placeholder="Ex. Defaut run matin, lot kraft KR-2024-07.",
            key="f3_context",
        )
    # Saisie M4/M5 manuelle, repliable
    with st.expander(
        "Facteurs Main-d'oeuvre et Milieu (saisie manuelle)",
        expanded=False,
    ):
        manual_inputs: dict[str, list[str]] = {}
        for branch in sorted(MANUAL_BRANCHES):
            val = st.text_input(
                f"{branch} - causes observees (separe par virgule)",
                placeholder="Ex. fatigue fin shift, rotation recente",
                key=f"f3_manual_{branch}",
            )
            manual_inputs[branch] = [
                c.strip() for c in val.split(",") if c.strip()
            ]
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- STEP 3 : Causes proposees par l'IA -------------------------
    st.markdown('<div class="f3-step">', unsafe_allow_html=True)
    _step(
        3, "Causes proposees par l'IA",
        "Trois cartes (une par M). Top-3 cause + score de confiance. "
        "Bouton 'Pourquoi' pour les contributions SHAP.",
    )

    v2_predictions: dict[str, MPrediction] = {}
    v2_explanations: dict[str, list[ShapContribution]] = {}

    if v2_engine is not None and v2_engine.is_trained() and parameters:
        v2_predictions = v2_engine.predict_multi(parameters, top_k=3).predictions
        v2_explanations = v2_engine.explain_multi(parameters, top_features=5)

    cols_m = st.columns(3, gap="medium")
    for i, m in enumerate(M_BRANCHES):
        with cols_m[i]:
            _render_m_card(
                m,
                v2_predictions.get(m),
                v2_explanations.get(m, []),
            )

    # Empty state global si rien
    if v2_engine is None or not v2_engine.is_trained():
        st.warning(
            "Modele IA non charge. Ouvre le panneau 'Etat des modeles IA' "
            "en bas pour entrainer un modele synthetique de demonstration."
        )
    elif not parameters:
        st.info(
            "Charge au moins un fichier de donnees pour activer les "
            "predictions par M."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- STEP 4 : Validation ----------------------------------------
    st.markdown('<div class="f3-step">', unsafe_allow_html=True)
    _step(
        4, "Valider la cause racine",
        "Choisis la branche et la cause finale, signe, enregistre. "
        "Cette validation alimente la base SQLite pour le futur "
        "reentrainement.",
    )

    c1, c2, c3 = st.columns([1, 1.4, 0.8], gap="medium")
    with c1:
        chosen_branch = st.selectbox(
            "Branche retenue", options=BRANCHES_5M, key="f3_chosen_branch",
        )
    with c2:
        # Suggestions automatiques de causes (top1 par M dispo) en option du selectbox
        suggested = []
        for m, pred in v2_predictions.items():
            if pred.available and pred.top_causes:
                suggested.append(pred.top_causes[0].cause)
        if suggested:
            st.caption(
                f"Suggestions IA : {' / '.join(suggested[:3])}"
            )
        chosen_cause = st.text_input(
            "Cause racine retenue",
            placeholder="Ex. Derive temperature four (+8°C)",
            key="f3_chosen_cause",
        )
    with c3:
        operator = st.text_input(
            "Valide par", placeholder="Initiales", key="f3_operator",
        )

    if st.button(
        "Enregistrer l'analyse",
        type="primary",
        use_container_width=True,
    ):
        if not defect_type or not chosen_cause:
            st.error("Type de defaut et cause racine sont obligatoires.")
            return
        # Causes affichees a l'enregistrement
        causes_all = empty_5m_structure()
        for m, pred in v2_predictions.items():
            if pred.available and not pred.is_none:
                causes_all[m] = [cp.cause for cp in pred.top_causes[:3]]
        for branch, cs in manual_inputs.items():
            causes_all[branch] = cs

        # Enrichir le contexte avec la trace IA V2 (audit / reentrainement)
        enriched_context = context or ""
        traces = []
        for m, pred in v2_predictions.items():
            if pred.available and pred.top_causes:
                cp = pred.top_causes[0]
                traces.append(
                    f"[IA {m}] top1={cp.cause} ({cp.confidence:.0%})"
                )
        if traces:
            enriched_context = (
                enriched_context + "\n--- IA V2 ---\n"
                + "\n".join(traces)
            ).strip()

        case = RcaCase(
            defect_type=defect_type,
            context=enriched_context,
            parameters=parameters,
            causes_5m=causes_all,
            validated_root_cause=chosen_cause,
            validated_branch=chosen_branch,
            operator=operator,
            created_at=datetime.utcnow().isoformat(),
        )
        new_id = save_rca(case, db_path)
        st.success(
            f"Analyse enregistree (ID #{new_id}). "
            f"Total : {len(all_cases) + 1} cas en base."
        )
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Bottom : etat des modeles + KPI tres discrets --------------
    new_engine = render_v2_models_panel(v2_engine)
    if new_engine is not None and new_engine is not v2_engine:
        st.session_state["f3_v2_engine"] = new_engine

    n_cases = len(all_cases)
    n_validated = sum(1 for c in all_cases if c.validated_root_cause)
    if v2_engine is not None and v2_engine.is_trained():
        n_models = sum(1 for m in M_BRANCHES if v2_engine.is_available()[m])
        ia_label = f"LightGBM V2 ({n_models}/{len(M_BRANCHES)} M)"
    else:
        ia_label = "TF-IDF V1"
    st.caption(
        f"Base : {n_cases} cas, {n_validated} valides. Moteur : {ia_label}."
    )


# ---------------------------------------------------------------------------
# Historique des analyses
# ---------------------------------------------------------------------------
def render_history_panel(db_path: Path = Path("li90.db")) -> None:
    """Affiche la liste des analyses deja enregistrees."""
    cases = load_all_rca(db_path)
    st.markdown(
        f'<div class="section-title">Historique des analyses</div>',
        unsafe_allow_html=True,
    )
    if not cases:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <div><b>Aucune analyse enregistree</b></div>
                <div style="font-size:12px;">Les premiers cas enrichiront la base IA.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for c in reversed(cases[-30:]):
        rows.append({
            "Date": c.created_at[:10],
            "Defaut": c.defect_type,
            "Branche": c.validated_branch,
            "Cause racine": c.validated_root_cause,
            "Operateur": c.operator,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
