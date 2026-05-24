"""
Dashboard LI90 moderne (Plotly + theme custom).
Vue synthese : KPI + cartes de controle + Pareto + alertes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui_theme import (
    COLOR_PRIMARY, COLOR_PRIMARY_LIGHT, COLOR_ACCENT,
    COLOR_OK, COLOR_WARN, COLOR_DANGER,
    COLOR_BORDER, COLOR_TEXT, COLOR_TEXT_MUTED,
    kpi_card, kpi_grid, section, pill,
)


# ---------------------------------------------------------------------------
# Theme Plotly unifie
# ---------------------------------------------------------------------------
def plotly_layout(title: str = "", height: int = 360) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=COLOR_TEXT, family="Inter, sans-serif"),
            x=0.01, xanchor="left",
        ),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color=COLOR_TEXT),
        xaxis=dict(
            gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER,
            showline=True, linewidth=1, linecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER,
            showline=True, linewidth=1, linecolor=COLOR_BORDER,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(255,255,255,0)",
        ),
    )


# ---------------------------------------------------------------------------
# KPI header
# ---------------------------------------------------------------------------
def render_kpi_header(
    n_points: int,
    n_params: int,
    n_alerts: int,
    ppm: float | None = None,
    cpk: float | None = None,
) -> None:
    cards: list[str] = []

    cards.append(kpi_card(
        "Mesures analysees", f"{n_points:,}".replace(",", " "),
        delta="Audit trail active", trend="neutral",
    ))
    cards.append(kpi_card(
        "Parametres surveilles", str(n_params),
        delta="Tous canaux actifs", trend="neutral",
    ))
    alert_trend = "down" if n_alerts > 0 else "up"
    alert_label = f"{n_alerts} a traiter" if n_alerts else "Processus stable"
    cards.append(kpi_card(
        "Alertes detectees", str(n_alerts),
        delta=alert_label, trend=alert_trend,
    ))
    if ppm is not None:
        cards.append(kpi_card(
            "Taux de defauts (PPM)", f"{ppm:,.0f}".replace(",", " "),
            delta="Cible 200 PPM", trend="neutral",
        ))
    if cpk is not None:
        trend = "up" if cpk >= 1.33 else ("down" if cpk < 1 else "neutral")
        delta = "Capable (≥1.33)" if cpk >= 1.33 else "Insuffisant (<1)" \
            if cpk < 1 else "Acceptable"
        cards.append(kpi_card(
            "Cpk moyen", f"{cpk:.2f}", delta=delta, trend=trend,
        ))

    kpi_grid(cards)


# ---------------------------------------------------------------------------
# Carte de controle (Shewhart) stylisee
# ---------------------------------------------------------------------------
def control_chart(series: pd.Series, param_name: str) -> go.Figure:
    mean = series.mean()
    std = series.std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    x = list(range(1, len(series) + 1))

    fig = go.Figure()

    # Bande +/- 3 sigma
    fig.add_hrect(
        y0=lcl, y1=ucl, fillcolor=COLOR_PRIMARY, opacity=0.04,
        line_width=0, layer="below",
    )

    # Ligne centrale
    fig.add_hline(
        y=mean, line_color=COLOR_PRIMARY, line_width=1.2, line_dash="solid",
        annotation_text=f"μ = {mean:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color=COLOR_PRIMARY),
    )
    fig.add_hline(
        y=ucl, line_color=COLOR_DANGER, line_width=1, line_dash="dash",
        annotation_text=f"UCL = {ucl:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color=COLOR_DANGER),
    )
    fig.add_hline(
        y=lcl, line_color=COLOR_DANGER, line_width=1, line_dash="dash",
        annotation_text=f"LCL = {lcl:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color=COLOR_DANGER),
    )

    # Points
    in_ctrl = (series >= lcl) & (series <= ucl)
    colors = [COLOR_PRIMARY if ok else COLOR_DANGER for ok in in_ctrl]

    fig.add_trace(go.Scatter(
        x=x, y=series.values,
        mode="lines+markers",
        line=dict(color=COLOR_PRIMARY_LIGHT, width=1.5),
        marker=dict(size=6, color=colors, line=dict(width=0)),
        name=param_name,
        hovertemplate="Point %{x}<br>Valeur %{y:.3f}<extra></extra>",
    ))

    layout = plotly_layout(title=f"Carte de controle — {param_name}", height=340)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Pareto dynamique des parametres critiques
# ---------------------------------------------------------------------------
def pareto_chart(scores: dict[str, float], top_n: int = 10) -> go.Figure:
    items = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    if not items:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnee disponible",
            showarrow=False,
            font=dict(size=13, color=COLOR_TEXT_MUTED),
        )
        fig.update_layout(**plotly_layout("Pareto des parametres critiques"))
        return fig

    names, values = zip(*items)
    cum = np.cumsum(values) / sum(values) * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=names, y=values,
            marker_color=[
                COLOR_DANGER if i < 3 else COLOR_WARN if i < 6 else COLOR_PRIMARY_LIGHT
                for i in range(len(names))
            ],
            marker_line_width=0,
            name="Score criticite",
            hovertemplate="%{x}<br>Score %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=names, y=cum,
            mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=2.5),
            marker=dict(size=6, color=COLOR_ACCENT),
            name="Cumul %",
            hovertemplate="%{x}<br>Cumul %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=80, line_color=COLOR_ACCENT, line_width=1, line_dash="dot",
        secondary_y=True,
        annotation_text="Seuil 80%",
        annotation_position="right",
        annotation_font=dict(size=10, color=COLOR_ACCENT),
    )

    layout = plotly_layout("Pareto des parametres critiques", height=380)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Score", secondary_y=False, showgrid=True)
    fig.update_yaxes(
        title_text="Cumul %", secondary_y=True, range=[0, 105], showgrid=False,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


# ---------------------------------------------------------------------------
# Heatmap de correlation parametres <-> defauts
# ---------------------------------------------------------------------------
def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0, COLOR_DANGER], [0.5, "white"], [1, COLOR_PRIMARY],
        ],
        zmid=0,
        zmin=-1, zmax=1,
        hoverongaps=False,
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.7, x=1.02),
    ))
    layout = plotly_layout("Matrice de correlation", height=420)
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-30)
    return fig


# ---------------------------------------------------------------------------
# Timeline des alertes
# ---------------------------------------------------------------------------
def alerts_timeline(alerts: pd.DataFrame) -> go.Figure:
    """
    alerts: DataFrame avec colonnes timestamp, parameter, severity (ok/warn/danger)
    """
    if alerts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune alerte sur la periode", showarrow=False,
            font=dict(size=13, color=COLOR_TEXT_MUTED),
        )
        fig.update_layout(**plotly_layout("Timeline des alertes", height=300))
        return fig

    color_map = {"ok": COLOR_OK, "warn": COLOR_WARN, "danger": COLOR_DANGER}
    fig = go.Figure()
    for sev, group in alerts.groupby("severity"):
        fig.add_trace(go.Scatter(
            x=group["timestamp"], y=group["parameter"],
            mode="markers",
            marker=dict(
                size=12, color=color_map.get(sev, COLOR_PRIMARY),
                line=dict(width=1, color="white"),
            ),
            name=sev,
            hovertemplate="%{y} — %{x}<extra></extra>",
        ))
    fig.update_layout(**plotly_layout("Timeline des alertes", height=340))
    return fig


# ---------------------------------------------------------------------------
# Bloc "alertes actives"
# ---------------------------------------------------------------------------
def render_active_alerts(alerts: list[dict]) -> None:
    """
    alerts : liste de dicts {parameter, message, severity, timestamp}
    """
    if not alerts:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">✓</div>
                <div><b>Aucune alerte active</b></div>
                <div style="font-size:12px">Tous les parametres sont dans les limites.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for a in alerts:
        sev = a.get("severity", "warn")
        st.markdown(
            f"""
            <div class="section-card" style="padding:0.8rem 1rem;
                margin-bottom:0.5rem; border-left:3px solid
                {COLOR_DANGER if sev == 'danger' else COLOR_WARN};">
                <div style="display:flex; justify-content:space-between;
                            align-items:center;">
                    <div>
                        <div style="font-weight:700; color:{COLOR_TEXT};
                                    font-size:13px;">{a['parameter']}</div>
                        <div style="color:{COLOR_TEXT_MUTED}; font-size:12px;
                                    margin-top:2px;">{a['message']}</div>
                    </div>
                    <div>{pill(sev.upper(), 'danger' if sev == 'danger' else 'warn')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
