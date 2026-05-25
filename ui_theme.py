"""
Theme UI/UX LI90 - inspire AlignPro (dark professionnel + accent orange).
Reduit la charge mentale visuelle, met le focus sur la donnee et l'action.
"""
from __future__ import annotations

import streamlit as st


# ===========================================================================
# Palette AlignPro / LI90
# ===========================================================================
COLOR_BG = "#0a0a0f"           # fond principal (presque noir)
COLOR_CARD = "#111118"         # surfaces cards
COLOR_SURFACE_2 = "#1a1a24"    # secondary
COLOR_MUTED = "#18181f"        # muted bg
COLOR_ACCENT_BG = "#1f1f2a"    # accent bg
COLOR_SIDEBAR = "#0d0d12"      # sidebar

COLOR_PRIMARY = "#f97316"      # orange signature (CTA, accents)
COLOR_PRIMARY_LIGHT = "#fb923c"
COLOR_PRIMARY_DARK = "#c2410c"

COLOR_TEXT = "#f4f4f5"         # texte principal
COLOR_TEXT_MUTED = "#a1a1aa"   # texte secondaire
COLOR_TEXT_DIM = "#71717a"     # texte tertiaire

COLOR_BORDER = "#27272f"       # bordures
COLOR_BORDER_HOVER = "#3f3f46"

COLOR_OK = "#22c55e"           # success
COLOR_WARN = "#f97316"         # warning (= primary, semantique aligne)
COLOR_INFO = "#3b82f6"         # info
COLOR_DANGER = "#ef4444"       # critical

# Alias pour retrocompat avec les anciens modules
COLOR_ACCENT = COLOR_PRIMARY


# ===========================================================================
# CSS global - injecte une seule fois via inject_theme()
# ===========================================================================
CSS = f"""
<style>
/* ----- Cache le menu multipage natif Streamlit (on gere notre propre nav) - */
[data-testid="stSidebarNav"] {{ display: none !important; }}
[data-testid="stSidebarNavItems"] {{ display: none !important; }}

/* ----- Cache UNIQUEMENT les liens GitHub et la marque Streamlit ----------
 * On garde le header et la sidebar visibles. On masque juste les elements
 * qui revelent le code source.
 */
#MainMenu {{ visibility: hidden !important; }}
footer {{ visibility: hidden !important; height: 0 !important; }}
.viewerBadge_container__r5tak {{ display: none !important; }}
.viewerBadge_link__1S137 {{ display: none !important; }}
a[href*="github.com"] {{ display: none !important; }}
[data-testid="manage-app-button"] {{ display: none !important; }}

/* ----- Reset Streamlit dark + tokens ------------------------------- */
.stApp {{
    background: {COLOR_BG} !important;
    color: {COLOR_TEXT};
}}
.block-container {{
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1500px;
}}
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 system-ui, sans-serif !important;
    color: {COLOR_TEXT};
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "rlig" 1, "calt" 1, "ss01" 1;
}}
h1, h2, h3, h4 {{
    color: {COLOR_TEXT} !important;
    letter-spacing: -0.02em;
}}

/* ----- Scrollbar ----------------------------------------------------- */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: {COLOR_BORDER};
    border-radius: 10px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {COLOR_BORDER_HOVER}; }}

/* ----- Sidebar ------------------------------------------------------- */
section[data-testid="stSidebar"] {{
    background: {COLOR_SIDEBAR} !important;
    border-right: 1px solid {COLOR_BORDER};
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1rem; }}
section[data-testid="stSidebar"] * {{ color: {COLOR_TEXT}; }}

/* ----- Boutons primary (orange glow) -------------------------------- */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {{
    background: {COLOR_PRIMARY} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 0 40px rgba(249, 115, 22, 0.18);
    transition: all 180ms cubic-bezier(0.2, 0.8, 0.2, 1);
}}
.stButton > button[kind="primary"]:hover {{
    background: {COLOR_PRIMARY_LIGHT} !important;
    transform: translateY(-1px);
    box-shadow: 0 0 60px rgba(249, 115, 22, 0.30);
}}
.stButton > button[kind="secondary"] {{
    background: {COLOR_SURFACE_2} !important;
    color: {COLOR_TEXT} !important;
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {COLOR_PRIMARY} !important;
    color: {COLOR_PRIMARY} !important;
}}

/* ----- Inputs ------------------------------------------------------- */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {COLOR_SURFACE_2} !important;
    color: {COLOR_TEXT} !important;
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 8px !important;
}}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {{
    border-color: {COLOR_PRIMARY} !important;
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.15);
}}

/* ----- File uploader ----------------------------------------------- */
[data-testid="stFileUploader"] {{
    background: {COLOR_CARD};
    border: 1px dashed {COLOR_BORDER};
    border-radius: 12px;
    padding: 1rem;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {COLOR_PRIMARY};
}}

/* ----- Dataframe ---------------------------------------------------- */
[data-testid="stDataFrame"] {{
    background: {COLOR_CARD};
    border-radius: 12px;
}}

/* ----- Expander ---------------------------------------------------- */
[data-testid="stExpander"] {{
    background: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
}}
[data-testid="stExpander"] summary {{
    background: {COLOR_CARD} !important;
    color: {COLOR_TEXT} !important;
    border-radius: 8px !important;
}}

/* ----- Tabs --------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {COLOR_BORDER};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {COLOR_TEXT_MUTED};
    font-weight: 500;
    border-radius: 0;
    padding: 10px 16px;
}}
.stTabs [aria-selected="true"] {{
    color: {COLOR_PRIMARY} !important;
    border-bottom: 2px solid {COLOR_PRIMARY} !important;
}}

/* ----- Header de page LI90 ----------------------------------------- */
.li90-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0 1.2rem 0;
    border-bottom: 1px solid {COLOR_BORDER};
    margin-bottom: 1.4rem;
}}
.li90-header-left {{ display: flex; align-items: center; gap: 12px; }}
.li90-logo {{
    width: 36px; height: 36px;
    background: linear-gradient(135deg, {COLOR_PRIMARY}, {COLOR_PRIMARY_DARK});
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 0.5px;
    box-shadow: 0 0 40px rgba(249, 115, 22, 0.30);
}}
.li90-title {{
    font-size: 17px;
    font-weight: 700;
    color: {COLOR_TEXT};
    letter-spacing: -0.01em;
}}
.li90-sub {{
    font-size: 11.5px;
    color: {COLOR_TEXT_MUTED};
    margin-top: 1px;
}}
.li90-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(249, 115, 22, 0.10);
    color: {COLOR_PRIMARY};
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(249, 115, 22, 0.30);
}}
.li90-badge::before {{
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {COLOR_PRIMARY};
    box-shadow: 0 0 8px {COLOR_PRIMARY};
}}

/* ----- KPI cards (grid) -------------------------------------------- */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 1.4rem;
}}
.kpi-card {{
    background: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    transition: all 200ms cubic-bezier(0.2, 0.8, 0.2, 1);
}}
.kpi-card:hover {{
    border-color: {COLOR_BORDER_HOVER};
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,0,0,.40);
}}
.kpi-label {{
    font-size: 11.5px;
    color: {COLOR_TEXT_MUTED};
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 600;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: 700;
    color: {COLOR_TEXT};
    margin-top: 4px;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}}
.kpi-delta {{
    font-size: 11.5px;
    margin-top: 6px;
    font-weight: 500;
}}
.kpi-delta.up      {{ color: {COLOR_OK}; }}
.kpi-delta.down    {{ color: {COLOR_DANGER}; }}
.kpi-delta.neutral {{ color: {COLOR_TEXT_MUTED}; }}

/* ----- Section titles ---------------------------------------------- */
.section-title {{
    font-size: 14px;
    font-weight: 700;
    color: {COLOR_TEXT};
    margin: 0.8rem 0 0.6rem 0;
    letter-spacing: -0.005em;
}}
.section-subtitle {{
    font-size: 12px;
    color: {COLOR_TEXT_MUTED};
    margin-bottom: 0.8rem;
}}

/* ----- Pills (statut, tags) ---------------------------------------- */
.pill {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 10.5px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}}
.pill::before {{
    content: ""; width: 5px; height: 5px;
    border-radius: 50%;
    display: inline-block;
}}
.pill-ok    {{ background: rgba(34,197,94,.12);  color: {COLOR_OK}; }}
.pill-ok::before    {{ background: {COLOR_OK}; box-shadow: 0 0 6px {COLOR_OK}; }}
.pill-warn  {{ background: rgba(249,115,22,.12); color: {COLOR_PRIMARY}; }}
.pill-warn::before  {{ background: {COLOR_PRIMARY}; box-shadow: 0 0 6px {COLOR_PRIMARY}; }}
.pill-info  {{ background: rgba(59,130,246,.12); color: {COLOR_INFO}; }}
.pill-info::before  {{ background: {COLOR_INFO}; }}
.pill-danger {{ background: rgba(239,68,68,.12); color: {COLOR_DANGER}; }}
.pill-danger::before {{ background: {COLOR_DANGER}; box-shadow: 0 0 6px {COLOR_DANGER}; }}
.pill-neutral {{ background: {COLOR_SURFACE_2}; color: {COLOR_TEXT_MUTED}; }}
.pill-neutral::before {{ background: {COLOR_TEXT_DIM}; }}

/* ----- AI suggestion card ----------------------------------------- */
.ai-card {{
    background: linear-gradient(135deg,
        rgba(249,115,22,0.06), {COLOR_CARD});
    border: 1px solid rgba(249,115,22,0.25);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all 200ms cubic-bezier(0.2, 0.8, 0.2, 1);
}}
.ai-card:hover {{
    border-color: rgba(249,115,22,0.50);
    box-shadow: 0 0 40px rgba(249, 115, 22, 0.18);
}}
.ai-card-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
}}
.ai-branch {{
    font-size: 11px;
    font-weight: 700;
    color: {COLOR_PRIMARY};
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.ai-cause {{
    font-size: 14px;
    font-weight: 600;
    color: {COLOR_TEXT};
    line-height: 1.35;
    margin-bottom: 6px;
}}
.ai-confidence-bar {{
    height: 4px;
    background: {COLOR_SURFACE_2};
    border-radius: 999px;
    overflow: hidden;
    margin: 8px 0 6px 0;
}}
.ai-confidence-fill {{
    height: 100%;
    background: linear-gradient(90deg, {COLOR_PRIMARY}, {COLOR_PRIMARY_LIGHT});
    border-radius: 999px;
    transition: width 400ms cubic-bezier(0.2, 0.8, 0.2, 1);
}}
.ai-explanation {{
    font-size: 11.5px;
    color: {COLOR_TEXT_MUTED};
    line-height: 1.45;
}}

/* ----- Empty state ------------------------------------------------- */
.empty-state {{
    background: {COLOR_CARD};
    border: 1px dashed {COLOR_BORDER};
    border-radius: 12px;
    padding: 28px 16px;
    text-align: center;
    color: {COLOR_TEXT_MUTED};
}}
.empty-state-icon {{
    font-size: 28px;
    margin-bottom: 10px;
    opacity: 0.6;
}}

/* ----- Animations utilitaires -------------------------------------- */
@keyframes pulse-soft {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.6; }}
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-6px); }}
}}
.animate-pulse-soft {{ animation: pulse-soft 2s ease-in-out infinite; }}
.animate-float {{ animation: float 6s ease-in-out infinite; }}
</style>
"""


# ===========================================================================
# Helpers publics
# ===========================================================================
def inject_theme() -> None:
    """A appeler une seule fois en haut de chaque page (idempotent)."""
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(
    title: str, subtitle: str = "", badge: str = "v0.4 desktop",
) -> None:
    """Header standard d'une page interne LI90.

    Le logo LI90 est deja affiche dans la sidebar : on ne le ré-affiche pas
    ici pour eviter le chevauchement avec le titre.
    """
    badge_html = (f'<div class="li90-badge">{badge}</div>' if badge else "")
    st.markdown(
        f"""
        <div class="li90-header">
            <div class="li90-header-left">
                <div>
                    <div class="li90-title">{title}</div>
                    <div class="li90-sub">{subtitle}</div>
                </div>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str, value: str, delta: str = "", trend: str = "neutral",
) -> str:
    delta_html = (
        f'<div class="kpi-delta {trend}">{delta}</div>' if delta else ""
    )
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    """


def kpi_grid(cards: list[str]) -> None:
    html = '<div class="kpi-grid">' + "".join(cards) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = (f'<div class="section-subtitle">{subtitle}</div>' if subtitle else "")
    st.markdown(
        f'<div class="section-title">{title}</div>{sub}',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "info") -> str:
    """Pill HTML : kind in {ok, warn, info, danger, neutral}."""
    return f'<span class="pill pill-{kind}">{text}</span>'


def ai_suggestion_card(
    cause: str, branch: str, confidence: float, explanation: str = "",
) -> str:
    pct = int(round(confidence * 100))
    return f"""
        <div class="ai-card">
            <div class="ai-card-header">
                <div class="ai-branch">{branch}</div>
                <div style="font-size:12px; font-weight:700;
                           color:{COLOR_PRIMARY}">
                    {pct}% confiance
                </div>
            </div>
            <div class="ai-cause">{cause}</div>
            <div class="ai-confidence-bar">
                <div class="ai-confidence-fill" style="width:{pct}%"></div>
            </div>
            <div class="ai-explanation">{explanation}</div>
        </div>
    """
