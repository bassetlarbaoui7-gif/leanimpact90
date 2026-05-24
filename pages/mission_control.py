"""
mission_control.py - L'ecran unique qui remplace Vue A + Vue B.

Inspiration : Linear, Stripe Dashboard, Palantir, Notion. Le user ne
choisit jamais "ou aller", le logiciel lui montre la prochaine action
prioritaire selon son role.

Structure :
  1. Header        - Bonjour {nom}, role, etat usine
  2. Hero action   - 1 grosse carte "fais ca maintenant"
  3. Inbox         - Ce qui m'attend (filtre par role)
  4. Pulse         - 3 chiffres usine
"""
from __future__ import annotations

import streamlit as st

from core import db
from ui_theme import (
    inject_theme,
    COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_CARD,
    COLOR_BORDER, COLOR_OK, COLOR_DANGER,
)


# ---------------------------------------------------------------------------
# Configuration des "hero actions" par role
# ---------------------------------------------------------------------------
HERO_BY_ROLE = {
    "operator": {
        "title": "Signaler un incident",
        "desc": ("Un bruit bizarre, un defaut, une panne ? "
                 "Decris en 30 secondes, l'equipe prend le relais."),
        "cta": "Signaler maintenant",
        "icon": "●",
        "target": "signaler",
    },
    "technician": {
        "title": "Fiabiliser les signalements",
        "desc": "Nettoie, enrichis le contexte, transmets au Resp. AC.",
        "cta": "Voir les signalements bruts",
        "icon": "■",
        "target": "signalements_bruts",
    },
    "maintenance": {
        "title": "Fiabiliser les signalements",
        "desc": "Nettoie, enrichis le contexte, transmets au Resp. AC.",
        "cta": "Voir les signalements bruts",
        "icon": "■",
        "target": "signalements_bruts",
    },
    "ac_manager": {
        "title": "Piloter les projets d'amelioration",
        "desc": ("Cadre, analyse avec l'IA, valide, deploie. "
                 "Tout le workflow en un fil continu."),
        "cta": "Voir mes projets en cours",
        "icon": "◆",
        "target": "projets_en_cours",
    },
    "production": {
        "title": "Valider les projets prets",
        "desc": ("Approuve les actions d'amelioration. "
                 "Tu vois le gain attendu en € et en productivite."),
        "cta": "Voir les projets a valider",
        "icon": "▲",
        "target": "projets_a_valider",
    },
    "ceo": {
        "title": "Vue strategique de l'usine",
        "desc": "OEE, productivite, ROI des actions d'amelioration.",
        "cta": "Ouvrir le tableau de bord",
        "icon": "★",
        "target": "dashboard",
    },
}


# ---------------------------------------------------------------------------
# Helpers de comptage (utilises pour l'inbox + pulse)
# ---------------------------------------------------------------------------
def _safe_count(fn) -> int:
    """Tente un comptage SQLite, retourne 0 si la base est vide ou erreur."""
    try:
        return len(fn())
    except Exception:
        return 0


def _counts() -> dict[str, int]:
    return {
        "inc_brut":         _safe_count(lambda: db.list_incidents("brut")),
        "inc_fiabilise":    _safe_count(lambda: db.list_incidents("fiabilise")),
        "proj_cadrage":     _safe_count(lambda: db.list_projets_ac("cadrage")),
        "proj_analyse":     _safe_count(lambda: db.list_projets_ac("analyse")),
        "proj_validation":  _safe_count(
            lambda: db.list_projets_ac("validation_en_cours")),
        "actions_a_faire":  _safe_count(
            lambda: db.list_actions(statut="a_faire")),
        "actions_en_cours": _safe_count(
            lambda: db.list_actions(statut="en_cours")),
    }


# ---------------------------------------------------------------------------
# Section 1 : Header
# ---------------------------------------------------------------------------
def _render_header(role: str, user_name: str) -> None:
    role_labels = {
        "operator":    "Operateur",
        "technician":  "Technicien N+1",
        "maintenance": "Resp. Maintenance",
        "ac_manager":  "Responsable AC",
        "production":  "Resp. Production",
        "ceo":         "Direction",
    }
    role_lbl = role_labels.get(role, "—")
    greeting = "Bonjour"
    name_part = f", {user_name}" if user_name else ""

    st.markdown(
        f"""
        <style>
          .mc-header {{
            display:flex; align-items:center; justify-content:space-between;
            padding: 8px 0 22px 0;
            border-bottom: 1px solid {COLOR_BORDER};
            margin-bottom: 28px;
          }}
          .mc-h-title {{
            font-size: 26px; font-weight: 800; color: {COLOR_TEXT};
            letter-spacing: -0.02em;
          }}
          .mc-h-sub {{
            font-size: 13px; color: {COLOR_TEXT_MUTED}; margin-top: 2px;
          }}
          .mc-pill {{
            display:inline-flex; align-items:center; gap:8px;
            padding: 6px 14px; border-radius: 999px;
            background: rgba(34, 197, 94, 0.10);
            border: 1px solid rgba(34, 197, 94, 0.30);
            color: {COLOR_OK}; font-size: 12.5px; font-weight: 500;
          }}
          .mc-pill .dot {{
            width: 7px; height: 7px; border-radius: 50%;
            background: {COLOR_OK};
          }}
        </style>
        <div class="mc-header">
          <div>
            <div class="mc-h-title">{greeting}{name_part}.</div>
            <div class="mc-h-sub">Vue : <b style="color:{COLOR_TEXT}">{role_lbl}</b> · Usine en marche</div>
          </div>
          <div class="mc-pill"><span class="dot"></span>Systeme actif</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 2 : Hero action
# ---------------------------------------------------------------------------
def _render_hero_action(role: str, counts: dict[str, int]) -> None:
    cfg = HERO_BY_ROLE.get(role, HERO_BY_ROLE["ac_manager"])

    # Compteur contextuel sous le titre (depuis SQLite)
    counter_text = ""
    if role == "operator":
        n = counts["actions_a_faire"]
        if n > 0:
            counter_text = f"{n} action(s) en attente sur ton poste"
    elif role in ("technician", "maintenance"):
        n = counts["inc_brut"]
        counter_text = f"{n} signalement(s) brut(s) a fiabiliser"
    elif role == "ac_manager":
        n_cad = counts["proj_cadrage"]
        n_ana = counts["proj_analyse"]
        n_val = counts["proj_validation"]
        counter_text = (
            f"{n_cad} a cadrer · {n_ana} en analyse IA · {n_val} en validation"
        )
    elif role == "production":
        n = counts["proj_validation"]
        counter_text = f"{n} projet(s) en attente de ta validation"
    elif role == "ceo":
        n = (counts["proj_cadrage"] + counts["proj_analyse"]
             + counts["proj_validation"])
        counter_text = f"{n} projet(s) en cours dans l'usine"

    st.markdown(
        f"""
        <style>
          .mc-hero {{
            position: relative;
            padding: 30px 32px;
            border-radius: 20px;
            background: linear-gradient(135deg,
                rgba(249,115,22,0.12) 0%,
                rgba(249,115,22,0.04) 60%,
                {COLOR_CARD} 100%);
            border: 1px solid rgba(249,115,22,0.30);
            margin-bottom: 28px;
            overflow: hidden;
          }}
          .mc-hero::before {{
            content: ""; position: absolute; top: -40%; right: -10%;
            width: 320px; height: 320px; border-radius: 50%;
            background: radial-gradient(circle,
                rgba(249,115,22,0.18) 0%,
                rgba(249,115,22,0.0) 70%);
          }}
          .mc-hero-ico {{
            display:inline-flex; align-items:center; justify-content:center;
            width: 52px; height: 52px; border-radius: 14px;
            background: rgba(249,115,22,0.15);
            color: {COLOR_PRIMARY}; font-size: 22px; font-weight: 700;
            margin-bottom: 16px;
          }}
          .mc-hero-label {{
            display: inline-block;
            color: {COLOR_PRIMARY}; font-size: 12px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em;
            margin-bottom: 10px;
          }}
          .mc-hero-title {{
            font-size: 28px; font-weight: 800; color: {COLOR_TEXT};
            letter-spacing: -0.02em; margin-bottom: 10px;
            position: relative;
          }}
          .mc-hero-desc {{
            color: {COLOR_TEXT_MUTED}; font-size: 14.5px;
            line-height: 1.55; max-width: 580px;
            position: relative;
          }}
          .mc-hero-counter {{
            margin-top: 14px;
            color: {COLOR_TEXT}; font-size: 13px;
            position: relative;
          }}
          .mc-hero-counter b {{ color: {COLOR_PRIMARY}; }}
        </style>
        <div class="mc-hero">
          <div class="mc-hero-ico">{cfg['icon']}</div>
          <div class="mc-hero-label">Pour toi maintenant</div>
          <div class="mc-hero-title">{cfg['title']}</div>
          <div class="mc-hero-desc">{cfg['desc']}</div>
          {f'<div class="mc-hero-counter"><b>•</b> {counter_text}</div>'
              if counter_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bouton d'action - sous la carte hero
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            cfg['cta'],
            type="primary",
            use_container_width=True,
            key="mc_hero_cta",
        ):
            # On set un flag pour que la page suivante sache ou aller.
            # Au step suivant on connectera ca a la vraie route.
            st.session_state["mc_target"] = cfg["target"]
            st.info(
                f"Action demandee : {cfg['target']}. "
                "La navigation contextuelle arrive a l'etape suivante."
            )


# ---------------------------------------------------------------------------
# Section 3 : Inbox (ce qui m'attend, filtre par role)
# ---------------------------------------------------------------------------
def _render_inbox(role: str) -> None:
    st.markdown(
        f"""
        <div style="font-size:13px; color:{COLOR_PRIMARY}; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.06em;
                    margin: 4px 0 12px 0;">
          Ton inbox
        </div>
        """,
        unsafe_allow_html=True,
    )

    # On charge la liste pertinente selon le role
    if role == "operator":
        df = db.list_incidents()
        # Filtre : on montre seulement les incidents que cet operateur a cree
        # (operateur_nom = user_name si rempli). A defaut : tous.
        user_name = st.session_state.get("user_name", "")
        if user_name and "operateur_nom" in df.columns:
            df = df[df["operateur_nom"].astype(str).str.lower()
                    == user_name.lower()]
        title = "Tes derniers signalements"
        cols  = ["id", "machine", "type_incident", "severite", "statut",
                 "cree_le"]
    elif role in ("technician", "maintenance"):
        df = db.list_incidents("brut")
        title = "Signalements bruts a fiabiliser"
        cols  = ["id", "machine", "type_incident", "severite",
                 "operateur_nom", "cree_le"]
    elif role == "ac_manager":
        df = db.list_projets_ac()
        df = df[df["statut"].isin([
            "cadrage", "analyse", "validation_en_cours", "valide",
            "solution_propose",
        ])] if not df.empty else df
        title = "Tes projets en cours"
        cols  = ["id", "titre", "statut", "cree_par", "cree_le"]
    elif role == "production":
        df = db.list_projets_ac("validation_en_cours")
        title = "Projets en attente de ta validation"
        cols  = ["id", "titre", "cree_par", "cree_le"]
    elif role == "ceo":
        df = db.list_projets_ac()
        title = "Tous les projets actifs"
        cols  = ["id", "titre", "statut", "cree_par", "cree_le"]
    else:
        df = db.list_incidents()
        title = "Activite recente"
        cols  = ["id", "machine", "statut", "cree_le"]

    with st.container(border=True):
        st.markdown(
            f"<div style='font-weight:600; color:{COLOR_TEXT}; "
            f"font-size:14.5px; margin-bottom:8px;'>{title}</div>",
            unsafe_allow_html=True,
        )
        if df.empty:
            st.markdown(
                f"<div style='color:{COLOR_TEXT_MUTED}; font-size:13px;'>"
                "Rien pour l'instant — l'inbox sera remplie au fur et a "
                "mesure de l'activite.</div>",
                unsafe_allow_html=True,
            )
        else:
            cols_show = [c for c in cols if c in df.columns]
            st.dataframe(
                df[cols_show].head(8),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Section 4 : Pulse (3 chiffres usine)
# ---------------------------------------------------------------------------
def _render_pulse(counts: dict[str, int]) -> None:
    n_signaux  = counts["inc_brut"] + counts["inc_fiabilise"]
    n_projets  = (counts["proj_cadrage"] + counts["proj_analyse"]
                  + counts["proj_validation"])
    n_actions  = counts["actions_a_faire"] + counts["actions_en_cours"]

    st.markdown(
        f"""
        <div style="font-size:13px; color:{COLOR_PRIMARY}; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.06em;
                    margin: 28px 0 12px 0;">
          Pouls de l'usine
        </div>
        <div style="display:grid; grid-template-columns:repeat(3, 1fr);
                    gap:14px;">
          <div style="padding:18px 22px; background:{COLOR_CARD};
                      border:1px solid {COLOR_BORDER}; border-radius:14px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Signaux du terrain</div>
            <div style="font-size:30px; font-weight:800; color:{COLOR_TEXT};
                        margin-top:4px;">{n_signaux}</div>
            <div style="font-size:11.5px; color:{COLOR_TEXT_MUTED};
                        margin-top:4px;">
              {counts['inc_brut']} brut(s) · {counts['inc_fiabilise']} fiabilise(s)
            </div>
          </div>
          <div style="padding:18px 22px; background:{COLOR_CARD};
                      border:1px solid {COLOR_BORDER}; border-radius:14px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Projets en cours</div>
            <div style="font-size:30px; font-weight:800;
                        color:{COLOR_PRIMARY}; margin-top:4px;">{n_projets}</div>
            <div style="font-size:11.5px; color:{COLOR_TEXT_MUTED};
                        margin-top:4px;">
              {counts['proj_cadrage']} cadrage · {counts['proj_analyse']} analyse · {counts['proj_validation']} validation
            </div>
          </div>
          <div style="padding:18px 22px; background:{COLOR_CARD};
                      border:1px solid {COLOR_BORDER}; border-radius:14px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Actions actives</div>
            <div style="font-size:30px; font-weight:800; color:{COLOR_OK};
                        margin-top:4px;">{n_actions}</div>
            <div style="font-size:11.5px; color:{COLOR_TEXT_MUTED};
                        margin-top:4px;">
              {counts['actions_a_faire']} a faire · {counts['actions_en_cours']} en cours
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page principale (rendue par Streamlit)
# ---------------------------------------------------------------------------
ROLE_LABELS = {
    "operator":    "Operateur",
    "technician":  "Technicien N+1",
    "maintenance": "Resp. Maintenance",
    "ac_manager":  "Responsable AC",
    "production":  "Resp. Production",
    "ceo":         "Direction",
}


def _render_sidebar() -> str:
    """Sidebar minimaliste : logo + role + 4 nav items + bouton retour."""
    with st.sidebar:
        # Logo LI90
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
                                box-shadow:0 0 18px rgba(249,115,22,0.35);">
                      L90
                    </div>
                    <div>
                        <div style="font-weight:800; font-size:15px;
                                    color:{COLOR_TEXT};">LI90</div>
                        <div style="font-size:11px; color:{COLOR_TEXT_MUTED};">
                            Mission Control &middot; beta
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
            "Role", options=role_lbls, index=cur_idx,
            label_visibility="collapsed",
        )
        st.session_state["role"] = role_keys[role_lbls.index(sel_lbl)]

        st.markdown("---")

        # Navigation minimaliste (4 entrees)
        nav = st.radio(
            "Navigation",
            options=[
                "Aujourd'hui",
                "Signalements",
                "Projets",
                "Actions",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if st.button("← Ancienne interface",
                     use_container_width=True,
                     help="Revenir aux 2 vues classiques"):
            st.switch_page("landing.py")

        st.caption("LI90 · v0.5-beta")

    return nav


def main() -> None:
    st.set_page_config(
        page_title="LI90 — Mission Control",
        page_icon="●",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    # Init session
    if "role" not in st.session_state:
        st.session_state["role"] = "ac_manager"

    # Init base (idempotent)
    db.init_db()

    # Sidebar (retourne la nav choisie)
    nav_choice = _render_sidebar()

    role      = st.session_state.get("role", "ac_manager")
    user_name = st.session_state.get("user_name", "")
    counts    = _counts()

    # Pour cette beta : seul "Aujourd'hui" est branche.
    # Les 3 autres montrent un placeholder (branchement complet au jour 2).
    if nav_choice == "Aujourd'hui":
        _render_header(role, user_name)
        _render_hero_action(role, counts)
        _render_inbox(role)
        _render_pulse(counts)
    else:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 80px 20px;
                        color:{COLOR_TEXT_MUTED};">
              <div style="font-size:54px; color:{COLOR_PRIMARY};
                          font-weight:800;">●</div>
              <div style="font-size:22px; color:{COLOR_TEXT};
                          font-weight:700; margin-top:8px;">
                {nav_choice}
              </div>
              <div style="font-size:13px; margin-top:8px; max-width:480px;
                          margin-left:auto; margin-right:auto;
                          line-height:1.6;">
                Cette section sera branchee au jour 2 de la refonte UX.
                Pour la beta, seul "Aujourd'hui" est actif.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()


# Streamlit execute le fichier de haut en bas a chaque rerun :
# on appelle main() directement pour declencher le rendu.
main()
