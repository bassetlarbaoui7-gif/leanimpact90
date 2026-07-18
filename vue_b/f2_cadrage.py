"""
F2 - Cadrage du probleme (catalogue de defauts + wizard QQOQCCP + containment).

Le Resp. AC reprend un incident fiabilise et demarre un projet AC :
  1. Choix de l'incident a traiter
  2. Catalogue de defauts normalise -> pre-remplit titre et QQOQCCP
  3. Wizard QQOQCCP 7 questions, pre-rempli depuis l'incident
     (colonnes qqoqcp_* de projets_ac)
  4. Action de protection (containment) creee dans le meme geste
     (action priorite 1 taggee [CONTAINMENT])
"""
from __future__ import annotations

import streamlit as st

from core import db
from ui_theme import COLOR_TEXT_MUTED, COLOR_PRIMARY, section
from vue_b.synthese import render_synthese


# ---------------------------------------------------------------------------
# Catalogue de defauts normalise (#1) — les plus frequents par metier.
# Chaque fiche pre-remplit le titre projet et le QQOQCCP.
# Les defauts rares passent par "Autre (saisie libre)".
# ---------------------------------------------------------------------------
CATALOGUE_DEFAUTS = [
    {"code": "BRA-01", "libelle": "Joints brases froids (AOI)",
     "famille": "Brasage / refusion",
     "quoi": "Joints brases froids detectes a l'AOI"},
    {"code": "BRA-02", "libelle": "Voids excessifs sous BGA",
     "famille": "Brasage / serigraphie",
     "quoi": "Voids excessifs sous billes BGA (inspection RX)"},
    {"code": "BRA-03", "libelle": "Tombstoning composants chip",
     "famille": "Brasage / refusion",
     "quoi": "Tombstoning sur composants chip (0402/0603)"},
    {"code": "SER-01", "libelle": "Depot de pate irregulier (SPI)",
     "famille": "Serigraphie",
     "quoi": "Epaisseur de pate a braser hors tolerance au SPI"},
    {"code": "CMS-01", "libelle": "Composant absent / decale",
     "famille": "Pose CMS",
     "quoi": "Composant absent ou decale detecte a l'AOI"},
    {"code": "QUA-01", "libelle": "Derive parametre process",
     "famille": "Process",
     "quoi": "Derive d'un parametre process hors limites"},
    {"code": "EMB-01", "libelle": "Defaut de collage / soudure sac",
     "famille": "Sacherie / emballage",
     "quoi": "Defaut de collage ou de soudure sur ligne sacherie"},
    {"code": "AUT-00", "libelle": "Autre (saisie libre)",
     "famille": "—", "quoi": ""},
]


def _catalogue_labels() -> list[str]:
    return [f"{d['code']} — {d['libelle']}" for d in CATALOGUE_DEFAUTS]


def _defaut_by_label(label: str) -> dict:
    code = label.split(" — ")[0]
    for d in CATALOGUE_DEFAUTS:
        if d["code"] == code:
            return d
    return CATALOGUE_DEFAUTS[-1]


# ---------------------------------------------------------------------------
# Wizard QQOQCCP (7 questions, pre-rempli depuis incident + catalogue)
# ---------------------------------------------------------------------------
def _wizard_qqoqccp(incident: dict, defaut: dict) -> dict:
    """Affiche les 7 champs et retourne les valeurs saisies."""
    st.markdown(
        f"<div style='font-size:11px;font-weight:700;color:{COLOR_PRIMARY};"
        f"text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 4px;'>"
        f"QQOQCCP — pre-rempli, ne complete que le specifique</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        quoi = st.text_input(
            "Quoi (le probleme)",
            value=defaut["quoi"] or (incident.get("description") or "")[:120],
            key="f2_q_quoi",
        )
        ou = st.text_input(
            "Ou (ligne / machine / zone)",
            value=incident.get("machine") or "",
            key="f2_q_ou",
        )
        quand = st.text_input(
            "Quand (debut, evolution)",
            placeholder="ex: depuis le 30/06, aggravation continue",
            key="f2_q_quand",
        )
        qui = st.text_input(
            "Qui (equipes / postes touches)",
            value=(f"Signale par {incident.get('operateur_nom')}"
                   if incident.get("operateur_nom") else ""),
            placeholder="ex: 3 equipes touchees, nuit davantage",
            key="f2_q_qui",
        )
    with col2:
        comment = st.text_input(
            "Comment (mode de detection)",
            placeholder="ex: detection AOI / controle final",
            key="f2_q_comment",
        )
        combien = st.text_input(
            "Combien (ampleur chiffree)",
            placeholder="ex: NC 0,3 % -> 3 % en 17 jours",
            key="f2_q_combien",
        )
        pourquoi = st.text_input(
            "Pourquoi c'est important (enjeu)",
            placeholder="ex: risque livraison client + cout retouche",
            key="f2_q_pourquoi",
        )
    return {
        "qqoqcp_quoi": (quoi + (f" — Ampleur : {combien}" if combien else "")),
        "qqoqcp_ou": ou,
        "qqoqcp_quand": quand,
        "qqoqcp_qui": qui,
        "qqoqcp_comment": comment,
        "qqoqcp_pourquoi": pourquoi,
    }


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------
def render() -> None:
    role = st.session_state.get("role", "ac_manager")
    user_name = st.session_state.get("user_name", "")

    # --- Demarrer un projet a partir d'un incident fiabilise --------------
    with st.container(border=True):
        section("Cadrer un probleme",
                "Catalogue -> QQOQCCP pre-rempli -> containment. "
                "10 minutes, homogene, traçable.")
        df_fia = db.list_incidents(statut="fiabilise")
        if df_fia.empty:
            st.info("Aucun incident fiabilise pour le moment. "
                    "Va sur F1 - Collecte pour fiabiliser un incident brut.")
        else:
            # 1. Incident
            opts = []
            id_to_label: dict[int, str] = {}
            for _, row in df_fia.iterrows():
                rid = int(row["id"])
                lbl = (f"#{rid} - {row['machine']} - "
                       f"{(row['description'] or '')[:50]}")
                opts.append(rid)
                id_to_label[rid] = lbl

            col_inc, col_def = st.columns([1, 1])
            with col_inc:
                target_id = st.selectbox(
                    "Incident a cadrer",
                    options=opts,
                    format_func=lambda i: id_to_label.get(i, str(i)),
                    key="f2_incident",
                )
            incident = db.get_incident(int(target_id)) or {}

            # 2. Catalogue de defauts
            with col_def:
                label_defaut = st.selectbox(
                    "Defaut (catalogue normalise)",
                    options=_catalogue_labels(),
                    key="f2_defaut",
                    help=("Les 20-30 defauts les plus frequents, "
                          "standardises. Le defaut choisi pre-remplit "
                          "le QQOQCCP et alimente la recherche de cas "
                          "similaires."),
                )
            defaut = _defaut_by_label(label_defaut)

            # 3. Titre auto-propose
            titre_sugg = ""
            if defaut["code"] != "AUT-00":
                titre_sugg = (f"Reduire {defaut['libelle'].lower()} — "
                              f"{incident.get('machine', '')}")
            titre = st.text_input(
                "Titre du projet AC",
                value=titre_sugg,
                placeholder="ex: Reduire les joints brases froids — CMS 2",
                key="f2_titre",
            )

            # 4. Wizard QQOQCCP
            qq = _wizard_qqoqccp(incident, defaut)

            # 5. Containment dans le meme geste
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;"
                f"color:{COLOR_PRIMARY};text-transform:uppercase;"
                f"letter-spacing:0.05em;margin:12px 0 4px;'>"
                f"Containment — proteger le client tout de suite</div>",
                unsafe_allow_html=True,
            )
            col_c1, col_c2 = st.columns([2, 1])
            with col_c1:
                containment_txt = st.text_input(
                    "Action de protection immediate",
                    placeholder=("ex: tri des lots depuis le 29/06 + "
                                 "controle AOI renforce"),
                    key="f2_containment",
                )
            with col_c2:
                containment_qui = st.text_input(
                    "Equipe notifiee",
                    placeholder="ex: Chef equipe A",
                    key="f2_containment_qui",
                )

            # 6. Creation
            disabled = (not titre.strip()) or (role == "operator")
            if st.button(
                "Creer le projet cadre",
                type="primary",
                use_container_width=True,
                disabled=disabled,
                help=("Reserve aux non-operateurs" if role == "operator"
                      else "Cree le projet avec QQOQCCP + containment"),
                key="f2_submit",
            ):
                pid = db.create_projet_ac(
                    titre=titre,
                    incident_id=int(target_id),
                    cree_par=user_name,
                    cree_par_role=role,
                )
                db.update_projet_ac(pid, **qq)
                db.update_incident_statut(int(target_id), "en_projet")
                if containment_txt.strip():
                    db.add_action(
                        projet_id=pid,
                        titre=f"[CONTAINMENT] {containment_txt.strip()}",
                        assignee=containment_qui.strip(),
                        priorite=1,
                    )
                    st.success(
                        f"Projet #{pid} cree, QQOQCCP enregistre, "
                        f"containment actif (action priorite 1). "
                        f"Passe a l'analyse de cause racine."
                    )
                else:
                    st.success(
                        f"Projet #{pid} cree, QQOQCCP enregistre. "
                        f"Passe a l'analyse de cause racine."
                    )
                st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # --- Projets en cours de cadrage --------------------------------------
    with st.container(border=True):
        section("Problemes en cours de cadrage",
                "A reprendre pour passer a la suite.")
        df = db.list_projets_ac(statut="cadrage")
        if df.empty:
            st.info("Rien a cadrer pour l'instant.")
        else:
            cols_show = ["id", "titre", "incident_id", "qqoqcp_quoi",
                         "cree_par", "cree_le"]
            cols_show = [c for c in cols_show if c in df.columns]
            st.dataframe(
                df[cols_show], use_container_width=True, hide_index=True,
            )

    # --- Gains & livrables de l'etape --------------------------------------
    render_synthese("f2")
