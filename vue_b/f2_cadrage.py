"""
F2 - Cadrage du probleme (QQOQCP).

Le Resp. AC reprend un incident fiabilise et demarre un projet AC.
Forme complet QQOQCP : etape 4. Ici on prepare juste la creation
du projet et la liste des projets en cadrage.
"""
from __future__ import annotations

import streamlit as st

from core import db
from ui_theme import section


def render() -> None:
    role = st.session_state.get("role", "ac_manager")
    user_name = st.session_state.get("user_name", "")

    # --- Demarrer un projet a partir d'un incident fiabilise --------------
    with st.container(border=True):
        section("Cadrer un probleme",
                "Choisis un incident a traiter, donne-lui un nom clair.")
        df_fia = db.list_incidents(statut="fiabilise")
        if df_fia.empty:
            st.info("Aucun incident fiabilise pour le moment. "
                    "Va sur F1 - Collecte pour fiabiliser un incident brut.")
        else:
            # Construit les options : "#id - machine - description (30c)"
            opts = []
            id_to_label: dict[int, str] = {}
            for _, row in df_fia.iterrows():
                rid = int(row["id"])
                lbl = (f"#{rid} - {row['machine']} - "
                       f"{(row['description'] or '')[:50]}")
                opts.append(rid)
                id_to_label[rid] = lbl

            col_inc, col_title = st.columns([1, 2])
            with col_inc:
                target_id = st.selectbox(
                    "Incident a cadrer",
                    options=opts,
                    format_func=lambda i: id_to_label.get(i, str(i)),
                )
            with col_title:
                titre = st.text_input(
                    "Titre du projet AC",
                    placeholder="ex: Reduire les defauts de soudure sur M3",
                )
            disabled = (not titre.strip()) or (role == "operator")
            if st.button(
                "Demarrer",
                type="primary",
                use_container_width=True,
                disabled=disabled,
                help=("Reserve aux non-operateurs" if role == "operator"
                      else "Crée le projet, on passe à la suite"),
            ):
                pid = db.create_projet_ac(
                    titre=titre,
                    incident_id=int(target_id),
                    cree_par=user_name,
                    cree_par_role=role,
                )
                db.update_incident_statut(int(target_id), "en_projet")
                st.success(
                    f"Projet #{pid} cree. Tu peux passer a Comprendre."
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
            cols_show = ["id", "titre", "incident_id", "cree_par",
                         "cree_par_role", "cree_le"]
            cols_show = [c for c in cols_show if c in df.columns]
            st.dataframe(
                df[cols_show], use_container_width=True, hide_index=True,
            )
