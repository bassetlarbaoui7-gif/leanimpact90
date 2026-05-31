"""
F4 - Validation distribuee.

Une fois les causes IA validees par le Resp. AC, le projet passe en
validation_en_cours. Resp. Prod et Tech N+1 valident chacun de leur cote,
sans reunion.

Etape 2 : liste des projets en validation + bouton manuel valide/refuse
(workflow distribue propre arrive en etape 5).
"""
from __future__ import annotations

import streamlit as st

from core import db, workflow
from core.workflow import ProjetStatus
from ui_theme import section


def _can_validate(role: str) -> bool:
    return role in ("production", "technician", "maintenance", "ac_manager")


def render() -> None:
    role = st.session_state.get("role", "ac_manager")
    user_name = st.session_state.get("user_name", "")

    with st.container(border=True):
        section("A valider de votre cote",
                "Chacun valide depuis son poste. Pas besoin de se reunir.")
        df = db.list_projets_ac(statut="validation_en_cours")
        if df.empty:
            st.info("Rien a valider pour le moment.")
            return

        cols_show = ["id", "titre", "cree_par", "cree_le"]
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True, hide_index=True)

        if not _can_validate(role):
            st.caption(
                "Tu peux consulter mais pas valider depuis ce role."
            )
            return

        # --- Action de validation -----------------------------------------
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_id, col_dec, col_btn = st.columns([1, 1, 1])
        with col_id:
            target_id = st.number_input(
                "Projet a valider",
                min_value=1, step=1,
                value=int(df.iloc[0]["id"]),
            )
        with col_dec:
            decision = st.selectbox(
                "Decision",
                ["valide", "refuse"],
            )
        with col_btn:
            commentaire = st.text_input(
                "Commentaire (optionnel)", value="",
            )

        if st.button("Envoyer ma decision", type="primary",
                     use_container_width=True):
            # Trace la validation
            db.add_validation(
                projet_id=int(target_id),
                role_valideur=role,
                decision=decision,
                nom_valideur=user_name,
                commentaire=commentaire,
            )
            target_state = (
                ProjetStatus.VALIDE if decision == "valide"
                else ProjetStatus.REFUSE
            )
            ok, msg = workflow.transition(int(target_id), target_state)
            if ok:
                st.success(
                    f"Decision enregistree pour le projet #{int(target_id)}. "
                    f"L'equipe est tenue au courant."
                )
                st.rerun()
            else:
                st.warning(f"Pas encore possible : {msg}")
