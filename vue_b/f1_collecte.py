"""
F1 - Collecte terrain.

L'operateur signale un incident (form + audio en etape 3). Tech N+1 le
fiabilise. Resp. AC le voit ensuite en F2.

Etape 2 (cette livraison) : formulaire minimal + liste + transition vers
'fiabilise' par Tech N+1. Audio (Whisper) viendra en etape 3.
"""
from __future__ import annotations

import streamlit as st

from core import db
from ui_theme import COLOR_TEXT_MUTED, COLOR_PRIMARY, COLOR_OK, section


SEVERITES = ["faible", "moyenne", "haute", "critique"]
TYPES_INCIDENT = [
    "Defaut produit", "Panne machine", "Derive parametre",
    "Probleme matiere", "Probleme methode", "Autre",
]


def _kpi_bar() -> None:
    """3 KPI rapides au dessus du form : bruts / fiabilises / total."""
    df_brut       = db.list_incidents(statut="brut")
    df_fiabilise  = db.list_incidents(statut="fiabilise")
    df_all        = db.list_incidents()
    n_brut, n_fia, n_all = len(df_brut), len(df_fiabilise), len(df_all)

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns: repeat(3, 1fr);
                    gap:12px; margin: 4px 0 18px 0;">
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">A traiter</div>
            <div style="font-size:24px; font-weight:700; color:{COLOR_PRIMARY};">{n_brut}</div>
          </div>
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Fiabilises</div>
            <div style="font-size:24px; font-weight:700; color:{COLOR_OK};">{n_fia}</div>
          </div>
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Total</div>
            <div style="font-size:24px; font-weight:700;">{n_all}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    role = st.session_state.get("role", "ac_manager")
    _kpi_bar()

    # --- Formulaire de saisie ----------------------------------------------
    with st.container(border=True):
        section("Nouveau signalement",
                "L'operateur saisit ce qu'il observe. Audio + photo : etape 3.")
        col1, col2 = st.columns(2)
        with col1:
            machine = st.text_input("Machine", placeholder="ex: M3 - Soudeuse")
            type_inc = st.selectbox("Type d'incident", TYPES_INCIDENT)
            severite = st.select_slider(
                "Severite",
                options=SEVERITES,
                value="moyenne",
            )
        with col2:
            operateur = st.text_input(
                "Operateur (optionnel)",
                value=st.session_state.get("user_name", ""),
            )
            description = st.text_area(
                "Description libre",
                placeholder=("Ce que tu vois, ce que tu entends, depuis quand,"
                             " ce qui a change..."),
                height=120,
            )
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            submit = st.button(
                "Creer l'incident",
                type="primary",
                use_container_width=True,
                disabled=not machine.strip(),
            )
        with col_info:
            st.caption("L'incident sera marque 'brut'. Le Technicien N+1"
                       " l'enrichira et le transmettra a la suite.")
        if submit:
            iid = db.create_incident(
                machine=machine,
                description=description,
                operateur_nom=operateur,
                type_incident=type_inc,
                severite=severite,
                cree_par_role=role,
            )
            st.success(f"Incident #{iid} cree (statut: brut)")
            st.rerun()

    # --- Liste des incidents bruts ----------------------------------------
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        section("Incidents en attente de fiabilisation",
                "Le Technicien N+1 nettoie, complete, puis valide.")
        df = db.list_incidents(statut="brut")
        if df.empty:
            st.info("Aucun incident brut en attente.")
        else:
            cols_show = [
                "id", "machine", "type_incident", "severite",
                "operateur_nom", "description", "cree_le",
            ]
            cols_show = [c for c in cols_show if c in df.columns]
            st.dataframe(
                df[cols_show],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "machine": "Machine",
                    "type_incident": "Type",
                    "severite": "Severite",
                    "operateur_nom": "Operateur",
                    "description": "Description",
                    "cree_le": "Cree le",
                },
            )

            # Action Tech N+1 : fiabiliser (etape 2 minimaliste)
            if role in ("technician", "ac_manager", "maintenance"):
                col_sel, col_btn = st.columns([2, 1])
                with col_sel:
                    target_id = st.number_input(
                        "ID a fiabiliser (Tech N+1)",
                        min_value=1, step=1, value=int(df.iloc[0]["id"]),
                    )
                with col_btn:
                    st.markdown(
                        "<div style='height:28px'></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Marquer fiabilise →",
                                 use_container_width=True):
                        ok = db.update_incident_statut(
                            int(target_id), "fiabilise",
                        )
                        if ok:
                            st.success(
                                f"Incident #{int(target_id)} marque fiabilise."
                                f" Il apparait maintenant dans F2 - Cadrage."
                            )
                            st.rerun()
                        else:
                            st.error(f"Incident #{int(target_id)} introuvable.")
