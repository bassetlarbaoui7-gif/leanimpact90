"""
F6 - Suivi d'action + ROI reel.

Apres validation de la solution, les actions sont planifiees, assignees,
suivies. ROI reel mesure a la cloture.

Etape 2 : liste des actions existantes + creation manuelle + statut.
Tableau de bord ROI viendra en etape 7.
"""
from __future__ import annotations

import streamlit as st

from core import db
from ui_theme import COLOR_TEXT_MUTED, COLOR_OK, COLOR_DANGER, section


STATUTS_ACTION = ["a_faire", "en_cours", "fait", "bloque"]
COULEURS_STATUT = {
    "a_faire":  COLOR_TEXT_MUTED,
    "en_cours": COLOR_DANGER,
    "fait":     COLOR_OK,
    "bloque":   COLOR_DANGER,
}


def _kpi_actions() -> None:
    df = db.list_actions()
    n_total = len(df)
    n_fait = int((df["statut"] == "fait").sum()) if not df.empty else 0
    n_bloque = int((df["statut"] == "bloque").sum()) if not df.empty else 0
    pct_fait = (n_fait / n_total * 100.0) if n_total else 0.0

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:repeat(3, 1fr);
                    gap:12px; margin:4px 0 18px 0;">
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Actions</div>
            <div style="font-size:24px; font-weight:700;">{n_total}</div>
          </div>
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Avancement</div>
            <div style="font-size:24px; font-weight:700; color:{COLOR_OK};">
              {pct_fait:.0f}%
            </div>
          </div>
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Bloquees</div>
            <div style="font-size:24px; font-weight:700; color:{COLOR_DANGER};">
              {n_bloque}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    _kpi_actions()

    # --- Creation manuelle d'une action ----------------------------------
    df_proj = db.list_projets_ac()
    with st.container(border=True):
        section("Ajouter une action",
                "Pour que la solution se transforme en resultat.")
        if df_proj.empty:
            st.info("Aucun projet en cours.")
        else:
            opts = list(df_proj["id"].astype(int))
            id_to_lbl = {
                int(r["id"]): f"#{int(r['id'])} - {r['titre']}"
                for _, r in df_proj.iterrows()
            }
            col_p, col_t, col_pr = st.columns([1, 2, 1])
            with col_p:
                target_pid = st.selectbox(
                    "Projet",
                    options=opts,
                    format_func=lambda i: id_to_lbl.get(i, str(i)),
                )
            with col_t:
                titre_action = st.text_input(
                    "Titre",
                    placeholder="ex: Recalibrer soudeuse M3",
                )
            with col_pr:
                priorite = st.selectbox("Priorite", [1, 2, 3],
                                        format_func=lambda p: (
                                            "1 (critique)" if p == 1 else
                                            "2 (normale)" if p == 2 else
                                            "3 (basse)"
                                        ))
            assignee = st.text_input("Assignee", placeholder="ex: Tech equipe A")
            if st.button("Creer", type="primary",
                         disabled=not titre_action.strip(),
                         use_container_width=True):
                aid = db.add_action(
                    projet_id=int(target_pid),
                    titre=titre_action,
                    assignee=assignee,
                    priorite=priorite,
                )
                st.success(f"Action #{aid} creee. A toi de la suivre.")
                st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # --- Plan d'action global --------------------------------------------
    with st.container(border=True):
        section("Toutes vos actions",
                "Le critique d'abord. Personne n'oublie.")
        df = db.list_actions()
        if df.empty:
            st.info("Pas encore d'action a suivre.")
        else:
            cols_show = ["id", "projet_id", "titre", "assignee",
                         "priorite", "statut", "echeance", "cree_le"]
            cols_show = [c for c in cols_show if c in df.columns]
            st.dataframe(
                df[cols_show], use_container_width=True, hide_index=True,
            )
