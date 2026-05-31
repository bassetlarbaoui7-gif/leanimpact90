"""
F5 - Solution & faisabilite + gains par role.

Sur un projet valide, le Resp. AC propose une solution. Le logiciel
calcule cout / temps / gain et les traduit pour chaque direction
(Resp. Prod = productivite+€, Tech N+1 = facilite ops, Op = gain direct).

Etape 2 : liste des projets valides + placeholder. Formulaire complet
+ moteur de faisabilite viendront en etape 6.
"""
from __future__ import annotations

import streamlit as st

from core import db
from ui_theme import COLOR_PRIMARY, COLOR_TEXT_MUTED, section


def render() -> None:
    with st.container(border=True):
        section("Pret a decider",
                "L'equipe a aligne. A vous de choisir la solution.")
        df = db.list_projets_ac(statut="valide")
        if df.empty:
            st.info("Rien a decider pour l'instant.")
        else:
            cols_show = ["id", "titre", "cree_par", "cree_le"]
            cols_show = [c for c in cols_show if c in df.columns]
            st.dataframe(df[cols_show], use_container_width=True,
                         hide_index=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Carte : chaque role voit le gain dans SA langue
    st.markdown(
        f"""
        <div style="padding:20px 22px; background:#111118;
                    border:1px solid #27272f; border-radius:14px;">
          <div style="font-weight:700; color:white; margin-bottom:6px;">
            Le gain visible pour chacun
          </div>
          <div style="color:{COLOR_TEXT_MUTED}; font-size:12.5px;
                      margin-bottom:14px;">
            Pas de chiffres abstraits. Chaque role voit ce qui compte
            pour lui.
          </div>
          <div style="display:grid; grid-template-columns:repeat(3, 1fr);
                      gap:14px;">
            <div>
              <div style="font-size:12px; color:{COLOR_PRIMARY};
                          font-weight:600; margin-bottom:4px;">
                Responsable Production
              </div>
              <div style="color:{COLOR_TEXT_MUTED}; font-size:12.5px;
                          line-height:1.55;">
                Productivite gagnee, euros par mois, retour sur
                investissement
              </div>
            </div>
            <div>
              <div style="font-size:12px; color:{COLOR_PRIMARY};
                          font-weight:600; margin-bottom:4px;">
                Technicien N+1
              </div>
              <div style="color:{COLOR_TEXT_MUTED}; font-size:12.5px;
                          line-height:1.55;">
                Interventions evitees, heures recuperees,
                moins de stress
              </div>
            </div>
            <div>
              <div style="font-size:12px; color:{COLOR_PRIMARY};
                          font-weight:600; margin-bottom:4px;">
                Operateur
              </div>
              <div style="color:{COLOR_TEXT_MUTED}; font-size:12.5px;
                          line-height:1.55;">
                Temps de setup reduit, moins de defauts, poste plus
                serein
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
