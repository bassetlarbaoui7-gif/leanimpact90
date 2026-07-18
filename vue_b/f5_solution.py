"""
F5 - Solution & adoption terrain (cascade de conviction).

Sur un projet valide, le Resp. AC propose une solution chiffree
(cout / delai / gain). Le logiciel la traduit ensuite dans le langage
de chaque role :
  - Resp. Production : cadence, euros, zero arret ligne -> il donne le feu vert
  - Tech N+1 : mode operatoire, temps gagne pour SON equipe -> il porte la
    consigne aupres des operateurs (c'est lui qui sait les convaincre)
  - Operateur : consigne courte et visuelle, pas un rapport

Champs DB utilises : solution_proposee, cout_estime, temps_estime_jours,
gain_estime_eur, gain_productivite (colonnes existantes de projets_ac).
"""
from __future__ import annotations

import streamlit as st

from core import db
from ui_theme import (
    COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_OK,
    COLOR_CARD, COLOR_BORDER, section,
)
from vue_b.synthese import render_synthese


# ---------------------------------------------------------------------------
# Selection projet (valide ou deja en solution)
# ---------------------------------------------------------------------------
def _select_projet() -> dict | None:
    df_v = db.list_projets_ac(statut="valide")
    df_s = db.list_projets_ac(statut="solution_propose")
    import pandas as pd
    df = pd.concat([df_v, df_s], ignore_index=True) \
        if not (df_v.empty and df_s.empty) else df_v
    if df.empty:
        st.info(
            "Aucun projet valide pour l'instant. La solution se propose "
            "apres l'alignement des parties prenantes (etape 4)."
        )
        return None
    opts = list(df["id"].astype(int))
    id_to_lbl = {
        int(r["id"]): f"#{int(r['id'])} - {r['titre']} ({r['statut']})"
        for _, r in df.iterrows()
    }
    pid = st.selectbox(
        "Projet", options=opts,
        format_func=lambda i: id_to_lbl.get(i, str(i)),
        key="f5_projet",
    )
    return db.get_projet_ac(int(pid))


# ---------------------------------------------------------------------------
# Fiches par role (traduction des gains)
# ---------------------------------------------------------------------------
def _fiche(titre_role: str, accroche: str, lignes: list[str],
           badge: str) -> str:
    items = "".join(
        f"<div style='font-size:12.5px;color:{COLOR_TEXT};line-height:1.55;"
        f"margin-bottom:4px;'>&#8226; {l}</div>"
        for l in lignes if l
    )
    return f"""
    <div style="background:{COLOR_CARD};border:1px solid {COLOR_BORDER};
                border-radius:12px;padding:16px 18px;height:100%;">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;margin-bottom:6px;">
        <div style="font-size:13px;color:{COLOR_PRIMARY};font-weight:700;">
          {titre_role}</div>
        <span style="background:{COLOR_OK}22;color:{COLOR_OK};
                     font-size:10px;font-weight:700;padding:3px 9px;
                     border-radius:999px;border:1px solid {COLOR_OK}55;">
          {badge}</span>
      </div>
      <div style="font-size:11.5px;color:{COLOR_TEXT_MUTED};
                  margin-bottom:10px;">{accroche}</div>
      {items}
    </div>
    """


def _render_fiches_roles(projet: dict) -> None:
    sol = (projet.get("solution_proposee") or "").strip()
    cout = projet.get("cout_estime") or 0
    delai = projet.get("temps_estime_jours") or 0
    gain_eur = projet.get("gain_estime_eur") or 0
    gain_prod = projet.get("gain_productivite") or 0

    consigne_courte = sol if len(sol) <= 140 else sol[:137] + "..."

    st.markdown(
        f"<div style='font-size:11px;font-weight:700;color:{COLOR_PRIMARY};"
        f"text-transform:uppercase;letter-spacing:0.05em;margin:14px 0 8px;'>"
        f"La meme solution, traduite pour chaque role — la cascade de "
        f"conviction</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(_fiche(
            "Responsable Production",
            "Il valide et donne le feu vert.",
            [
                f"Gain estime : <b>{gain_eur:,.0f} €/an</b>".replace(",", " ")
                if gain_eur else "Gain : non-conformites evitees",
                f"Productivite : <b>+{gain_prod:.1f} %</b>"
                if gain_prod else "",
                f"Mise en place : <b>{delai:.0f} jour(s)</b>, "
                f"sans arret ligne planifie" if delai else "",
                f"Cout : <b>{cout:,.0f} €</b>".replace(",", " ")
                if cout else "Cout : negligeable",
            ],
            "DECIDE",
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(_fiche(
            "Technicien N+1",
            "Convaincu, il porte la consigne aupres des operateurs.",
            [
                f"Mode operatoire : {consigne_courte}" if sol else
                "Mode operatoire a definir",
                "Moins de retouches a gerer pour son equipe",
                "Il adapte la consigne au terrain — c'est lui le relais",
            ],
            "PORTE",
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(_fiche(
            "Operateur",
            "Une consigne courte — pas un rapport de 15 pages.",
            [
                f"« {consigne_courte} »" if sol else "Consigne a definir",
                "Ce qui change a ton poste, rien de plus",
                "Bouton « je bloque » si ca ne marche pas : c'est traite",
            ],
            "APPLIQUE",
        ), unsafe_allow_html=True)

    st.caption(
        "Garantie Terrain : avant diffusion, l'action passe les 25 conditions "
        "d'adoption (benefice visible, sponsor, relais, equite 3x8, "
        "conformite...) — module en developpement, livre pour le POC."
    )


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------
def render() -> None:
    projet = _select_projet()
    if projet is None:
        render_synthese("f5")
        return
    pid = int(projet["id"])

    # --- Formulaire solution ----------------------------------------------
    with st.container(border=True):
        section("Proposer la solution",
                "Une solution chiffree : cout, delai, gain. Le logiciel "
                "la traduit ensuite pour chaque role.")
        sol_txt = st.text_area(
            "Solution (contre-mesure + verrou anti-recurrence)",
            value=projet.get("solution_proposee") or "",
            placeholder=("ex: Re-serrage du connecteur element chauffant "
                         "zone 6 + passage d'une plaque de profilage "
                         "obligatoire apres chaque maintenance four "
                         "(poka-yoke checklist)"),
            height=100,
            key="f5_sol_txt",
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cout = st.number_input(
                "Cout (€)", min_value=0.0, step=100.0,
                value=float(projet.get("cout_estime") or 0),
                key="f5_cout",
            )
        with c2:
            delai = st.number_input(
                "Delai (jours)", min_value=0.0, step=0.5,
                value=float(projet.get("temps_estime_jours") or 0),
                key="f5_delai",
            )
        with c3:
            gain_eur = st.number_input(
                "Gain (€/an)", min_value=0.0, step=1000.0,
                value=float(projet.get("gain_estime_eur") or 0),
                key="f5_gain",
            )
        with c4:
            gain_prod = st.number_input(
                "Productivite (+%)", min_value=0.0, step=0.5,
                value=float(projet.get("gain_productivite") or 0),
                key="f5_prod",
            )

        if st.button("Enregistrer et traduire par role",
                     type="primary",
                     use_container_width=True,
                     disabled=not sol_txt.strip(),
                     key="f5_save"):
            db.update_projet_ac(
                pid,
                solution_proposee=sol_txt.strip(),
                cout_estime=cout,
                temps_estime_jours=delai,
                gain_estime_eur=gain_eur,
                gain_productivite=gain_prod,
            )
            if projet.get("statut") == "valide":
                db.update_projet_ac(pid, statut="solution_propose")
            st.success(
                "Solution enregistree. Chaque role recoit sa fiche — "
                "le Tech N+1 porte la consigne au terrain."
            )
            st.rerun()

    # --- Fiches par role (si solution presente) ---------------------------
    if (projet.get("solution_proposee") or "").strip():
        _render_fiches_roles(projet)

    # --- Gains & livrables de l'etape --------------------------------------
    render_synthese("f5")
