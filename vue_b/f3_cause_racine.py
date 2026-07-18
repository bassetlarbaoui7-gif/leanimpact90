"""
F3 - Comprendre la vraie cause.

Design : VUE FOCUS (charge mentale minimale).
  - La cause la plus probable est mise en avant (grande carte verte)
  - Les 4 autres familles en liste compacte, chacune depliable
  - L'Ishikawa complet (arete de poisson) accessible en 1 clic
  - 1 clic pour valider l'analyse
"""
from __future__ import annotations

import streamlit as st

from core import db
from core.cbr import case_base as cb
from core.cbr.path_engine import (
    generate_full_tree, save_tree_to_db, BRANCHES_5M,
)
from core.cbr.classifier import analyze_node
from core.cbr.ishikawa_visual import (
    render_ishikawa_svg, ETAT_PROPOSE, ETAT_VALIDE,
)
from ui_theme import (
    COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_OK,
    COLOR_DANGER, COLOR_CARD, COLOR_BORDER, section,
)
from vue_b.synthese import render_synthese

_TYPE_COLORS = {
    "symptome":      COLOR_TEXT_MUTED,
    "condition":     "#3b82f6",
    "cause_directe": COLOR_PRIMARY,
    "cause_racine":  COLOR_OK,
}


# ---------------------------------------------------------------------------
# Selection du projet
# ---------------------------------------------------------------------------
def _select_projet() -> int | None:
    df = db.list_projets_ac()
    if df.empty:
        st.info("Aucun projet AC. Va sur F2 - Cadrage pour en demarrer un.")
        return None
    df = df[df["statut"] != "clos"] if "statut" in df.columns else df
    if df.empty:
        st.info("Aucun projet en cours d'analyse.")
        return None
    opts = list(df["id"].astype(int))
    id_to_lbl = {
        int(r["id"]): f"#{int(r['id'])} - {r['titre']} ({r['statut']})"
        for _, r in df.iterrows()
    }
    return st.selectbox(
        "Projet a analyser", options=opts,
        format_func=lambda i: id_to_lbl.get(i, str(i)),
        key="f3_projet_select",
    )


# ---------------------------------------------------------------------------
# Helpers donnees
# ---------------------------------------------------------------------------
def _branch_root_and_chain(projet_id: int, branche: str) -> tuple[dict | None, list[dict]]:
    """Retourne (noeud_cause_racine, chaine_complete) pour une branche."""
    arbre = cb.list_chemins_projet(projet_id, branche_m=branche)
    if arbre.empty:
        return None, []
    chaine = arbre.sort_values("niveau").to_dict("records")
    racine = None
    for n in chaine:
        if n.get("est_cause_racine"):
            racine = n
    if racine is None and chaine:
        racine = chaine[-1]
    return racine, chaine


def _all_branches_ranked(projet_id: int) -> list[dict]:
    """
    Retourne la liste des branches avec leur cause racine, triee par
    confiance decroissante. Chaque element :
      {branche, racine, chaine, confidence, etat}
    """
    out = []
    for branche in BRANCHES_5M:
        racine, chaine = _branch_root_and_chain(projet_id, branche)
        if racine is None:
            continue
        # Etat selon feedback
        score = cb.get_feedback_score(int(racine["id"]))
        etat = ETAT_VALIDE if score["net"] > 0 else ETAT_PROPOSE
        out.append({
            "branche":    branche,
            "racine":     racine,
            "chaine":     chaine,
            "confidence": float(racine.get("confidence", 0.5)),
            "etat":       etat,
        })
    out.sort(key=lambda x: -x["confidence"])
    return out


def _stars(conf: float) -> str:
    n = max(0, min(5, round(conf * 5)))
    return "★" * n + "☆" * (5 - n)


def _etats_dict(ranked: list[dict]) -> dict[str, str]:
    return {r["branche"]: r["etat"] for r in ranked}


# ---------------------------------------------------------------------------
# Rendu d'une chaine 5 Pourquoi (le detail) - utilise dans les expanders
# ---------------------------------------------------------------------------
def _render_chaine(chaine: list[dict]) -> None:
    for n in chaine:
        reponse = str(n["reponse"])
        niveau  = int(n["niveau"])
        analyse = analyze_node(reponse, niveau)
        col     = _TYPE_COLORS.get(analyse["type"], COLOR_PRIMARY)
        badge_root = (
            f'<span style="background:{COLOR_OK};color:white;font-size:9px;'
            f'padding:2px 7px;border-radius:999px;margin-left:8px;">RACINE</span>'
            if analyse["is_root"] else ""
        )
        st.markdown(
            f"""
            <div style="display:flex;gap:10px;align-items:flex-start;
                        padding:8px 12px;margin-bottom:5px;background:{COLOR_CARD};
                        border-left:3px solid {col};border-radius:8px;">
              <div style="background:{col};color:white;min-width:22px;height:22px;
                          border-radius:50%;display:flex;align-items:center;
                          justify-content:center;font-size:11px;font-weight:700;">
                {niveau}</div>
              <div style="flex:1;">
                <div style="font-size:10.5px;color:{COLOR_TEXT_MUTED};">{n['question']}</div>
                <div style="font-size:13px;color:{COLOR_TEXT};font-weight:600;">
                  {reponse}{badge_root}</div>
                <div style="font-size:10px;color:{col};margin-top:2px;">
                  {analyse['type_label']} &middot; {n['confidence']:.0%}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if analyse["is_blame"]:
            st.markdown(
                f"""
                <div style="background:rgba(239,68,68,0.10);border:1px solid {COLOR_DANGER};
                            border-radius:8px;padding:7px 11px;margin:0 0 6px 32px;
                            font-size:11px;color:{COLOR_TEXT};">
                  ⚠ <b style="color:{COLOR_DANGER};">Piege blame humain.</b>
                  {analyse['relance_question']}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Vue Focus
# ---------------------------------------------------------------------------
def _render_focus(projet_id: int, projet: dict) -> None:
    ranked = _all_branches_ranked(projet_id)
    if not ranked:
        st.warning("Arbre vide - relance l'analyse.")
        return

    principale = ranked[0]
    autres     = ranked[1:]

    # --- Carte cause principale ---
    r = principale
    col_etat = COLOR_OK if r["etat"] == ETAT_VALIDE else COLOR_PRIMARY
    chaine_txt = " → ".join(
        str(n["reponse"]) for n in r["chaine"]
    )
    st.markdown(
        f"""
        <div style="font-size:11px;color:{COLOR_TEXT_MUTED};font-weight:600;
                    letter-spacing:0.06em;margin-bottom:8px;">
          CAUSE RACINE LA PLUS PROBABLE
        </div>
        <div style="background:linear-gradient(135deg,rgba(34,197,94,0.10),
                    rgba(34,197,94,0.02));border:1px solid {col_etat};
                    border-radius:16px;padding:22px 24px;margin-bottom:24px;">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:10px;">
            <span style="background:{col_etat};color:white;font-size:11px;
                         font-weight:700;padding:4px 12px;border-radius:999px;">
              ◆ {r['branche'].upper()}</span>
            <span style="color:{col_etat};font-size:14px;font-weight:700;">
              {_stars(r['confidence'])} {r['confidence']:.0%}</span>
          </div>
          <div style="font-size:21px;font-weight:800;color:{COLOR_TEXT};
                      line-height:1.25;margin-bottom:10px;">
            {r['racine']['reponse']}</div>
          <div style="font-size:12.5px;color:{COLOR_TEXT_MUTED};line-height:1.5;">
            On va du visible jusqu'a la vraie cause : {chaine_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Detail de la cause principale (deplie par defaut)
    with st.expander(f"Voir le detail — {principale['branche']}", expanded=False):
        _render_chaine(principale["chaine"])

    # --- Autres familles ---
    st.markdown(
        f"<div style='font-size:11px;color:{COLOR_TEXT_MUTED};font-weight:600;"
        f"letter-spacing:0.06em;margin:6px 0 12px 0;'>"
        f"AUTRES PISTES A REGARDER</div>",
        unsafe_allow_html=True,
    )
    for r in autres:
        col_etat = COLOR_OK if r["etat"] == ETAT_VALIDE else "#fb923c"
        with st.expander(
            f"{r['branche']}  —  {r['racine']['reponse'][:50]}   "
            f"{_stars(r['confidence'])}",
            expanded=False,
        ):
            _render_chaine(r["chaine"])


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------
def render() -> None:
    projet_id = _select_projet()
    if projet_id is None:
        return

    projet = db.get_projet_ac(projet_id)
    arbre_existant = cb.list_chemins_projet(projet_id)
    a_un_arbre = not arbre_existant.empty

    # --- Lancer / relancer l'analyse ---
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if not a_un_arbre:
            st.markdown(
                f"<div style='color:{COLOR_TEXT_MUTED};font-size:13px;'>"
                "Lance l'analyse. Les 5 Pourquoi par famille apparaissent "
                "en quelques secondes, prets a valider.</div>",
                unsafe_allow_html=True,
            )
    with col_b:
        label = "Relancer l'analyse" if a_un_arbre else "Lancer l'analyse"
        if st.button(label, type="primary", use_container_width=True, key="f3_run"):
            with st.spinner("Analyse en cours..."):
                incident_id = projet.get("incident_id")
                if not incident_id:
                    st.error("Projet sans incident lie. Recree-le depuis F2.")
                    return
                result = generate_full_tree(int(incident_id))
                save_tree_to_db(projet_id, result["tree"])
                db.update_projet_ac(projet_id, statut="analyse")
                st.session_state["f3_last_similar"] = result["similar_cases"]
                st.session_state["f3_last_conf"] = result["global_confidence"]
            st.rerun()

    if not a_un_arbre:
        return

    # --- Bandeau contexte ---
    similar = st.session_state.get("f3_last_similar")
    conf    = st.session_state.get("f3_last_conf")
    if similar is not None and not similar.empty:
        top = similar.iloc[0]
        msg = (f"Un cas tres proche a deja ete traite chez vous "
               f"(cas <b>#{int(top['id'])}</b>) - on s'en inspire.")
        if conf:
            msg += f" &middot; fiabilite {conf:.0%}"
    else:
        msg = ("C'est le premier cas de ce type. "
               "Plus vous validez, plus le logiciel devient sur.")
        if conf:
            msg += f" &middot; fiabilite {conf:.0%}"
    st.markdown(
        f"""
        <div style="display:flex;gap:12px;align-items:center;background:{COLOR_CARD};
                    border:1px solid {COLOR_BORDER};border-radius:10px;
                    padding:10px 16px;margin:8px 0 18px 0;">
          <span style="color:{COLOR_PRIMARY};font-weight:700;">●</span>
          <span style="color:{COLOR_TEXT};font-size:13px;">{msg}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- VUE FOCUS ---
    _render_focus(projet_id, projet)

    # --- Toggle vue detaillee ---
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    show_ishikawa = st.toggle("Voir la vue detaillee (pour reunions et audits)",
                              key="f3_show_ishikawa")
    if show_ishikawa:
        ranked = _all_branches_ranked(projet_id)
        etats  = _etats_dict(ranked)
        tree_svg = {
            b: cb.list_chemins_projet(projet_id, branche_m=b)
                  .sort_values("niveau").to_dict("records")
            for b in BRANCHES_5M
        }
        svg = render_ishikawa_svg(tree_svg, projet.get("titre", "Probleme"),
                                  etats=etats)
        st.markdown(svg, unsafe_allow_html=True)

    # --- Correction humaine d'un noeud (l'IA propose, vous decidez) ---
    with st.expander("✎ Corriger un noeud de l'analyse", expanded=False):
        st.caption(
            "L'IA propose, vous decidez. Votre correction est enregistree "
            "et entraine le modele pour les prochains cas."
        )
        branche_corr = st.selectbox(
            "Branche", options=list(BRANCHES_5M), key="f3_corr_branche",
        )
        chaine_df = cb.list_chemins_projet(
            projet_id, branche_m=branche_corr,
        )
        if chaine_df.empty:
            st.info("Aucun noeud dans cette branche.")
        else:
            chaine_df = chaine_df.sort_values("niveau")
            node_ids = list(chaine_df["id"].astype(int))
            id_to_node = {
                int(r["id"]): (f"N{int(r['niveau'])} — "
                               f"{str(r['reponse'])[:70]}")
                for _, r in chaine_df.iterrows()
            }
            node_id = st.selectbox(
                "Noeud a corriger", options=node_ids,
                format_func=lambda i: id_to_node.get(i, str(i)),
                key="f3_corr_node",
            )
            current = chaine_df[chaine_df["id"] == node_id].iloc[0]
            # key dynamique par noeud : le champ suit la selection et
            # ne peut pas ecraser un autre noeud apres un rerun.
            new_reponse = st.text_input(
                "Nouvelle formulation",
                value=str(current["reponse"]),
                key=f"f3_corr_txt_{int(node_id)}",
            )
            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                mark_root = st.checkbox(
                    "Marquer comme cause racine",
                    value=bool(current.get("est_cause_racine")),
                    key="f3_corr_root",
                )
            with col_c2:
                if st.button("Enregistrer la correction",
                             type="primary",
                             use_container_width=True,
                             key="f3_corr_save",
                             disabled=not new_reponse.strip()):
                    ok = cb.update_chemin(
                        int(node_id),
                        reponse=new_reponse.strip(),
                        est_cause_racine=mark_root,
                    )
                    if ok:
                        st.success(
                            "Noeud corrige. Le modele tiendra compte de "
                            "cette formulation a la validation."
                        )
                        st.rerun()
                    else:
                        st.error("Correction impossible (noeud introuvable).")

    # --- Validation 1 clic ---
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    col_v1, col_v2 = st.columns([2, 1])
    with col_v2:
        role = st.session_state.get("role", "ac_manager")
        if st.button("✓ Valider l'analyse", type="primary",
                     use_container_width=True, key="f3_validate"):
            arbre = cb.list_chemins_projet(projet_id)
            racines = arbre[arbre["est_cause_racine"] == 1]
            for _, rr in racines.iterrows():
                cb.add_feedback(
                    int(rr["id"]), +1,
                    valide_par=st.session_state.get("user_name", ""),
                    role_valideur=role,
                    commentaire="Validation analyse F3",
                )
            db.update_projet_ac(projet_id, statut="validation_en_cours")
            st.success(
                "Valide. L'equipe va aligner chacun de son cote, sans reunion."
            )
            st.rerun()
    with col_v1:
        st.caption(
            "En validant, vous gagnez du temps pour la prochaine fois : "
            "le logiciel s'en souvient."
        )

    # --- Gains & livrables de l'etape ---
    render_synthese("f3")
