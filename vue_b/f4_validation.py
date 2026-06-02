"""
F4 - Aligner (validation distribuee, sans reunion).

Logique :
  Apres F3, le projet est en statut "validation_en_cours".
  Il a besoin de DEUX validations distinctes pour passer en "valide" :
    - 1 validation cote technique (Tech N+1 ou Maintenance)
    - 1 validation cote business (Resp. Production)
  Si l'un des deux refuse, le projet repart en "cadrage" pour ajustement.
  Resp. AC voit l'avancement global. Operateur n'a rien a faire ici.

Tout se passe en parallele, chacun depuis son poste, quand il peut.
"""
from __future__ import annotations

import streamlit as st

from core import db, workflow
from core.workflow import ProjetStatus
from core.cbr import case_base as cb
from core.cbr.path_engine import BRANCHES_5M
from core.cbr.feedback import (
    record_validation_outcome, enrich_from_validated_projet,
)
from ui_theme import (
    COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_OK, COLOR_DANGER,
    COLOR_CARD, COLOR_BORDER, section,
)


# Groupes de roles - qui valide quoi
ROLES_TECH = ("technician", "maintenance")
ROLES_PROD = ("production",)


# ---------------------------------------------------------------------------
# Etat de validation d'un projet
# ---------------------------------------------------------------------------
def _validation_state(projet_id: int) -> dict:
    """
    Retourne l'etat des validations d'un projet :
      tech_done / tech_decision (valide/refuse) / tech_at / tech_by
      prod_done / prod_decision / prod_at / prod_by
      all_valid : True si TECH=valide ET PROD=valide
      any_refused : True si l'un des 2 a refuse
    """
    state = {
        "tech_done": False, "tech_decision": None,
        "tech_at": None, "tech_by": "",
        "prod_done": False, "prod_decision": None,
        "prod_at": None, "prod_by": "",
    }
    vals = db.list_validations(projet_id)
    if vals.empty:
        state["all_valid"] = False
        state["any_refused"] = False
        return state

    # On prend la DERNIERE decision par groupe (la plus recente)
    for _, v in vals.iterrows():
        role = v["role_valideur"]
        if role in ROLES_TECH and not state["tech_done"]:
            state["tech_done"]     = True
            state["tech_decision"] = v["decision"]
            state["tech_at"]       = v["cree_le"]
            state["tech_by"]       = v.get("nom_valideur", "") or ""
        elif role in ROLES_PROD and not state["prod_done"]:
            state["prod_done"]     = True
            state["prod_decision"] = v["decision"]
            state["prod_at"]       = v["cree_le"]
            state["prod_by"]       = v.get("nom_valideur", "") or ""

    state["all_valid"] = (
        state["tech_decision"] == "valide" and
        state["prod_decision"] == "valide"
    )
    state["any_refused"] = (
        state["tech_decision"] == "refuse" or
        state["prod_decision"] == "refuse"
    )
    return state


def _role_can_validate(role: str, state: dict) -> tuple[bool, str]:
    """
    True si le user actuel a encore une validation a faire sur ce projet.
    Retourne (peut_valider, raison).
    """
    if role in ROLES_TECH:
        if state["tech_done"]:
            return False, f"Deja vote ({state['tech_decision']})"
        return True, "En attente de votre vote technique"
    if role in ROLES_PROD:
        if state["prod_done"]:
            return False, f"Deja vote ({state['prod_decision']})"
        return True, "En attente de votre vote production"
    return False, "Ce role n'est pas valideur"


def _apply_transitions_if_complete(projet_id: int, state: dict) -> str | None:
    """
    Si les deux validations sont la (succes OU refus), applique la
    transition workflow. Retourne le nouveau statut ou None.
    """
    if state["any_refused"]:
        # Au moins un refus : projet repart en cadrage
        ok, _ = workflow.transition(projet_id, ProjetStatus.REFUSE)
        if ok:
            workflow.transition(projet_id, ProjetStatus.CADRE)
            return "refuse_back_to_cadrage"
    elif state["tech_done"] and state["prod_done"] and state["all_valid"]:
        # Les deux ont valide
        ok, _ = workflow.transition(projet_id, ProjetStatus.VALIDE)
        if ok:
            return "valide"
    return None


# ---------------------------------------------------------------------------
# Carte d'un projet (rendu commun)
# ---------------------------------------------------------------------------
def _render_projet_card(projet: dict, state: dict) -> None:
    """Affiche l'etat d'un projet : titre + 2 pastilles (Tech / Prod)."""
    def _pill(label: str, done: bool, decision: str | None) -> str:
        if not done:
            color = COLOR_TEXT_MUTED
            txt = f"{label} : en attente"
        elif decision == "valide":
            color = COLOR_OK
            txt = f"{label} : valide"
        else:
            color = COLOR_DANGER
            txt = f"{label} : refus"
        return (
            f'<span style="background:{color}22;color:{color};font-size:11px;'
            f'font-weight:600;padding:4px 12px;border-radius:999px;'
            f'border:1px solid {color}55;">{txt}</span>'
        )

    st.markdown(
        f"""
        <div style="padding:14px 18px;background:{COLOR_CARD};
                    border:1px solid {COLOR_BORDER};border-radius:12px;
                    margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:8px;">
            <div style="font-size:14px;font-weight:700;color:{COLOR_TEXT};">
              Projet #{projet['id']}  &middot;  {projet['titre']}
            </div>
            <div style="font-size:11px;color:{COLOR_TEXT_MUTED};">
              cree par {projet.get('cree_par', '—') or '—'}
            </div>
          </div>
          <div style="display:flex;gap:8px;">
            {_pill('Technique', state['tech_done'], state['tech_decision'])}
            {_pill('Production', state['prod_done'], state['prod_decision'])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Detail d'un projet : contexte + causes racines proposees
# ---------------------------------------------------------------------------
def _render_projet_detail(projet_id: int) -> None:
    arbre = cb.list_chemins_projet(projet_id)
    if arbre.empty:
        st.warning("Pas d'analyse cause racine pour ce projet.")
        return

    racines = arbre[arbre["est_cause_racine"] == 1].sort_values(
        "confidence", ascending=False
    )
    if racines.empty:
        racines = arbre.sort_values("confidence", ascending=False).head(5)

    st.markdown(
        f"<div style='font-size:12px;color:{COLOR_TEXT_MUTED};"
        f"font-weight:600;text-transform:uppercase;letter-spacing:0.05em;"
        f"margin:8px 0 8px 0;'>Causes proposees a votre validation</div>",
        unsafe_allow_html=True,
    )
    for _, r in racines.iterrows():
        st.markdown(
            f"""
            <div style="padding:10px 14px;background:{COLOR_CARD};
                        border-left:3px solid {COLOR_PRIMARY};
                        border-radius:8px;margin-bottom:6px;">
              <div style="font-size:11px;color:{COLOR_PRIMARY};
                          font-weight:700;">
                {r['branche_m'].upper()}  &middot;  confiance {r['confidence']:.0%}
              </div>
              <div style="font-size:13px;color:{COLOR_TEXT};
                          font-weight:600;margin-top:2px;">
                {r['reponse']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Vue VALIDEUR (Tech N+1, Maintenance, Production)
# ---------------------------------------------------------------------------
def _render_vue_valideur(df_projets, role: str, user_name: str) -> None:
    # Filtre : projets ou l'utilisateur a encore son vote a donner
    mes_projets = []
    for _, projet in df_projets.iterrows():
        state = _validation_state(int(projet["id"]))
        if role in ROLES_TECH and not state["tech_done"]:
            mes_projets.append((projet.to_dict(), state))
        elif role in ROLES_PROD and not state["prod_done"]:
            mes_projets.append((projet.to_dict(), state))

    section(
        f"A valider de votre cote  ({len(mes_projets)})",
        "Chacun valide depuis son poste, quand il peut. Pas de reunion."
    )

    if not mes_projets:
        st.success("Vous etes a jour. Rien a valider pour le moment.")
        return

    # Liste compacte de tous mes projets
    for projet, state in mes_projets:
        _render_projet_card(projet, state)

    st.markdown("---")

    # Choix d'un projet pour valider
    opts = [int(p["id"]) for p, _ in mes_projets]
    id_to_titre = {int(p["id"]): p["titre"] for p, _ in mes_projets}
    target_id = st.selectbox(
        "Projet a examiner",
        options=opts,
        format_func=lambda i: f"#{i} - {id_to_titre.get(i, '')}",
        key="f4_target",
    )

    with st.container(border=True):
        _render_projet_detail(int(target_id))

        st.markdown(
            "<div style='height:8px'></div>", unsafe_allow_html=True
        )
        commentaire = st.text_input(
            "Commentaire (optionnel - obligatoire en cas de refus)",
            key="f4_comm",
        )
        col_v, col_r = st.columns(2)
        with col_v:
            if st.button(
                "Valider",
                type="primary",
                use_container_width=True,
                key="f4_valider",
            ):
                _enregistrer_vote(
                    int(target_id), role, user_name,
                    decision="valide", commentaire=commentaire,
                )
                st.rerun()
        with col_r:
            disabled_refus = (not commentaire.strip())
            if st.button(
                "Refuser",
                use_container_width=True,
                disabled=disabled_refus,
                key="f4_refuser",
                help=("Le commentaire est obligatoire pour un refus."
                      if disabled_refus else "Refus avec commentaire"),
            ):
                _enregistrer_vote(
                    int(target_id), role, user_name,
                    decision="refuse", commentaire=commentaire,
                )
                st.rerun()


def _enregistrer_vote(projet_id: int, role: str, user_name: str,
                      *, decision: str, commentaire: str) -> None:
    """Enregistre la decision + applique transition si les 2 votes sont la.

    Branche aussi la boucle d'apprentissage CBR :
      - Chaque vote ajoute un feedback +1 / -1 sur les causes racines
      - Si projet passe en "valide" : enrichissement (la base apprend)
    """
    db.add_validation(
        projet_id=projet_id,
        role_valideur=role,
        decision=decision,
        nom_valideur=user_name,
        commentaire=commentaire,
    )
    # Feedback CBR : +1 si valide, -1 si refuse (sur les causes racines)
    record_validation_outcome(
        projet_id=projet_id,
        role_valideur=role,
        decision=decision,
        nom_valideur=user_name,
        commentaire=commentaire,
    )
    # Recalculer state et appliquer transition si complet
    state = _validation_state(projet_id)
    result = _apply_transitions_if_complete(projet_id, state)
    if result == "valide":
        # Projet entierement valide : la base apprend pour les futurs cas
        rapport = enrich_from_validated_projet(projet_id)
        st.success(
            f"Les deux votes sont la. Projet valide -> "
            f"{rapport['noeuds_enrichis']} elements appris pour les "
            f"prochains cas similaires."
        )
    elif result == "refuse_back_to_cadrage":
        st.warning(
            "Un refus enregistre. Le projet repart en cadrage pour ajustement."
        )
    else:
        st.success(
            f"Votre vote ({decision}) est enregistre. "
            "On attend l'autre cote pour conclure."
        )


# ---------------------------------------------------------------------------
# Vue OBSERVATEUR (Resp. AC, Direction) - voit l'avancement
# ---------------------------------------------------------------------------
def _render_vue_observateur(df_projets) -> None:
    section(
        f"Avancement des validations  ({len(df_projets)})",
        "Les decisions arrivent en parallele, sans reunion."
    )

    if df_projets.empty:
        st.info("Aucun projet en attente de validation pour le moment.")
        return

    for _, projet in df_projets.iterrows():
        state = _validation_state(int(projet["id"]))
        _render_projet_card(projet.to_dict(), state)

    # Tip a l'observateur
    st.caption(
        "Les valideurs (Tech N+1 et Resp. Production) recoivent le projet "
        "des qu'il est en attente. Vous n'avez rien a faire ici - juste "
        "observer."
    )


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------
def render() -> None:
    role      = st.session_state.get("role", "ac_manager")
    user_name = st.session_state.get("user_name", "")

    if role == "operator":
        st.info(
            "Vous n'avez pas de validation a faire ici. "
            "Cette etape concerne le Tech N+1 et le Resp. Production."
        )
        return

    df = db.list_projets_ac(statut="validation_en_cours")

    if role in ROLES_TECH or role in ROLES_PROD:
        _render_vue_valideur(df, role, user_name)
    else:
        # ac_manager / ceo / autres : vue observateur
        _render_vue_observateur(df)
