"""
vue_b/synthese.py - Ecran "Gains & livrables" affiche a la fin de chaque
etape F1 -> F6.

Objectif demo/POC : en 10 secondes, un responsable comprend la valeur de
l'etape — objectif chiffre, mecanismes qui garantissent le gain, livrables
generes. Aligne sur les ecrans de synthese du script video Safran.

Usage dans chaque page :
    from vue_b.synthese import render_synthese
    render_synthese("f3")
"""
from __future__ import annotations

import streamlit as st

from ui_theme import (
    COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_OK,
    COLOR_CARD, COLOR_BORDER,
)


# ---------------------------------------------------------------------------
# Contenu par etape (source : script video Safran v5)
# ---------------------------------------------------------------------------
SYNTHESES: dict[str, dict] = {
    "f1": {
        "titre": "Collecte terrain",
        "objectif_avant": "3 semaines",
        "objectif_apres": "30 minutes",
        "mecanismes": [
            "Signalement operateur en 30 s depuis le poste",
            "Requete automatique des donnees machine liees au defaut",
            "Fiabilisation par le Tech N+1 (celui qui connait la ligne)",
            "Une info saisie UNE fois — reutilisee a toutes les etapes",
        ],
        "livrables": [
            "Incident horodate et fiabilise",
            "Donnees rattachees au defaut",
            "Tracabilite complete (qui, quoi, quand)",
        ],
    },
    "f2": {
        "titre": "Cadrage & containment",
        "objectif_avant": "45 min de cadrage + 1/2 journee de containment",
        "objectif_apres": "10 min + 30 min",
        "mecanismes": [
            "Catalogue de defauts normalise (fini les doubles vocabulaires)",
            "QQOQCCP pre-rempli depuis l'incident — on ne complete que "
            "le specifique",
            "Action de protection creee dans le meme geste (priorite 1)",
            "Un probleme bien cadre = un Ishikawa qui vise juste",
        ],
        "livrables": [
            "QQOQCCP complet et homogene",
            "Plan de containment actif",
            "Projet AC trace, pret pour l'analyse",
        ],
    },
    "f3": {
        "titre": "Analyse de cause racine",
        "objectif_avant": "3 semaines",
        "objectif_apres": "3 heures",
        "mecanismes": [
            "Memoire de l'usine : cas similaires retrouves en 1 seconde",
            "Ishikawa 5M pre-rempli par la bibliotheque sectorielle",
            "5 Pourquoi forces jusqu'a une cause actionnable (anti-blame)",
            "Priorisation par confiance — pas a l'instinct",
            "Correction humaine possible : l'IA propose, vous decidez, "
            "et votre correction entraine le modele (+1/-1)",
        ],
        "livrables": [
            "Arbre 5M complet (5 branches x 5 niveaux)",
            "Top causes scorees et expliquees",
            "Audit trail de l'analyse",
        ],
    },
    "f4": {
        "titre": "Alignement des parties prenantes",
        "objectif_avant": "1 semaine de reunions",
        "objectif_apres": "5 minutes, zero reunion",
        "mecanismes": [
            "Diffusion instantanee de l'analyse aux valideurs",
            "Chacun vote depuis son poste, quand il peut (2 min)",
            "2 votes +1 -> valide automatiquement",
            "Refus = commentaire obligatoire + retour cadrage — "
            "le desaccord est trace, jamais enterre",
        ],
        "livrables": [
            "Cause racine validee multi-roles",
            "Rapport de validation horodate",
            "Audit trail (qui a valide quoi, quand)",
        ],
    },
    "f5": {
        "titre": "Solution & adoption terrain",
        "objectif_avant": "3 semaines de resistance",
        "objectif_apres": "3 jours d'adoption",
        "mecanismes": [
            "La meme solution traduite dans le langage de chaque role",
            "Cascade de conviction : Resp. Prod valide -> Tech N+1 "
            "(celui qui sait convaincre les operateurs) porte la consigne",
            "Consigne operateur courte et visuelle — pas un rapport",
            "Garantie Terrain : 25 conditions d'adoption verifiees "
            "avant diffusion (module livre pour le POC)",
        ],
        "livrables": [
            "Fiches solution par role (Prod / Tech / Operateur)",
            "Solution chiffree : cout, delai, gain",
            "Plan d'adoption",
        ],
    },
    "f6": {
        "titre": "Suivi & capitalisation",
        "objectif_avant": "Suivi aleatoire",
        "objectif_apres": "Systematique et mesure",
        "mecanismes": [
            "UNE action a la fois — sinon rien n'est isolable",
            "Priorite, assignee, statut : personne n'oublie",
            "ROI reel mesure a la cloture (pas estime)",
            "Chaque cas clos enrichit la base : le prochain defaut "
            "similaire sera resolu plus vite",
        ],
        "livrables": [
            "Plan d'action date et assigne",
            "ROI mesure",
            "Cas capitalise dans la base de connaissances",
        ],
    },
}


def render_synthese(step: str) -> None:
    """Affiche le bloc 'Gains & livrables' de l'etape (bouton depliable)."""
    data = SYNTHESES.get(step)
    if not data:
        return

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander(
        f"📊  Gains & livrables — {data['titre']}", expanded=False,
    ):
        # Bandeau objectif
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:14px;
                        background:linear-gradient(135deg,
                        rgba(249,115,22,0.12), rgba(249,115,22,0.03));
                        border:1px solid {COLOR_PRIMARY};border-radius:12px;
                        padding:14px 18px;margin-bottom:14px;">
              <div style="font-size:12px;color:{COLOR_TEXT_MUTED};">
                Aujourd'hui<br>
                <span style="font-size:15px;color:{COLOR_TEXT};
                             font-weight:700;text-decoration:line-through;
                             text-decoration-color:{COLOR_PRIMARY};">
                  {data['objectif_avant']}</span>
              </div>
              <div style="font-size:22px;color:{COLOR_PRIMARY};
                          font-weight:800;">→</div>
              <div style="font-size:12px;color:{COLOR_TEXT_MUTED};">
                Avec LI90<br>
                <span style="font-size:17px;color:{COLOR_OK};
                             font-weight:800;">{data['objectif_apres']}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_m, col_l = st.columns([1.3, 1])
        with col_m:
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;"
                f"color:{COLOR_PRIMARY};text-transform:uppercase;"
                f"letter-spacing:0.05em;margin-bottom:6px;'>"
                f"Ce qui garantit ce gain</div>",
                unsafe_allow_html=True,
            )
            for m in data["mecanismes"]:
                st.markdown(
                    f"<div style='display:flex;gap:8px;font-size:12.5px;"
                    f"color:{COLOR_TEXT};margin-bottom:5px;line-height:1.45;'>"
                    f"<span style='color:{COLOR_OK};font-weight:700;'>✓</span>"
                    f"<span>{m}</span></div>",
                    unsafe_allow_html=True,
                )
        with col_l:
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;"
                f"color:{COLOR_PRIMARY};text-transform:uppercase;"
                f"letter-spacing:0.05em;margin-bottom:6px;'>"
                f"Livrables generes</div>",
                unsafe_allow_html=True,
            )
            for l in data["livrables"]:
                st.markdown(
                    f"<div style='background:{COLOR_CARD};"
                    f"border:1px solid {COLOR_BORDER};border-radius:8px;"
                    f"padding:7px 11px;margin-bottom:5px;font-size:12px;"
                    f"color:{COLOR_TEXT};'>&#8226; {l}</div>",
                    unsafe_allow_html=True,
                )
