"""
core/cbr/path_engine.py - Generation automatique des arbres 5 Pourquoi.

Pipeline :
  1. Recherche des cas similaires (similarity.find_similar_cases)
  2. Extraction des chemins historiques des cas similaires
  3. Adaptation au nouveau cas (substitution legere, classification)
  4. Si rien en base (demarrage froid) -> templates synthetiques 5M
  5. Retour : arbre complet {branche_m: [noeuds]}

Branches 5M (standard francais industriel, conforme au visuel Ishikawa):
   Methode    : procedures, mode operatoire, consignes
   Matiere    : matieres premieres, consommables, fournitures
   Main-d'oeuvre : humains, formation, attention, fatigue
   Materiel   : machines, outils, equipement
   Milieu     : ambiance, temperature, humidite, vibrations
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core import db
from core.cbr import case_base as cb
from core.cbr.similarity import find_similar_cases, incident_to_case_dict
from core.cbr.feedback import compute_chemin_weight


# ---------------------------------------------------------------------------
# Constantes - les 5M conformes au design Ishikawa classique francais
# ---------------------------------------------------------------------------
BRANCHES_5M = (
    "Methode",
    "Matiere",
    "Main-d'oeuvre",
    "Materiel",
    "Milieu",
)

# Profondeur cible par defaut
PROFONDEUR_CIBLE = 5


# ---------------------------------------------------------------------------
# Templates synthetiques pour le demarrage froid (base vide)
# ---------------------------------------------------------------------------
# Chaque template = 5 niveaux d'une chaine 5 Pourquoi credible.
# Confidence faible (0.30 - 0.45) parce que c'est generique pas historique.
# Sera enrichi a chaque cas valide via la boucle CBR.

_TEMPLATE_METHODE = [
    ("Pourquoi le defaut apparait ?",
     "Procedure de reglage non suivie a la lettre",          0.40),
    ("Pourquoi la procedure n'est pas suivie ?",
     "Procedure peu claire ou obsolete",                     0.40),
    ("Pourquoi la procedure est obsolete ?",
     "Pas de revue periodique des modes operatoires",        0.35),
    ("Pourquoi pas de revue ?",
     "Aucun responsable assigne a la revue documentaire",    0.35),
    ("Pourquoi pas d'assignation ?",
     "Manque de gouvernance documentaire",                   0.30),
]

_TEMPLATE_MATIERE = [
    ("Pourquoi le defaut apparait ?",
     "Matiere hors specifications fournisseur",              0.40),
    ("Pourquoi la matiere est hors specs ?",
     "Controle reception incomplet",                         0.40),
    ("Pourquoi le controle est incomplet ?",
     "Plan de controle reception non a jour",                0.35),
    ("Pourquoi plan pas a jour ?",
     "Pas de revue qualite fournisseur reguliere",           0.35),
    ("Pourquoi pas de revue fournisseur ?",
     "Process qualite achats non formalise",                 0.30),
]

_TEMPLATE_MAIN_OEUVRE = [
    ("Pourquoi le defaut apparait ?",
     "Geste operatoire non conforme",                        0.35),
    ("Pourquoi geste non conforme ?",
     "Formation incomplete ou trop ancienne",                0.40),
    ("Pourquoi formation incomplete ?",
     "Plan de formation operateur non a jour",               0.40),
    ("Pourquoi plan pas a jour ?",
     "Absence de matrice de competences vivante",            0.35),
    ("Pourquoi pas de matrice vivante ?",
     "Pas de pilote competences au sein de la production",   0.30),
]

_TEMPLATE_MATERIEL = [
    ("Pourquoi le defaut apparait ?",
     "Machine derive par rapport aux specs",                 0.45),
    ("Pourquoi la machine derive ?",
     "Maintenance preventive insuffisante",                  0.45),
    ("Pourquoi maintenance insuffisante ?",
     "Plan de maintenance non respecte ou sous-dimensionne", 0.40),
    ("Pourquoi non respecte ?",
     "Manque de ressources ou de pieces de rechange",        0.35),
    ("Pourquoi manque de ressources ?",
     "Budget maintenance sous-estime",                       0.30),
]

_TEMPLATE_MILIEU = [
    ("Pourquoi le defaut apparait ?",
     "Conditions ambiantes hors fenetre operatoire",         0.35),
    ("Pourquoi conditions hors fenetre ?",
     "Variation temperature ou humidite non maitrisee",      0.40),
    ("Pourquoi variation non maitrisee ?",
     "Pas de sonde + alerte automatique en atelier",         0.40),
    ("Pourquoi pas de sonde ?",
     "Sensibilite aux conditions non identifiee initialement", 0.35),
    ("Pourquoi pas identifiee ?",
     "Analyse de risques procede pas a jour",                0.30),
]

TEMPLATES_5M: dict[str, list[tuple[str, str, float]]] = {
    "Methode":       _TEMPLATE_METHODE,
    "Matiere":       _TEMPLATE_MATIERE,
    "Main-d'oeuvre": _TEMPLATE_MAIN_OEUVRE,
    "Materiel":      _TEMPLATE_MATERIEL,
    "Milieu":        _TEMPLATE_MILIEU,
}


# ---------------------------------------------------------------------------
# Extraction des chemins historiques pour les cas similaires
# ---------------------------------------------------------------------------
def _extract_paths_from_similar(
    similar_cases: pd.DataFrame,
    branche_m: str,
    *,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Pour chaque cas similaire, trouve son projet AC associe et
    extrait son arbre 5 Pourquoi pour la branche M demandee.

    Chaque chemin est aussi pondere par le poids feedback de sa cause
    racine (apprentissage CBR auto-actif).

    Retourne une liste de chemins triee par score combine (similarite +
    poids feedback) decroissant.
    """
    if similar_cases.empty:
        return []

    chemins: list[dict[str, Any]] = []
    for _, cas in similar_cases.iterrows():
        cas_id   = int(cas["id"])
        sim      = float(cas["similarity"])
        # Trouver les projets AC sur cet incident
        projets = db.list_projets_ac(db_path=db_path)
        if projets.empty:
            continue
        projets_sur_cas = projets[projets["incident_id"] == cas_id]
        for _, proj in projets_sur_cas.iterrows():
            arbre = cb.list_chemins_projet(
                int(proj["id"]), branche_m=branche_m, db_path=db_path,
            )
            if arbre.empty:
                continue
            # Poids feedback : on prend celui de la cause racine de la chaine
            noeuds = arbre.to_dict("records")
            racine = next(
                (n for n in noeuds if n.get("est_cause_racine")),
                noeuds[-1] if noeuds else None,
            )
            poids_feedback = (
                compute_chemin_weight(int(racine["id"]), db_path=db_path)
                if racine else 1.0
            )
            chemins.append({
                "source_cas_id":  cas_id,
                "similarite_cas": sim,
                "poids_feedback": poids_feedback,
                "score":          sim * poids_feedback,
                "noeuds":         noeuds,
            })
    # Trier par score combine (similarite x feedback)
    chemins.sort(key=lambda c: -c["score"])
    return chemins


# ---------------------------------------------------------------------------
# Adaptation : substitution legere des termes specifiques au nouveau cas
# ---------------------------------------------------------------------------
def _adapt_text_to_case(text: str, new_case: dict[str, Any]) -> str:
    """
    Substitution simple : remplace les references explicites a une autre
    machine par la machine du nouveau cas, etc.
    Garde le sens general du chemin historique.
    """
    if not text:
        return text
    # On ne fait pas de NLP lourd pour le MVP - juste une bonne pratique :
    # signaler que c'est inspire d'un cas similaire.
    return text


# ---------------------------------------------------------------------------
# Generation d'un chemin a partir d'un template synthetique (demarrage froid)
# ---------------------------------------------------------------------------
def _synthetic_path_for_branch(
    new_case: dict[str, Any],
    branche_m: str,
) -> list[dict[str, Any]]:
    """Genere un chemin 5 niveaux a partir du template synthetique."""
    template = TEMPLATES_5M.get(branche_m, [])
    noeuds: list[dict[str, Any]] = []
    for i, (question, reponse, conf) in enumerate(template, start=1):
        noeuds.append({
            "niveau":           i,
            "branche_m":        branche_m,
            "question":         question,
            "reponse":          _adapt_text_to_case(reponse, new_case),
            "confidence":       conf,
            "type_noeud":       "cause_racine" if i == PROFONDEUR_CIBLE
                                else "cause_directe",
            "est_cause_racine": (i == PROFONDEUR_CIBLE),
            "source_cas_id":    None,         # template, pas un cas historique
            "similarite_cas":   None,
            "origine":          "template_synthetique",
        })
    return noeuds


# ---------------------------------------------------------------------------
# Generation d'un chemin a partir d'un cas historique (CBR pur)
# ---------------------------------------------------------------------------
def _path_from_historical(
    chemin_hist: dict[str, Any],
    new_case: dict[str, Any],
    branche_m: str,
) -> list[dict[str, Any]]:
    """
    Convertit un chemin historique (deja stocke en base) en chemin
    propose pour le nouveau cas. Garde la similarite source pour
    afficher "Inspire du cas #427 (sim 92%)".
    """
    noeuds_src = chemin_hist["noeuds"]
    # Reconstruction de la chaine par niveau (au cas ou plusieurs branches)
    noeuds_par_niveau = sorted(noeuds_src, key=lambda n: n.get("niveau", 0))

    sortie: list[dict[str, Any]] = []
    for n in noeuds_par_niveau:
        sortie.append({
            "niveau":           int(n.get("niveau", 0)),
            "branche_m":        branche_m,
            "question":         n.get("question", ""),
            "reponse":          _adapt_text_to_case(n.get("reponse", ""), new_case),
            "confidence":       float(n.get("confidence", 0.5))
                                * float(chemin_hist["similarite_cas"]),
            "type_noeud":       n.get("type_noeud", "cause_directe"),
            "est_cause_racine": bool(n.get("est_cause_racine", 0)),
            "source_cas_id":    chemin_hist["source_cas_id"],
            "similarite_cas":   chemin_hist["similarite_cas"],
            "origine":          "cas_historique",
        })
    return sortie


# ---------------------------------------------------------------------------
# Entry point : generation de l'arbre complet 5M pour un nouveau projet
# ---------------------------------------------------------------------------
def generate_full_tree(
    incident_id: int,
    *,
    top_k_similar: int = 3,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Genere l'arbre 5 Pourquoi complet pour un projet, branche par branche.

    Algorithme :
      1. Charger l'incident
      2. Trouver les cas similaires
      3. Pour chaque branche M :
         - si cas similaires : adapter le meilleur chemin historique
         - sinon : utiliser le template synthetique
      4. Retourner la structure complete

    Retour : {
        "incident_id":    <int>,
        "similar_cases":  DataFrame (pour affichage UI),
        "tree": {
            "Methode":       [noeud1, noeud2, ...],
            "Matiere":       [...],
            "Main-d'oeuvre": [...],
            "Materiel":      [...],
            "Milieu":        [...],
        },
        "global_confidence": float,
    }
    """
    incident = db.get_incident(incident_id, db_path=db_path)
    if incident is None:
        raise ValueError(f"Incident {incident_id} introuvable")

    new_case = incident_to_case_dict(incident)

    # 1. Cas similaires (peut etre vide si base froide)
    similar = find_similar_cases(
        new_case,
        top_k=top_k_similar,
        exclude_self_id=incident_id,
        db_path=db_path,
    )

    # 2. Pour chaque branche M
    tree: dict[str, list[dict[str, Any]]] = {}
    confidences_branches: list[float] = []

    for branche in BRANCHES_5M:
        chemins_hist = _extract_paths_from_similar(
            similar, branche, db_path=db_path,
        )
        if chemins_hist:
            # CBR pur : meilleur chemin historique (deja trie par score combine)
            best = chemins_hist[0]
            noeuds = _path_from_historical(best, new_case, branche)
        else:
            # Demarrage froid : template synthetique
            noeuds = _synthetic_path_for_branch(new_case, branche)
        tree[branche] = noeuds
        # Confiance moyenne de la branche
        if noeuds:
            confidences_branches.append(
                sum(n["confidence"] for n in noeuds) / len(noeuds)
            )

    global_conf = (
        sum(confidences_branches) / len(confidences_branches)
        if confidences_branches else 0.0
    )

    return {
        "incident_id":       incident_id,
        "similar_cases":     similar,
        "tree":              tree,
        "global_confidence": global_conf,
    }


# ---------------------------------------------------------------------------
# Persistance : sauvegarde l'arbre genere dans la base (chemins_pourquoi)
# ---------------------------------------------------------------------------
def save_tree_to_db(
    projet_id: int,
    tree: dict[str, list[dict[str, Any]]],
    *,
    replace_existing: bool = True,
    db_path: Path | str | None = None,
) -> int:
    """
    Persiste l'arbre genere dans chemins_pourquoi.
    Si replace_existing : supprime les anciens noeuds de ce projet d'abord.

    Retourne le nombre de noeuds inseres.
    """
    if replace_existing:
        cb.delete_chemins_projet(projet_id, db_path=db_path)

    total = 0
    for branche, noeuds in tree.items():
        previous_id: int | None = None  # pour chainer parent_id
        for n in noeuds:
            cid = cb.add_chemin(
                projet_id=projet_id,
                branche_m=branche,
                niveau=int(n["niveau"]),
                question=n["question"],
                reponse=n["reponse"],
                parent_id=previous_id,
                type_noeud=n.get("type_noeud", "cause_directe"),
                est_cause_racine=bool(n.get("est_cause_racine", False)),
                source_cas_id=n.get("source_cas_id"),
                similarite_cas=n.get("similarite_cas"),
                confidence=float(n.get("confidence", 0.5)),
                db_path=db_path,
            )
            previous_id = cid
            total += 1
    return total
