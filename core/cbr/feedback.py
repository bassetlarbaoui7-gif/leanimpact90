"""
core/cbr/feedback.py - Apprentissage automatique de la base CBR.

A chaque validation humaine, ce module :
  1. Trace un feedback (+1 / -1) sur les noeuds concernes
  2. Met a jour le poids de chaque chemin (bonus si valide, penalite si refuse)
  3. Marque le projet comme "cas de reference" reutilisable pour le CBR

Aucun re-training de modele ML. Le savoir-faire vit dans la base
SQLite. Plus la base grandit, plus le moteur est precis.
"""
from __future__ import annotations

import math
from pathlib import Path

from core import db
from core.cbr import case_base as cb


# ---------------------------------------------------------------------------
# Tracage du feedback apres une validation
# ---------------------------------------------------------------------------
def record_validation_outcome(
    projet_id: int,
    role_valideur: str,
    decision: str,
    *,
    nom_valideur: str = "",
    commentaire: str = "",
    db_path: Path | str | None = None,
) -> int:
    """
    A appeler depuis F4 (validation distribuee) quand un valideur vote.
    Trace un feedback (+1 si valide, -1 si refus) sur CHAQUE cause racine
    du projet. Le moteur CBR utilisera ces poids pour ranker les chemins
    historiques lors d'un nouveau cas similaire.

    Retourne le nombre de feedbacks ajoutes.
    """
    if decision not in ("valide", "refuse"):
        return 0
    sign = +1 if decision == "valide" else -1

    arbre = cb.list_chemins_projet(projet_id, db_path=db_path)
    if arbre.empty:
        return 0
    racines = arbre[arbre["est_cause_racine"] == 1]

    n_added = 0
    for _, r in racines.iterrows():
        cb.add_feedback(
            chemin_id=int(r["id"]),
            decision=sign,
            commentaire=commentaire or f"Vote {decision} de {role_valideur}",
            valide_par=nom_valideur,
            role_valideur=role_valideur,
            db_path=db_path,
        )
        n_added += 1
    return n_added


# ---------------------------------------------------------------------------
# Calcul du poids d'un chemin (utilise par le ranking CBR)
# ---------------------------------------------------------------------------
def compute_chemin_weight(
    chemin_id: int,
    *,
    db_path: Path | str | None = None,
) -> float:
    """
    Poids global d'un chemin :
      1.0 si jamais valide
      > 1.0 si valide plusieurs fois (bonus log-scale)
      < 1.0 si refuse (penalite, plancher 0.1)

    Formule : 1 + 0.3 * log(1 + n_valid) - 0.5 * n_refus
    """
    score = cb.get_feedback_score(chemin_id, db_path=db_path)
    bonus = 0.3 * math.log(1 + score["valid"])
    malus = 0.5 * score["refus"]
    weight = 1.0 + bonus - malus
    return max(0.1, weight)


# ---------------------------------------------------------------------------
# Enrichissement final : le projet devient cas de reference
# ---------------------------------------------------------------------------
def enrich_from_validated_projet(
    projet_id: int,
    *,
    db_path: Path | str | None = None,
) -> dict:
    """
    A appeler quand un projet passe au statut 'valide' (les 2 votes OK).
    - Augmente la confidence de chaque noeud du projet d'un facteur
      proportionnel au feedback recu (jusqu'a 95% max).
    - Marque l'incident lie comme "fiable" (sera privilegie au CBR).
    - Retourne un rapport de l'enrichissement.

    L'idee : un projet entierement valide devient un "trésor" pour les
    futurs cas similaires.
    """
    rapport = {
        "projet_id": projet_id,
        "noeuds_enrichis": 0,
        "incidents_marques": 0,
    }

    # 1. Booster la confiance de chaque noeud du projet
    arbre = cb.list_chemins_projet(projet_id, db_path=db_path)
    for _, n in arbre.iterrows():
        chemin_id = int(n["id"])
        w = compute_chemin_weight(chemin_id, db_path=db_path)
        # Nouvelle confidence : ancienne × w, plafonnee a 0.95
        nouvelle = min(0.95, float(n["confidence"]) * w)
        cb.update_chemin(
            chemin_id,
            confidence=nouvelle,
            db_path=db_path,
        )
        rapport["noeuds_enrichis"] += 1

    # 2. Marquer l'incident lie pour qu'il soit privilégie au CBR
    proj = db.get_projet_ac(projet_id, db_path=db_path)
    if proj and proj.get("incident_id"):
        # Convention : statut 'clos' sur l'incident signale qu'il est devenu
        # un cas de reference (ne ressort plus dans les listes a traiter).
        # Le CBR continue de l'utiliser pour la similarite (toutes les
        # entrees servent au CBR).
        db.update_incident_statut(
            int(proj["incident_id"]), "clos", db_path=db_path,
        )
        rapport["incidents_marques"] = 1
    return rapport
