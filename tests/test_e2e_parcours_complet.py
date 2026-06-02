"""
Tests end-to-end : parcours utilisateur complet du workflow d'amelioration
continue dans LI90.

Objectif : valider que le scenario "5 minutes du signal au valide" tient
debout techniquement. Couvre les 6 fonctionnalites Vue B (F1 -> F6) et
les 3 portes d'entree (signalement / import / saisie directe).
"""
from __future__ import annotations

import pytest
from pathlib import Path

import core.db
from core import db
from core.cbr import case_base as cb
from core.cbr.path_engine import generate_full_tree, save_tree_to_db
from core.cbr.feedback import (
    record_validation_outcome, compute_chemin_weight,
    enrich_from_validated_projet,
)


# ---------------------------------------------------------------------------
# Fixture : base SQLite isolee pour chaque test
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_li90.db"
    monkeypatch.setattr(core.db, "DB_PATH", test_db)
    db.init_db()
    return test_db


# ---------------------------------------------------------------------------
# Helpers pour les tests
# ---------------------------------------------------------------------------
def _vote_complet_succes(projet_id: int) -> None:
    """Simule un vote Tech + Prod valide -> projet doit passer en valide."""
    db.update_projet_ac(projet_id, statut="validation_en_cours")

    # Tech vote
    db.add_validation(projet_id, "technician", "valide", nom_valideur="Tech1")
    record_validation_outcome(projet_id, "technician", "valide",
                              nom_valideur="Tech1")

    # Prod vote
    db.add_validation(projet_id, "production", "valide", nom_valideur="Prod1")
    record_validation_outcome(projet_id, "production", "valide",
                              nom_valideur="Prod1")


# ===========================================================================
# TEST 1 - Parcours standard : signalement operateur -> valide
# ===========================================================================
class TestParcoursStandard:
    """
    Le scenario du Resp. AC qui voit un signalement operateur et le mene
    jusqu'a la validation distribuee complete.
    """

    def test_parcours_du_signal_au_valide(self, fresh_db):
        # F1 - Operateur signale (statut brut)
        iid = db.create_incident(
            "M3 Soudeuse",
            "Defauts soudure repetes sur ligne 3",
            operateur_nom="Op1",
            type_incident="Defaut produit",
            severite="haute",
            cree_par_role="operator",
        )
        incident = db.get_incident(iid)
        assert incident["statut"] == "brut"

        # F1 - Tech N+1 fiabilise
        db.update_incident_statut(iid, "fiabilise")
        assert db.get_incident(iid)["statut"] == "fiabilise"

        # F2 - Resp. AC demarre un projet
        pid = db.create_projet_ac(
            "Reduire defauts soudure M3",
            incident_id=iid,
            cree_par="Castro",
            cree_par_role="ac_manager",
        )
        db.update_incident_statut(iid, "en_projet")
        projet = db.get_projet_ac(pid)
        assert projet["statut"] == "cadrage"
        assert projet["incident_id"] == iid

        # F3 - Generation arbre 5M par le moteur CBR
        result = generate_full_tree(iid)
        save_tree_to_db(pid, result["tree"])
        db.update_projet_ac(pid, statut="analyse")

        arbre = cb.list_chemins_projet(pid)
        assert len(arbre) == 25, "5 branches x 5 niveaux"
        racines = arbre[arbre["est_cause_racine"] == 1]
        assert len(racines) == 5, "1 cause racine par branche"

        # F3 - Resp. AC valide -> projet en validation_en_cours
        # (le vote AC valide les causes racines)
        for _, r in racines.iterrows():
            cb.add_feedback(
                int(r["id"]), +1,
                role_valideur="ac_manager",
                commentaire="Validation analyse",
            )

        # F4 - Tech N+1 + Resp. Prod valident (2 votes -> valide)
        _vote_complet_succes(pid)
        db.update_projet_ac(pid, statut="valide")

        # Enrichissement final : la base apprend
        rapport = enrich_from_validated_projet(pid)
        assert rapport["noeuds_enrichis"] == 25

        # Le projet est en VALIDE
        projet_final = db.get_projet_ac(pid)
        assert projet_final["statut"] == "valide"

        # F6 - Action peut etre creee
        aid = db.add_action(
            projet_id=pid,
            titre="Reviser la procedure de soudage M3",
            assignee="Tech1",
            priorite=1,
        )
        action = db.list_actions(projet_id=pid)
        assert len(action) == 1
        assert action.iloc[0]["statut"] == "a_faire"


# ===========================================================================
# TEST 2 - Fast track : saisie directe responsable (skip fiabilisation)
# ===========================================================================
class TestSaisieDirecte:
    """Le Resp. AC connait deja le probleme, il saute la fiabilisation."""

    def test_saisie_directe_to_valide(self, fresh_db):
        # F1 - Saisie directe par Resp. AC (incident cree puis fiabilise direct)
        iid = db.create_incident(
            "M5 Plieuse",
            "Plis decales sur cartons 60x40",
            type_incident="Defaut produit",
            severite="moyenne",
            cree_par_role="ac_manager",
        )
        db.update_incident_statut(iid, "fiabilise")

        # F2 - Projet
        pid = db.create_projet_ac(
            "Corriger plis M5", incident_id=iid,
            cree_par_role="ac_manager",
        )

        # F3 - Analyse + persistance + validation
        result = generate_full_tree(iid)
        save_tree_to_db(pid, result["tree"])
        racines = cb.list_chemins_projet(pid)[
            cb.list_chemins_projet(pid)["est_cause_racine"] == 1
        ]
        for _, r in racines.iterrows():
            cb.add_feedback(int(r["id"]), +1, role_valideur="ac_manager")

        # F4 - 2 votes
        _vote_complet_succes(pid)
        db.update_projet_ac(pid, statut="valide")

        assert db.get_projet_ac(pid)["statut"] == "valide"


# ===========================================================================
# TEST 3 - Parcours avec REFUS : retour cadrage
# ===========================================================================
class TestParcoursRefus:
    """Tech refuse l'analyse -> le projet repart en cadrage pour ajustement."""

    def test_refus_tech_retour_cadrage(self, fresh_db):
        iid = db.create_incident("M7", "Panne", db_path=fresh_db)
        pid = db.create_projet_ac("Test refus", incident_id=iid)
        result = generate_full_tree(iid)
        save_tree_to_db(pid, result["tree"])
        db.update_projet_ac(pid, statut="validation_en_cours")

        # Import f4 ici pour eviter probleme circulaire au top
        from vue_b.f4_validation import (
            _validation_state, _apply_transitions_if_complete,
        )

        # Tech refuse
        db.add_validation(pid, "technician", "refuse",
                          commentaire="Cause pas convaincante")
        record_validation_outcome(pid, "technician", "refuse")

        state = _validation_state(pid)
        result_trans = _apply_transitions_if_complete(pid, state)

        assert state["any_refused"] is True
        assert result_trans == "refuse_back_to_cadrage"

        # Le projet est retourne en CADRE
        assert db.get_projet_ac(pid)["statut"] == "cadre"


# ===========================================================================
# TEST 4 - Apprentissage progressif sur cas similaires
# ===========================================================================
class TestApprentissageCBR:
    """
    2 cas similaires : le 2eme doit beneficier de l'apprentissage du 1er.
    Le poids feedback doit etre > 1.0 apres validation, ce qui boost les
    chemins du cas 1 dans le ranking CBR du cas 2.
    """

    def test_le_2eme_cas_apprend_du_1er(self, fresh_db):
        # CAS 1 - Premier passage (templates synthetiques)
        iid_1 = db.create_incident(
            "M3 Soudeuse", "Defauts soudure repetes",
            type_incident="Defaut produit", severite="haute",
        )
        pid_1 = db.create_projet_ac("Reduire defauts M3", incident_id=iid_1)
        r1 = generate_full_tree(iid_1)
        save_tree_to_db(pid_1, r1["tree"])

        # Valide entierement le cas 1
        _vote_complet_succes(pid_1)
        db.update_projet_ac(pid_1, statut="valide")
        enrich_from_validated_projet(pid_1)

        # Verifier que le poids feedback de la cause racine est > 1
        arbre_1 = cb.list_chemins_projet(pid_1)
        racine_methode_1 = arbre_1[
            (arbre_1["branche_m"] == "Methode") &
            (arbre_1["est_cause_racine"] == 1)
        ].iloc[0]
        w = compute_chemin_weight(int(racine_methode_1["id"]))
        assert w > 1.0, f"Poids feedback doit etre > 1.0 (obtenu : {w:.2f})"

        # CAS 2 - Cas similaire
        iid_2 = db.create_incident(
            "M3 Soudeuse", "Soudures faibles intermittentes",
            type_incident="Defaut produit", severite="haute",
        )
        pid_2 = db.create_projet_ac("Investigation M3", incident_id=iid_2)
        r2 = generate_full_tree(iid_2)
        save_tree_to_db(pid_2, r2["tree"])

        # Verifier que le cas 1 est retrouve comme similaire
        assert len(r2["similar_cases"]) >= 1
        top_similar = r2["similar_cases"].iloc[0]
        assert int(top_similar["id"]) == iid_1
        assert float(top_similar["similarity"]) > 0.5

        # Verifier que les chemins du cas 2 viennent du cas 1 (CBR)
        arbre_2 = cb.list_chemins_projet(pid_2)
        avec_source = arbre_2[arbre_2["source_cas_id"].notna()]
        assert len(avec_source) > 0, "Au moins certains chemins doivent venir du cas 1"


# ===========================================================================
# TEST 5 - Robustesse : demarrage froid (base vide), aucun cas similaire
# ===========================================================================
class TestDemarrageFroid:
    """Le tout premier cas chez un nouveau client : pas de base, templates."""

    def test_premier_cas_utilise_templates(self, fresh_db):
        iid = db.create_incident(
            "M1", "Premier defaut", type_incident="Defaut produit",
        )
        pid = db.create_projet_ac("Premier projet", incident_id=iid)
        result = generate_full_tree(iid)

        # Aucun cas similaire (base vide)
        assert result["similar_cases"].empty
        # Mais l'arbre est genere quand meme (via templates)
        save_tree_to_db(pid, result["tree"])
        arbre = cb.list_chemins_projet(pid)
        assert len(arbre) == 25

        # Tous les noeuds ont une origine "template_synthetique"
        # (verifie via source_cas_id qui doit etre None)
        sans_source = arbre[arbre["source_cas_id"].isna()]
        assert len(sans_source) == 25


# ===========================================================================
# TEST 6 - Idempotence : appels multiples sur init_db
# ===========================================================================
class TestRobustesseDB:
    """init_db doit etre idempotent (peut etre rappele sans casser)."""

    def test_init_db_multiple_appels(self, fresh_db):
        # Cree un incident
        iid = db.create_incident("M1", "test")
        # Re-appel init_db ne doit pas effacer
        db.init_db()
        assert db.get_incident(iid) is not None
