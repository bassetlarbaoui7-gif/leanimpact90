"""
Tests de preparation de la demo video Safran.

Couvre les ajouts F2 (QQOQCCP + containment), F3 (correction de noeud),
F5 (solution chiffree + fiches par role) et le composant synthese.
Tous les tests sont au niveau DB / logique — pas de rendu Streamlit.

Lancer :  python -m pytest tests/test_demo_video_readiness.py -q
"""
from __future__ import annotations

import pytest

import core.db
from core import db
from core.cbr import case_base as cb
from core.cbr.path_engine import generate_full_tree, save_tree_to_db


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_li90.db"
    monkeypatch.setattr(core.db, "DB_PATH", test_db)
    db.init_db()
    return test_db


def _incident_et_projet() -> tuple[int, int]:
    iid = db.create_incident(
        "Ligne CMS 2 - Four de refusion",
        "AOI : 6 cartes calculateur rejetees, joints brases froids.",
        operateur_nom="Op1",
        type_incident="Defaut produit",
        severite="haute",
        cree_par_role="operator",
    )
    db.update_incident_statut(iid, "fiabilise")
    pid = db.create_projet_ac(
        "Reduire joints brases froids — CMS 2",
        incident_id=iid,
        cree_par="Castro",
        cree_par_role="ac_manager",
    )
    return iid, pid


# ===========================================================================
# F2 — QQOQCCP + containment
# ===========================================================================
class TestF2CadrageDemo:
    def test_qqoqccp_sauvegarde_et_relu(self, fresh_db):
        _, pid = _incident_et_projet()
        ok = db.update_projet_ac(
            pid,
            qqoqcp_quoi="Joints brases froids — Ampleur : 0,3% -> 3%",
            qqoqcp_ou="Ligne CMS 2, four refusion zone 6",
            qqoqcp_quand="Depuis le 30/06, aggravation continue",
            qqoqcp_qui="3 equipes touchees, nuit davantage",
            qqoqcp_comment="Detection AOI",
            qqoqcp_pourquoi="Risque livraison + cout retouche",
        )
        assert ok
        projet = db.get_projet_ac(pid)
        assert "0,3% -> 3%" in projet["qqoqcp_quoi"]
        assert projet["qqoqcp_ou"].startswith("Ligne CMS 2")

    def test_containment_cree_en_priorite_1(self, fresh_db):
        _, pid = _incident_et_projet()
        aid = db.add_action(
            projet_id=pid,
            titre="[CONTAINMENT] Tri des lots depuis le 29/06 + AOI renforce",
            assignee="Chef equipe A",
            priorite=1,
        )
        actions = db.list_actions()
        row = actions[actions["id"] == aid].iloc[0]
        assert row["titre"].startswith("[CONTAINMENT]")
        assert int(row["priorite"]) == 1

    def test_catalogue_defauts_disponible(self):
        from vue_b.f2_cadrage import CATALOGUE_DEFAUTS, _defaut_by_label
        assert len(CATALOGUE_DEFAUTS) >= 5
        codes = [d["code"] for d in CATALOGUE_DEFAUTS]
        assert len(codes) == len(set(codes)), "codes defauts uniques"
        d = _defaut_by_label("BRA-01 — Joints brases froids (AOI)")
        assert d["code"] == "BRA-01"
        assert d["quoi"]


# ===========================================================================
# F3 — correction humaine d'un noeud
# ===========================================================================
class TestF3CorrectionNoeud:
    def test_correction_noeud_persiste(self, fresh_db):
        iid, pid = _incident_et_projet()
        result = generate_full_tree(iid)
        save_tree_to_db(pid, result["tree"])

        arbre = cb.list_chemins_projet(pid, branche_m="Materiel")
        assert not arbre.empty
        node_id = int(arbre.sort_values("niveau").iloc[2]["id"])

        ok = cb.update_chemin(
            node_id,
            reponse="Connecteur element chauffant zone 6 mal serre "
                    "(maintenance du 29/06)",
            est_cause_racine=False,
        )
        assert ok
        node = cb.get_chemin(node_id)
        assert "zone 6" in node["reponse"]


# ===========================================================================
# F5 — solution chiffree + traduction par role
# ===========================================================================
class TestF5SolutionDemo:
    def test_solution_et_statut(self, fresh_db):
        _, pid = _incident_et_projet()
        db.update_projet_ac(pid, statut="valide")
        ok = db.update_projet_ac(
            pid,
            solution_proposee="Re-serrage connecteur zone 6 + plaque de "
                              "profilage obligatoire apres maintenance",
            cout_estime=150.0,
            temps_estime_jours=1.0,
            gain_estime_eur=42000.0,
            gain_productivite=2.5,
            statut="solution_propose",
        )
        assert ok
        projet = db.get_projet_ac(pid)
        assert projet["statut"] == "solution_propose"
        assert projet["gain_estime_eur"] == 42000.0
        assert "plaque de profilage" in projet["solution_proposee"]


# ===========================================================================
# Synthese — contenu des 6 ecrans gains & livrables
# ===========================================================================
class TestSyntheses:
    def test_six_etapes_completes(self):
        from vue_b.synthese import SYNTHESES
        assert set(SYNTHESES.keys()) == {"f1", "f2", "f3", "f4", "f5", "f6"}
        for step, data in SYNTHESES.items():
            assert data["titre"], step
            assert data["objectif_avant"], step
            assert data["objectif_apres"], step
            assert len(data["mecanismes"]) >= 3, step
            assert len(data["livrables"]) >= 2, step
