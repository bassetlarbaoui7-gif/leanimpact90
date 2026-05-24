"""
Tests du module F3 Ishikawa : stockage SQLite, moteur IA TF-IDF,
auto-remplissage Machine / Matiere.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from ishikawa import (
    AUTO_BRANCHES, BRANCHES_5M, MANUAL_BRANCHES, SEMI_AUTO_BRANCHES,
    RcaCase, RootCauseEngine, auto_fill_machine_branch,
    auto_fill_matiere_branch, branch_mode, empty_5m_structure,
    init_db, load_all_rca, save_rca,
)


# ---------------------------------------------------------------------------
# Structure 5M
# ---------------------------------------------------------------------------
class TestStructure5M:
    def test_branches_completes(self):
        assert len(BRANCHES_5M) == 5
        assert set(AUTO_BRANCHES | SEMI_AUTO_BRANCHES | MANUAL_BRANCHES) == \
            set(BRANCHES_5M)

    def test_empty_structure(self):
        struct = empty_5m_structure()
        assert set(struct.keys()) == set(BRANCHES_5M)
        assert all(v == [] for v in struct.values())

    def test_branch_mode(self):
        assert branch_mode("Machine") == "auto"
        assert branch_mode("Methode") == "semi-auto"
        assert branch_mode("Milieu") == "manuel"


# ---------------------------------------------------------------------------
# Persistance SQLite
# ---------------------------------------------------------------------------
class TestPersistance:
    def test_save_and_reload(self, tmp_path):
        db = tmp_path / "test.db"
        case = RcaCase(
            defect_type="colle_insuffisante",
            context="Defaut observe sur la colleuse ligne 3 vers 14h",
            parameters={"temperature_colle": 175.5, "pression_ref": 4.8},
            causes_5m={
                "Machine": ["Temperature colle basse", "Buse encrassee"],
                "Matiere": ["Lot kraft atypique"],
                "Methode": [],
                "Main-d'oeuvre": [],
                "Milieu": ["Humidite atelier elevee"],
            },
            validated_root_cause="Temperature colle basse",
            validated_branch="Machine",
            operator="Jean Dupont",
        )
        rca_id = save_rca(case, db_path=db)
        assert rca_id > 0

        loaded = load_all_rca(db_path=db)
        assert len(loaded) == 1
        reloaded = loaded[0]
        assert reloaded.defect_type == "colle_insuffisante"
        assert reloaded.validated_root_cause == "Temperature colle basse"
        assert reloaded.parameters["temperature_colle"] == 175.5
        assert "Temperature colle basse" in reloaded.causes_5m["Machine"]

    def test_multiple_cases(self, tmp_path):
        db = tmp_path / "multi.db"
        for i in range(5):
            save_rca(RcaCase(
                defect_type=f"defaut_{i}",
                context=f"Contexte {i}",
                validated_root_cause=f"cause_{i}",
                validated_branch="Machine",
            ), db_path=db)
        loaded = load_all_rca(db_path=db)
        assert len(loaded) == 5

    def test_init_db_idempotent(self, tmp_path):
        db = tmp_path / "idem.db"
        init_db(db)
        init_db(db)  # doit pas planter
        assert db.exists()


# ---------------------------------------------------------------------------
# Moteur IA : TF-IDF + cosine
# ---------------------------------------------------------------------------
class TestRootCauseEngine:
    def test_empty_history_no_suggestions(self):
        engine = RootCauseEngine(min_history=3)
        engine.fit([])
        assert not engine.is_trained()
        sugg = engine.suggest("any", "context", {})
        assert sugg == []

    def test_below_threshold(self):
        engine = RootCauseEngine(min_history=3)
        engine.fit([RcaCase(defect_type="x", context="y")])
        assert not engine.is_trained()

    def test_retrieves_similar_case(self):
        cases = [
            RcaCase(
                defect_type="colle_insuffisante",
                context="colleuse ligne 3 temperature basse",
                parameters={"temperature_colle": 170.0},
                validated_root_cause="Temperature colle basse",
                validated_branch="Machine",
                created_at="2026-01-15T10:00:00",
            ),
            RcaCase(
                defect_type="deformation",
                context="deformation sac ligne 2 pression elevee",
                parameters={"pression_vis": 5.2},
                validated_root_cause="Pression vis trop elevee",
                validated_branch="Machine",
                created_at="2026-01-20T10:00:00",
            ),
            RcaCase(
                defect_type="tachage",
                context="tache encre ligne 1 lot kraft humide",
                parameters={"humidite_kraft": 12.5},
                validated_root_cause="Humidite kraft elevee",
                validated_branch="Matiere",
                created_at="2026-02-01T10:00:00",
            ),
        ]
        engine = RootCauseEngine(min_history=3)
        engine.fit(cases)
        assert engine.is_trained()

        # Requete proche du cas 1
        sugg = engine.suggest(
            defect_type="colle_insuffisante",
            context="colleuse ligne 3 probleme temperature",
            parameters={"temperature_colle": 172.0},
            top_k=3,
        )
        assert len(sugg) >= 1
        # La premiere suggestion doit etre "Temperature colle basse"
        assert sugg[0].cause == "Temperature colle basse"
        assert sugg[0].branch == "Machine"
        assert sugg[0].confidence > 0

    def test_top_k_limit(self):
        cases = [
            RcaCase(
                defect_type=f"d_{i}", context=f"probleme {i}",
                validated_root_cause=f"cause_{i}",
                validated_branch="Machine",
            )
            for i in range(10)
        ]
        engine = RootCauseEngine(min_history=3)
        engine.fit(cases)
        sugg = engine.suggest("d_1", "probleme 1", {}, top_k=3)
        assert len(sugg) <= 3


# ---------------------------------------------------------------------------
# Auto-remplissage Machine
# ---------------------------------------------------------------------------
class TestAutoFillMachine:
    def test_hors_plage_haut(self):
        suspects = auto_fill_machine_branch(
            {"temperature": 195.0, "pression": 4.5},
            {"temperature": (175.0, 185.0), "pression": (4.0, 5.0)},
        )
        assert len(suspects) == 1
        assert "temperature haut" in suspects[0]

    def test_hors_plage_bas(self):
        suspects = auto_fill_machine_branch(
            {"temperature": 165.0},
            {"temperature": (175.0, 185.0)},
        )
        assert "temperature bas" in suspects[0]

    def test_dans_plage(self):
        suspects = auto_fill_machine_branch(
            {"temperature": 180.0},
            {"temperature": (175.0, 185.0)},
        )
        assert suspects == []

    def test_parametre_sans_cible_ignore(self):
        suspects = auto_fill_machine_branch(
            {"unknown": 999.0, "temperature": 170.0},
            {"temperature": (175.0, 185.0)},
        )
        assert len(suspects) == 1
        assert "temperature" in suspects[0]


# ---------------------------------------------------------------------------
# Auto-remplissage Matiere (z-score)
# ---------------------------------------------------------------------------
class TestAutoFillMatiere:
    def test_lot_atypique_detecte(self):
        historical = pd.DataFrame({
            "humidite_kraft": np.random.normal(7.0, 0.5, 50),
            "densite": np.random.normal(120, 3, 50),
        })
        suspects = auto_fill_matiere_branch(
            {"humidite_kraft": 12.0, "densite": 120.0},
            historical,
            z_threshold=2.0,
        )
        assert any("humidite_kraft" in s for s in suspects)
        assert not any("densite" in s for s in suspects)

    def test_historique_insuffisant(self):
        historical = pd.DataFrame({"x": [1.0, 2.0]})  # < 5
        suspects = auto_fill_matiere_branch(
            {"x": 100.0}, historical, z_threshold=2.0,
        )
        assert suspects == []

    def test_std_nulle_ignore(self):
        historical = pd.DataFrame({"x": [1.0] * 10})  # std = 0
        suspects = auto_fill_matiere_branch(
            {"x": 100.0}, historical, z_threshold=2.0,
        )
        assert suspects == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
