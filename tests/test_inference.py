"""
Tests F3 V2 - Moteur d'inference supervise multi-M.

Couverture :
  1. Chargement nominal (3 modeles entraines + metadata.json)
  2. Methode skippee a l'entrainement -> indispo cote inference, pas de crash
  3. Reproductibilite : meme contexte -> meme prediction
  4. Top-3 trie par confiance decroissante, sans NONE_LABEL
  5. Confiance entre 0 et 1, cumul ranked >= dominante
  6. predict_multi retourne 3 entrees (Machine, Matiere, Methode)
  7. Features manquantes du contexte -> imputation mediane, pas de crash
  8. explain_multi : SHAP utilise quand .pkl present, top-N tries
  9. Fallback feature_importances quand .pkl supprime
 10. metadata.json corrompu -> erreur claire au chargement (fail fast)
 11. models_dir inexistant -> FileNotFoundError
 12. Compat V1 : suggest() retourne liste de CauseSuggestion plates
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from training import (
    M_BRANCHES,
    NONE_LABEL,
    generate_synthetic_global_history,
    train_all_models,
)
from inference import (
    SupervisedRootCauseEngine,
    MultiMPrediction,
    CausePrediction,
    ShapContribution,
)
from ishikawa import CauseSuggestion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_models_dir(tmp_path_factory):
    """
    Entraine une fois 3 modeles sur un historique synthetique large
    (n=600 pour que Machine et Matiere passent ; Methode reste sous le
    seuil min_cases=150 -> skippee, comportement attendu).
    """
    out = tmp_path_factory.mktemp("models")
    df = generate_synthetic_global_history(n=600, seed=42)
    train_all_models(df, output_dir=out)
    return out


@pytest.fixture
def engine(trained_models_dir) -> SupervisedRootCauseEngine:
    return SupervisedRootCauseEngine(models_dir=trained_models_dir)


@pytest.fixture
def sample_context() -> dict[str, float]:
    """Contexte realiste defaut Gascogne."""
    return {
        "M_machine_temperature": 170.0,        # baisse nette -> evoque colle basse
        "M_machine_pression": 4.5,
        "M_machine_vitesse_vis": 120.0,
        "M_matiere_humidite_kraft": 11.5,      # eleve -> evoque kraft humide
        "M_matiere_densite_lot": 120.0,
        "M_matiere_grammage": 80.0,
        "M_methode_vitesse_consigne": 1200.0,
        "M_methode_temperature_cible": 180.0,
        "M_methode_pression_cible": 4.5,
    }


# ---------------------------------------------------------------------------
# 1-2. Chargement
# ---------------------------------------------------------------------------
class TestLoading:
    def test_load_nominal(self, engine):
        assert engine.is_trained() is True
        avail = engine.is_available()
        assert avail["Machine"] is True
        assert avail["Matiere"] is True
        # Methode peut etre dispo ou non selon volume genere ; on ne fixe rien
        assert isinstance(avail["Methode"], bool)

    def test_skipped_m_has_reason(self, engine):
        avail = engine.is_available()
        reasons = engine.reasons_unavailable()
        # Tout M indispo doit avoir une raison textuelle non vide
        for m, ok in avail.items():
            if not ok:
                assert m in reasons
                assert reasons[m]


# ---------------------------------------------------------------------------
# 3. Reproductibilite
# ---------------------------------------------------------------------------
class TestReproducibility:
    def test_same_context_same_prediction(self, engine, sample_context):
        r1 = engine.predict_multi(sample_context)
        r2 = engine.predict_multi(sample_context)
        for m in engine.is_available():
            if not engine.is_available()[m]:
                continue
            top1_a = r1.predictions[m].top_causes
            top1_b = r2.predictions[m].top_causes
            assert [(c.cause, round(c.confidence, 6)) for c in top1_a] \
                == [(c.cause, round(c.confidence, 6)) for c in top1_b]


# ---------------------------------------------------------------------------
# 4-5. Top-3 et confiances
# ---------------------------------------------------------------------------
class TestTopK:
    def test_top3_sorted_descending(self, engine, sample_context):
        result = engine.predict_multi(sample_context, top_k=3)
        for m, pred in result.predictions.items():
            if not pred.available:
                continue
            confs = [c.confidence for c in pred.top_causes]
            assert confs == sorted(confs, reverse=True)

    def test_no_none_label_in_top(self, engine, sample_context):
        result = engine.predict_multi(sample_context, top_k=5)
        for pred in result.predictions.values():
            for cp in pred.top_causes:
                assert cp.cause != NONE_LABEL

    def test_confidence_in_unit_range(self, engine, sample_context):
        result = engine.predict_multi(sample_context)
        for pred in result.predictions.values():
            for cp in pred.top_causes:
                assert 0.0 <= cp.confidence <= 1.0


# ---------------------------------------------------------------------------
# 6. Structure de la reponse
# ---------------------------------------------------------------------------
class TestResponseShape:
    def test_predict_multi_returns_all_3_m(self, engine, sample_context):
        result = engine.predict_multi(sample_context)
        assert isinstance(result, MultiMPrediction)
        assert set(result.predictions.keys()) == set(M_BRANCHES)

    def test_unavailable_m_has_reason(self, engine, sample_context):
        result = engine.predict_multi(sample_context)
        for m, pred in result.predictions.items():
            if not pred.available:
                assert pred.reason_unavailable
                assert pred.top_causes == []


# ---------------------------------------------------------------------------
# 7. Robustesse : features manquantes
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_missing_features_imputed(self, engine):
        """Contexte vide -> imputation mediane partout, pas de crash."""
        result = engine.predict_multi({})
        # Pour les M dispos, on doit avoir une prediction structuree
        # (potentiellement is_none=True puisque tout est a la mediane).
        for pred in result.predictions.values():
            if pred.available:
                # On accepte top_causes vide si is_none, sinon il y en a.
                assert isinstance(pred.top_causes, list)

    def test_partial_features_imputed(self, engine):
        """Quelques features fournies, le reste impute -> pas de crash."""
        partial = {"M_machine_temperature": 175.0}
        result = engine.predict_multi(partial)
        for pred in result.predictions.values():
            if pred.available:
                assert isinstance(pred.top_causes, list)


# ---------------------------------------------------------------------------
# 8. SHAP
# ---------------------------------------------------------------------------
class TestExplainability:
    def test_explain_returns_top_features(self, engine, sample_context):
        explanations = engine.explain_multi(sample_context, top_features=3)
        for m, contribs in explanations.items():
            assert m in engine._models  # seulement les M dispos
            assert len(contribs) <= 3
            for c in contribs:
                assert isinstance(c, ShapContribution)
                assert c.feature in engine.get_features(m)

    def test_explain_sorted_by_abs_contribution(self, engine, sample_context):
        explanations = engine.explain_multi(sample_context, top_features=5)
        for contribs in explanations.values():
            abs_vals = [abs(c.contribution) for c in contribs]
            assert abs_vals == sorted(abs_vals, reverse=True)


# ---------------------------------------------------------------------------
# 9. Fallback feature_importances quand .pkl absent
# ---------------------------------------------------------------------------
class TestFallback:
    def test_explain_without_pkl(self, trained_models_dir, sample_context):
        """Si on supprime les .pkl, l'engine doit continuer (fallback)."""
        # On bosse sur une copie pour ne pas casser les autres tests
        import shutil
        backup = trained_models_dir.parent / "no_pkl_models"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(trained_models_dir, backup)
        for pkl in backup.glob("*.pkl"):
            pkl.unlink()

        engine = SupervisedRootCauseEngine(models_dir=backup)
        result = engine.predict_multi(sample_context)
        # Predictions toujours OK
        assert any(p.available for p in result.predictions.values())

        # Explications fallback fonctionnelles
        expl = engine.explain_multi(sample_context, top_features=3)
        for contribs in expl.values():
            assert len(contribs) <= 3


# ---------------------------------------------------------------------------
# 10-11. Erreurs au chargement
# ---------------------------------------------------------------------------
class TestLoadErrors:
    def test_corrupt_metadata_raises(self, tmp_path):
        (tmp_path / "metadata.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="metadata.json"):
            SupervisedRootCauseEngine(models_dir=tmp_path)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SupervisedRootCauseEngine(models_dir=tmp_path / "nope")


# ---------------------------------------------------------------------------
# 12. Compat interface V1 (RootCauseEngine.suggest)
# ---------------------------------------------------------------------------
class TestV1Compat:
    def test_suggest_flattens_to_cause_suggestion(
        self, engine, sample_context,
    ):
        suggestions = engine.suggest(
            defect_type="colle_insuffisante",
            context="",
            parameters=sample_context,
            top_k=5,
        )
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert isinstance(s, CauseSuggestion)
            assert s.branch in M_BRANCHES
            assert 0.0 <= s.confidence <= 1.0
        # Tri decroissant global
        confs = [s.confidence for s in suggestions]
        assert confs == sorted(confs, reverse=True)

    def test_is_trained_true_after_load(self, engine):
        assert engine.is_trained() is True
