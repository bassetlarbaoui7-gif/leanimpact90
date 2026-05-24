"""
Tests F3 V2 - Pipeline d'entrainement supervise par M.

Couverture :
  1. classify_cause_to_m       - mapping mots-cles -> M (multi-label)
  2. generate_synthetic_history - reproductibilite, schema, distribution
  3. build_per_m_dataset       - features, NONE_LABEL, multi-M tagging
  4. train_m_model             - retourne modele + metriques, accuracy
  5. Refus volume insuffisant
  6. Refus classes insuffisantes
  7. Export ONNX rechargeable + predictions coherentes
  8. train_all_models          - orchestration, metadata.json, skip Methode
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
    build_per_m_dataset,
    classify_cause_to_m,
    export_to_onnx,
    generate_synthetic_global_history,
    train_all_models,
    train_m_model,
)


# ---------------------------------------------------------------------------
# 1. classify_cause_to_m
# ---------------------------------------------------------------------------
class TestClassifyCauseToM:
    def test_machine_simple(self):
        res = classify_cause_to_m("Buse encrassee sur ligne 3")
        assert "Machine" in res

    def test_matiere_simple(self):
        res = classify_cause_to_m("Lot kraft humide fournisseur A")
        assert "Matiere" in res

    def test_methode_simple(self):
        res = classify_cause_to_m("Mauvaise recette appliquee par operateur")
        assert "Methode" in res

    def test_multi_label_machine_matiere(self):
        """Une cause peut concerner plusieurs M simultanement."""
        res = classify_cause_to_m(
            "Lot kraft humide + temperature colle basse"
        )
        assert "Matiere" in res
        assert "Machine" in res

    def test_aucune_correspondance(self):
        res = classify_cause_to_m("xyz blabla 12345")
        assert res == set()

    def test_chaine_vide(self):
        assert classify_cause_to_m("") == set()

    def test_none_safe(self):
        assert classify_cause_to_m(None) == set()

    def test_non_string_safe(self):
        assert classify_cause_to_m(42) == set()

    def test_accents_diacritiques(self):
        """Le regex doit matcher avec ou sans accents."""
        res1 = classify_cause_to_m("Temperature elevee")
        res2 = classify_cause_to_m("Température élevée")
        # Les deux doivent renvoyer Machine via "temperature"
        assert "Machine" in res1
        assert "Machine" in res2


# ---------------------------------------------------------------------------
# 2. generate_synthetic_global_history
# ---------------------------------------------------------------------------
class TestGenerateSynthetic:
    def test_taille_correcte(self):
        df = generate_synthetic_global_history(n=200, seed=0)
        assert len(df) == 200

    def test_seed_reproductible(self):
        """Meme seed => meme dataset (critique pour les tests)."""
        df1 = generate_synthetic_global_history(n=100, seed=123)
        df2 = generate_synthetic_global_history(n=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_seed_different_change_data(self):
        df1 = generate_synthetic_global_history(n=100, seed=1)
        df2 = generate_synthetic_global_history(n=100, seed=2)
        # Les causes-racines doivent differer au moins sur quelques lignes
        diff = (df1["cause_racine"] != df2["cause_racine"]).sum()
        assert diff > 5

    def test_schema_complet(self):
        df = generate_synthetic_global_history(n=50)
        for col in ["defect_id", "defect_type", "timestamp", "cause_racine"]:
            assert col in df.columns
        # Au moins 3 features par M
        for m in M_BRANCHES:
            prefix = f"M_{m.lower()}_"
            assert sum(c.startswith(prefix) for c in df.columns) >= 3

    def test_distribution_realiste(self):
        """Sur un gros echantillon, Machine domine, Methode minoritaire."""
        df = generate_synthetic_global_history(n=2000, seed=42)
        causes = df["cause_racine"].astype(str).str.lower()
        n_machine = causes.apply(
            lambda c: "Machine" in classify_cause_to_m(c)
        ).sum()
        n_methode = causes.apply(
            lambda c: "Methode" in classify_cause_to_m(c)
        ).sum()
        # Ordre attendu : Machine plus frequent que Methode
        assert n_machine > n_methode

    def test_features_numeriques(self):
        df = generate_synthetic_global_history(n=50)
        for col in df.columns:
            if col.startswith("M_"):
                assert pd.api.types.is_numeric_dtype(df[col])


# ---------------------------------------------------------------------------
# 3. build_per_m_dataset
# ---------------------------------------------------------------------------
class TestBuildPerMDataset:
    @pytest.fixture
    def history(self):
        return generate_synthetic_global_history(n=300, seed=42)

    def test_features_correctes(self, history):
        X, y, feats = build_per_m_dataset(history, m="Machine")
        for f in feats:
            assert f.startswith("M_machine_")
        assert X.shape[1] == len(feats)
        assert len(y) == len(history)

    def test_y_taggue_multi_m(self, history):
        """Une cause Machine doit aussi figurer dans le dataset Matiere
        si elle concerne aussi Matiere (multi-label)."""
        _, y_mach, _ = build_per_m_dataset(history, m="Machine")
        _, y_mat, _ = build_per_m_dataset(history, m="Matiere")
        # Multi-cause "Lot kraft humide + temperature colle basse"
        causes = history["cause_racine"].astype(str).values
        for i, c in enumerate(causes):
            if "kraft humide" in c.lower() and "temperature colle" in c.lower():
                assert y_mach[i] != NONE_LABEL
                assert y_mat[i] != NONE_LABEL

    def test_none_label_pour_autres_m(self, history):
        """Une cause purement Methode doit etre NONE_LABEL pour Machine."""
        _, y_mach, _ = build_per_m_dataset(history, m="Machine")
        causes = history["cause_racine"].astype(str).values
        for i, c in enumerate(causes):
            if c == "Mauvaise recette appliquee":
                assert y_mach[i] == NONE_LABEL

    def test_m_inconnue_leve(self, history):
        with pytest.raises(ValueError):
            build_per_m_dataset(history, m="Inexistant")

    def test_label_col_absent_leve(self, history):
        with pytest.raises(ValueError):
            build_per_m_dataset(history, m="Machine", label_col="zzz")

    def test_prefix_sans_colonnes_leve(self):
        df = pd.DataFrame({"cause_racine": ["x"], "autre": [1.0]})
        with pytest.raises(ValueError):
            build_per_m_dataset(df, m="Machine")


# ---------------------------------------------------------------------------
# 4. train_m_model - succes
# ---------------------------------------------------------------------------
class TestTrainMModelSucces:
    def test_entrainement_machine_atteint_seuil(self):
        """Sur synthetique 600 lignes : top-3 Machine doit depasser 80%."""
        df = generate_synthetic_global_history(n=600, seed=42)
        X, y, _ = build_per_m_dataset(df, m="Machine")
        model, le, metrics, reason = train_m_model(X, y, m_name="Machine")
        assert model is not None
        assert le is not None
        assert metrics is not None
        assert reason == ""
        assert metrics.top3_accuracy >= 0.80, (
            f"top3={metrics.top3_accuracy:.2%} sous seuil"
        )

    def test_metrics_structure_complete(self):
        df = generate_synthetic_global_history(n=400, seed=42)
        X, y, _ = build_per_m_dataset(df, m="Machine")
        _, _, metrics, _ = train_m_model(X, y, m_name="Machine")
        assert metrics.n_total > 0
        assert metrics.n_real_cases > 0
        assert metrics.n_classes >= 2
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.top3_accuracy <= 1.0
        assert isinstance(metrics.classes, list)


# ---------------------------------------------------------------------------
# 5. Refus volume insuffisant
# ---------------------------------------------------------------------------
class TestRefusVolume:
    def test_volume_insuffisant_retourne_raison(self):
        df = generate_synthetic_global_history(n=50, seed=42)
        X, y, _ = build_per_m_dataset(df, m="Machine")
        model, le, metrics, reason = train_m_model(
            X, y, m_name="Machine", min_cases=200,
        )
        assert model is None
        assert le is None
        assert metrics is None
        assert "insuffisant" in reason.lower()

    def test_methode_skip_sur_synthetique_400(self):
        """Methode est minoritaire (15%) - sur 400 lignes elle doit etre skip
        au seuil par defaut (150)."""
        df = generate_synthetic_global_history(n=400, seed=42)
        X, y, _ = build_per_m_dataset(df, m="Methode")
        model, _, _, reason = train_m_model(X, y, m_name="Methode")
        # Avec ~76 cas reels Methode < 150 : refus attendu
        assert model is None
        assert "insuffisant" in reason.lower()


# ---------------------------------------------------------------------------
# 6. Refus classes insuffisantes
# ---------------------------------------------------------------------------
class TestRefusClasses:
    def test_classes_insuffisantes_retourne_raison(self):
        """Une seule classe reelle (apres fusion rares) -> refus."""
        # 200 cas tous "Buse encrassee" + 100 NONE_LABEL
        n = 300
        rng = np.random.default_rng(0)
        X = pd.DataFrame({
            "M_machine_temp": rng.normal(180, 5, n),
            "M_machine_pression": rng.normal(4.5, 0.2, n),
        })
        y = np.array(
            ["Buse encrassee"] * 200 + [NONE_LABEL] * 100
        )
        model, _, _, reason = train_m_model(
            X, y, m_name="Machine",
            min_cases=50, min_classes=3,
        )
        assert model is None
        assert "classes" in reason.lower()


# ---------------------------------------------------------------------------
# 7. Export ONNX rechargeable
# ---------------------------------------------------------------------------
class TestExportONNX:
    def test_onnx_loadable_et_predit(self, tmp_path):
        """Export ONNX -> reload via onnxruntime -> predictions coherentes."""
        import onnxruntime as ort

        df = generate_synthetic_global_history(n=400, seed=42)
        X, y, feats = build_per_m_dataset(df, m="Machine")
        model, le, _, reason = train_m_model(X, y, m_name="Machine")
        assert model is not None, f"Pre-condition echouee : {reason}"

        onnx_path = tmp_path / "model_machine.onnx"
        export_to_onnx(model, feats, onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"],
        )
        # Prediction sur un echantillon
        X_sample = X.head(5).astype(np.float32).to_numpy()
        outputs = sess.run(None, {"input": X_sample})
        # On attend label + proba (zipmap=False)
        assert len(outputs) >= 1
        # Le second output (proba) doit avoir shape (5, n_classes)
        proba_onnx = outputs[1]
        assert proba_onnx.shape[0] == 5
        assert proba_onnx.shape[1] == len(le.classes_)
        # Toutes les probas dans [0, 1]
        assert (proba_onnx >= 0).all()
        assert (proba_onnx <= 1.0001).all()


# ---------------------------------------------------------------------------
# 8. train_all_models - orchestration complete
# ---------------------------------------------------------------------------
class TestTrainAllModels:
    def test_orchestration_synthetique_400(self, tmp_path):
        df = generate_synthetic_global_history(n=400, seed=42)
        report = train_all_models(df, output_dir=tmp_path)

        # Au moins Machine et Matiere entraines
        assert "Machine" in report.models_trained
        assert "Matiere" in report.models_trained
        # Methode skip sur 400 lignes (volume insuffisant)
        assert "Methode" in report.models_skipped

    def test_metadata_json_structure(self, tmp_path):
        df = generate_synthetic_global_history(n=400, seed=42)
        train_all_models(df, output_dir=tmp_path)

        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert "trained_at" in meta
        assert "history_size" in meta
        assert meta["history_size"] == 400
        assert "models" in meta

        # Chaque modele entraine doit avoir features, classes, metrics
        for m, info in meta["models"].items():
            assert "onnx_file" in info
            assert "features" in info
            assert "classes" in info
            assert "metrics" in info
            assert "accuracy" in info["metrics"]
            assert "top3_accuracy" in info["metrics"]
            # Le fichier ONNX doit reellement exister
            assert (tmp_path / info["onnx_file"]).exists()

    def test_summary_lisible(self, tmp_path):
        df = generate_synthetic_global_history(n=400, seed=42)
        report = train_all_models(df, output_dir=tmp_path)
        text = report.summary()
        assert "Entrainement F3 V2" in text
        assert "Machine" in text or "Matiere" in text
        # Lignes pour skip
        assert "SKIP" in text or "Methode" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
