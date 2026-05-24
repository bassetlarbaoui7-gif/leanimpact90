"""
Tests des helpers V2 ajoutes a ishikawa_ui.py.

Couverture (pure logic, pas de runtime Streamlit) :
  1. Import de ishikawa_ui sans erreur (toutes deps OK)
  2. try_load_supervised_engine sur dossier vide -> None (silencieux)
  3. try_load_supervised_engine sur metadata corrompu -> None (silencieux)
  4. try_load_supervised_engine sur modele entraine -> SupervisedRootCauseEngine
  5. v2_status_html(None) -> mention 'non charge'
  6. v2_status_html(engine) -> 3 pills (Machine, Matiere, Methode)
  7. shap_explanation_html sur liste vide -> table sans lignes
  8. shap_explanation_html avec contributions -> contient les noms de features
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from inference import ShapContribution, SupervisedRootCauseEngine
from training import generate_synthetic_global_history, train_all_models


# ---------------------------------------------------------------------------
# 1. Import du module UI sans crash
# ---------------------------------------------------------------------------
def test_import_ishikawa_ui():
    """ishikawa_ui doit s'importer proprement (toutes deps resolues)."""
    import ishikawa_ui  # noqa: F401
    # Helpers exposes
    assert hasattr(ishikawa_ui, "try_load_supervised_engine")
    assert hasattr(ishikawa_ui, "v2_status_html")
    assert hasattr(ishikawa_ui, "shap_explanation_html")
    assert hasattr(ishikawa_ui, "render_v2_models_panel")
    assert hasattr(ishikawa_ui, "render_v2_suggestions")
    assert hasattr(ishikawa_ui, "render_ishikawa_page")


# ---------------------------------------------------------------------------
# 2-4. try_load_supervised_engine
# ---------------------------------------------------------------------------
class TestTryLoad:
    def test_empty_dir_returns_none(self, tmp_path):
        from ishikawa_ui import try_load_supervised_engine
        assert try_load_supervised_engine(tmp_path) is None

    def test_corrupt_metadata_returns_none(self, tmp_path):
        """Doit echouer silencieusement, pas crasher la page."""
        from ishikawa_ui import try_load_supervised_engine
        (tmp_path / "metadata.json").write_text("{not json", encoding="utf-8")
        assert try_load_supervised_engine(tmp_path) is None

    def test_trained_models_returns_engine(self, tmp_path_factory):
        from ishikawa_ui import try_load_supervised_engine
        out = tmp_path_factory.mktemp("models_v2")
        df = generate_synthetic_global_history(n=400, seed=42)
        train_all_models(df, output_dir=out, min_cases=80)
        engine = try_load_supervised_engine(out)
        assert isinstance(engine, SupervisedRootCauseEngine)
        assert engine.is_trained()


# ---------------------------------------------------------------------------
# 5-6. v2_status_html
# ---------------------------------------------------------------------------
class TestStatusHtml:
    def test_none_engine_warn(self):
        from ishikawa_ui import v2_status_html
        html = v2_status_html(None)
        assert "non charge" in html.lower()
        assert "warn" in html

    def test_engine_three_pills(self, tmp_path_factory):
        from ishikawa_ui import try_load_supervised_engine, v2_status_html
        out = tmp_path_factory.mktemp("models_status")
        df = generate_synthetic_global_history(n=400, seed=42)
        train_all_models(df, output_dir=out, min_cases=80)
        engine = try_load_supervised_engine(out)
        html = v2_status_html(engine)
        # 3 pills attendues : une par M
        for m in ["Machine", "Matiere", "Methode"]:
            assert m in html


# ---------------------------------------------------------------------------
# 7-8. shap_explanation_html
# ---------------------------------------------------------------------------
class TestShapHtml:
    def test_empty_list_renders_table(self):
        from ishikawa_ui import shap_explanation_html
        html = shap_explanation_html([])
        assert "<table" in html
        assert "<tbody>" in html

    def test_contributions_show_features(self):
        from ishikawa_ui import shap_explanation_html
        contribs = [
            ShapContribution(
                feature="M_machine_temperature",
                value=170.0, contribution=0.45, median=180.0,
            ),
            ShapContribution(
                feature="M_matiere_humidite",
                value=11.0, contribution=-0.30, median=7.5,
            ),
        ]
        html = shap_explanation_html(contribs)
        assert "M_machine_temperature" in html
        assert "M_matiere_humidite" in html
        # Le signe + ou - doit apparaitre pour chaque contribution
        assert "+0.450" in html or "+0.45" in html
        assert "-0.30" in html or "-0.300" in html
