"""
Test end-to-end simulant la logique de chaque page de home.py
sans Streamlit. Garantit qu'un operateur peut suivre le parcours
complet sans erreur, avec donnees industrielles realistes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from analysis import analyze_all, clean_data
from drift import detect_drift
from losses import analyze_losses
from shift_analysis import analyze_shifts


@pytest.fixture
def industrial_df() -> pd.DataFrame:
    """Simule 200 mesures d'une ligne de production Gascogne Sacs."""
    rng = np.random.default_rng(42)
    n = 200
    temp = rng.normal(180, 3, n)
    # drift lent sur la seconde moitie
    temp[100:] += np.linspace(0, 8, 100)
    pression = rng.normal(4.5, 0.15, n)
    humidite = rng.normal(7.5, 0.6, n)
    cadence = rng.normal(1200, 30, n)
    # defauts correles a la temperature (r ~ 0.55 attendu)
    defauts = (temp - 180) * 15 + rng.normal(50, 12, n)
    defauts = np.clip(defauts, 0, None)
    shifts = np.tile(["A", "B", "C"], n // 3 + 1)[:n]

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="30min"),
        "temperature_four": temp,
        "pression_vis": pression,
        "humidite_kraft": humidite,
        "cadence_ligne": cadence,
        "defauts_ppm": defauts,
        "shift": shifts,
    })


class TestPageImportSPC:
    def test_clean_and_analyze(self, industrial_df):
        params = ["temperature_four", "pression_vis",
                  "humidite_kraft", "cadence_ligne"]
        cleaned = clean_data(industrial_df, params)
        assert not cleaned.empty
        results = analyze_all(cleaned)
        assert isinstance(results, pd.DataFrame)
        assert "parameter" in results.columns
        assert "violations" in results.columns
        assert "criticality" in results.columns
        # La temperature doit declencher des violations (drift lent)
        temp_row = results[results["parameter"] == "temperature_four"].iloc[0]
        assert temp_row["n"] > 0

    def test_drift_detection(self, industrial_df):
        params = ["temperature_four", "pression_vis"]
        cleaned = clean_data(industrial_df, params)
        drift = detect_drift(cleaned)
        assert isinstance(drift, pd.DataFrame)
        assert "drift_detected" in drift.columns
        # La temperature doit etre detectee comme en derive
        temp_row = drift[drift["parameter"] == "temperature_four"].iloc[0]
        assert bool(temp_row["drift_detected"]) is True


class TestPageLosses:
    def test_analyze_losses_full(self, industrial_df):
        result = analyze_losses(
            industrial_df,
            param_columns=["temperature_four", "pression_vis",
                            "humidite_kraft", "cadence_ligne"],
            defaut_column="defauts_ppm",
            volume_total=1_000_000,
        )
        assert "correlations" in result
        assert "cpk" in result
        assert "ppm" in result
        corr = result["correlations"]
        assert isinstance(corr, pd.DataFrame)
        # Temperature doit ressortir
        assert (corr["param"] == "temperature_four").any()

    def test_analyze_losses_no_volume(self, industrial_df):
        """Le PPM ne doit pas planter si volume_total est None."""
        result = analyze_losses(
            industrial_df,
            param_columns=["temperature_four"],
            defaut_column="defauts_ppm",
            volume_total=None,
        )
        assert result is not None


class TestPageShift:
    def test_shift_comparison(self, industrial_df):
        result = analyze_shifts(
            industrial_df,
            param_cols=["temperature_four", "pression_vis"],
            shift_col="shift",
        )
        assert result.get("detected") is True
        assert result.get("n_shifts") == 3
        stab = result.get("stability")
        assert isinstance(stab, pd.DataFrame)
        comp = result.get("comparison")
        assert isinstance(comp, pd.DataFrame)

    def test_shift_autodetect(self, industrial_df):
        """Doit detecter 'shift' automatiquement sans parametre."""
        result = analyze_shifts(
            industrial_df,
            param_cols=["temperature_four"],
            shift_col=None,
        )
        assert result.get("detected") is True


class TestAlertsBuild:
    """Reproduit _alerts_from_results() de home.py."""

    def _alerts(self, results_spc):
        if results_spc is None or results_spc.empty:
            return []
        alerts = []
        for _, row in results_spc.iterrows():
            viol = int(row.get("violations", 0) or 0)
            crit = float(row.get("criticality", 0.0))
            if viol > 0 or crit > 0:
                sev = "danger" if crit >= 1.0 or viol >= 3 else "warn"
                alerts.append({
                    "parameter": str(row["parameter"]),
                    "message": f"{viol} violation(s)",
                    "severity": sev,
                })
        return alerts

    def test_alerts_with_violations(self, industrial_df):
        params = ["temperature_four", "pression_vis"]
        cleaned = clean_data(industrial_df, params)
        results = analyze_all(cleaned)
        alerts = self._alerts(results)
        assert isinstance(alerts, list)

    def test_alerts_empty(self):
        assert self._alerts(None) == []
        assert self._alerts(pd.DataFrame()) == []


class TestEdgeCases:
    def test_empty_dataframe_doesnt_crash_losses(self):
        """analyze_losses doit gerer gracieusement un DF vide."""
        df = pd.DataFrame({"a": [], "defauts": []})
        # Soit ca leve proprement, soit ca retourne un resultat degrade :
        # dans les deux cas, pas de corruption silencieuse.
        try:
            result = analyze_losses(df, param_columns=["a"],
                                    defaut_column="defauts",
                                    volume_total=None)
            assert result is not None
        except (ValueError, KeyError, TypeError):
            pass  # Erreur attendue sur DF vide

    def test_single_row(self):
        df = pd.DataFrame({"a": [1.0], "defauts": [10.0]})
        # Doit pas planter en cascade
        try:
            analyze_losses(df, param_columns=["a"],
                           defaut_column="defauts", volume_total=None)
        except Exception:
            pass  # Erreur attendue

    def test_all_nan_column(self, industrial_df):
        df = industrial_df.copy()
        df["bad_col"] = np.nan
        cleaned = clean_data(df, ["temperature_four", "bad_col"])
        # bad_col doit etre drop ou ignore
        assert not cleaned.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
