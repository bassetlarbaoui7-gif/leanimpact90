"""
Tests du module data_quality : audit, verdict, synthese.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from data_quality import audit_dataframe, column_quality, summary_stats


class TestColumnQuality:
    def test_colonne_propre(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="temp")
        res = column_quality(s)
        assert res["verdict"] == "ok"
        assert res["pct_missing"] == 0
        assert res["type"] == "numeric"

    def test_colonne_30pct_manquant(self):
        s = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0,
                       6.0, 7.0, 8.0, 9.0, 10.0], name="temp")
        res = column_quality(s)
        assert res["verdict"] == "warn"
        assert 28 <= res["pct_missing"] <= 32

    def test_colonne_60pct_manquant(self):
        s = pd.Series([1.0] + [np.nan] * 6 + [2.0, 3.0, 4.0], name="x")
        res = column_quality(s)
        assert res["verdict"] == "danger"

    def test_trou_capteur_long(self):
        """20 NaN consecutifs : warning meme si pct global est faible."""
        s = pd.Series(
            [1.0] * 100 + [np.nan] * 20 + [1.0] * 100, name="sensor"
        )
        res = column_quality(s)
        assert res["max_gap_nan"] == 20
        assert res["verdict"] == "warn"

    def test_constante_signale(self):
        s = pd.Series([42.0] * 50, name="cste")
        res = column_quality(s)
        assert res["verdict"] == "warn"
        assert "onstant" in res["commentaire"].lower()

    def test_type_datetime(self):
        s = pd.Series(pd.date_range("2026-01-01", periods=10), name="ts")
        res = column_quality(s)
        assert res["type"] == "datetime"

    def test_type_categoriel(self):
        s = pd.Series(["A", "B", "C"] * 10, name="shift")
        res = column_quality(s)
        assert res["type"] == "categorical"


class TestAuditDataframe:
    def test_audit_complet(self):
        df = pd.DataFrame({
            "temp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "pression": [np.nan, np.nan, np.nan, 1.0, 2.0],
            "shift": ["A", "A", "B", "B", "C"],
        })
        audit = audit_dataframe(df)
        assert len(audit) == 3
        assert set(audit["column"]) == {"temp", "pression", "shift"}
        assert "verdict" in audit.columns

    def test_audit_vide(self):
        audit = audit_dataframe(pd.DataFrame())
        assert audit.empty
        assert "column" in audit.columns  # structure preservee

    def test_audit_none_safe(self):
        audit = audit_dataframe(None)
        assert audit.empty


class TestSummary:
    def test_summary_mixte(self):
        df = pd.DataFrame({
            "ok_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "warn_col": [42.0] * 5,  # constante
            "danger_col": [np.nan] * 5,
        })
        summary = summary_stats(df)
        assert summary["n_columns"] == 3
        assert summary["n_rows"] == 5
        assert summary["n_ok"] == 1
        assert summary["n_warn"] == 1
        assert summary["n_danger"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
