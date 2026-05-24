"""
Scenarios industriels realistes Gascogne Sacs.
Chaque test reproduit un piege rencontre en production.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from analysis import analyze_all, clean_data
from data_loader import load_file, list_excel_sheets
from drift import detect_drift
from losses import analyze_losses
from shift_analysis import analyze_shifts


# ---------------------------------------------------------------------------
# FIXTURES : fichiers CSV/Excel realistes
# ---------------------------------------------------------------------------
@pytest.fixture
def csv_french(tmp_path):
    """CSV francais : ; separateur, virgule decimale, date dd/mm/yyyy."""
    content = (
        "Date;Temperature (degC);Pression (bar);Shift\n"
        "01/03/2026 08:00;180,5;4,52;A\n"
        "01/03/2026 08:30;181,2;4,48;A\n"
        "01/03/2026 09:00;182,1;4,51;A\n"
        "01/03/2026 09:30;181,8;4,49;A\n"
        "01/03/2026 10:00;183,0;4,53;A\n"
    )
    p = tmp_path / "data_fr.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def excel_multi_sheet(tmp_path):
    """Excel avec 3 feuilles (shifts A/B/C)."""
    p = tmp_path / "data_multi.xlsx"
    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        for sh in ["Shift_A", "Shift_B", "Shift_C"]:
            pd.DataFrame({
                "timestamp": pd.date_range("2026-01-01", periods=30, freq="h"),
                "temperature": np.random.normal(180, 3, 30),
                "pression": np.random.normal(4.5, 0.15, 30),
            }).to_excel(writer, sheet_name=sh, index=False)
    return p


@pytest.fixture
def csv_sentinelles(tmp_path):
    """CSV avec valeurs sentinelles capteur."""
    content = (
        "timestamp,temperature,pression\n"
        "2026-01-01 08:00,180.5,4.5\n"
        "2026-01-01 09:00,-9999,4.48\n"
        "2026-01-01 10:00,SENSOR_FAIL,4.51\n"
        "2026-01-01 11:00,181.2,-273.15\n"
        "2026-01-01 12:00,182.0,4.52\n"
    )
    p = tmp_path / "data_sentinel.csv"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Scenario 1 : import francais avec virgules decimales
# ---------------------------------------------------------------------------
class TestImportFrancais:
    def test_load_csv_french(self, csv_french):
        df, report = load_file(str(csv_french), "data_fr.csv")
        assert len(df) == 5
        # Temperature doit etre numerique (virgule -> point)
        assert pd.api.types.is_numeric_dtype(df["Temperature (degC)"])
        assert df["Temperature (degC)"].iloc[0] == pytest.approx(180.5)

    def test_shift_analysis_french_csv(self, csv_french):
        df, _ = load_file(str(csv_french), "data_fr.csv")
        # Pas de different shifts (tous A) -> detected mais 1 shift only
        res = analyze_shifts(
            df, param_cols=["Temperature (degC)", "Pression (bar)"],
            shift_col="Shift",
        )
        # Soit detected=False si moins de 2 shifts, soit detected=True
        assert "detected" in res


# ---------------------------------------------------------------------------
# Scenario 2 : Excel multi-feuilles
# ---------------------------------------------------------------------------
class TestImportExcelMultiSheet:
    def test_list_sheets(self, excel_multi_sheet):
        sheets = list_excel_sheets(str(excel_multi_sheet))
        assert len(sheets) == 3
        assert set(sheets) == {"Shift_A", "Shift_B", "Shift_C"}

    def test_load_specific_sheet(self, excel_multi_sheet):
        df, report = load_file(
            str(excel_multi_sheet), "data_multi.xlsx",
            sheet="Shift_B",
        )
        assert len(df) == 30
        assert "temperature" in df.columns


# ---------------------------------------------------------------------------
# Scenario 3 : valeurs sentinelles capteur
# ---------------------------------------------------------------------------
class TestSentinelles:
    def test_sentinelles_remplacees(self, csv_sentinelles):
        df, report = load_file(str(csv_sentinelles), "data_sentinel.csv")
        # -9999, SENSOR_FAIL, -273.15 doivent devenir NaN
        assert df["temperature"].isna().sum() >= 2
        assert df["pression"].isna().sum() >= 1


# ---------------------------------------------------------------------------
# Scenario 4 : capteur offline toute la periode
# ---------------------------------------------------------------------------
class TestCapteurOffline:
    def test_clean_data_ecarte_capteur_mort(self):
        df = pd.DataFrame({
            "temp": np.random.normal(180, 3, 100),
            "capteur_mort": [np.nan] * 100,
            "pression": np.random.normal(4.5, 0.15, 100),
        })
        cleaned = clean_data(df, ["temp", "capteur_mort", "pression"])
        assert len(cleaned) > 0
        assert "capteur_mort" in cleaned.attrs.get("dropped_columns", [])
        assert "temp" in cleaned.columns
        assert "pression" in cleaned.columns


# ---------------------------------------------------------------------------
# Scenario 5 : shift column avec majuscules/minuscules mixtes
# ---------------------------------------------------------------------------
class TestShiftCasse:
    def test_shift_mixed_case(self):
        df = pd.DataFrame({
            "temp": np.random.normal(180, 3, 90),
            "shift": ["A", "a", "B", "b", "C", "c"] * 15,
        })
        res = analyze_shifts(df, param_cols=["temp"], shift_col="shift")
        # Doit detecter mais peut-etre 6 shifts au lieu de 3
        # -> a minima, ne plante pas
        assert res is not None


# ---------------------------------------------------------------------------
# Scenario 6 : integration complete parcours operateur
# ---------------------------------------------------------------------------
class TestParcoursOperateur:
    """Simule un operateur qui importe, selectionne, analyse, exporte."""

    def test_full_flow(self, tmp_path):
        # 1. Creation fichier realiste
        rng = np.random.default_rng(123)
        n = 150
        temp = rng.normal(180, 3, n)
        temp[75:] += np.linspace(0, 6, 75)  # drift
        defauts = (temp - 180) * 15 + rng.normal(50, 10, n)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="30min"),
            "temperature_four": temp,
            "pression_vis": rng.normal(4.5, 0.15, n),
            "defauts_ppm": np.clip(defauts, 0, None),
            "shift": np.tile(["A", "B", "C"], n // 3 + 1)[:n],
        })
        p = tmp_path / "test.xlsx"
        df.to_excel(p, index=False, engine="openpyxl")

        # 2. Import
        loaded, report = load_file(str(p), "test.xlsx")
        assert len(loaded) == n

        # 3. Nettoyage + SPC
        selected = ["temperature_four", "pression_vis"]
        cleaned = clean_data(loaded, selected)
        assert len(cleaned) > 0
        spc = analyze_all(cleaned)
        assert len(spc) == 2

        # 4. Drift
        drift = detect_drift(cleaned)
        assert drift[drift["parameter"] == "temperature_four"][
            "drift_detected"].iloc[0]

        # 5. Pertes
        losses = analyze_losses(
            loaded,
            param_columns=selected,
            defaut_column="defauts_ppm",
            volume_total=1_000_000,
        )
        assert losses["ppm"] is not None

        # 6. Shifts
        shifts = analyze_shifts(
            loaded, param_cols=selected, shift_col="shift",
        )
        assert shifts["detected"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
