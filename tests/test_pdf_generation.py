"""
Test de generation PDF avec donnees SPC + pertes realistes.
Verifie que le fichier se genere, est lisible, et contient les sections.
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
from losses import analyze_losses
from report import generate_pdf
from shift_analysis import analyze_shifts


@pytest.fixture
def scenario_complet(tmp_path):
    rng = np.random.default_rng(7)
    n = 120
    temp = rng.normal(180, 3, n)
    temp[60:] += 4
    defauts = np.clip((temp - 180) * 12 + rng.normal(40, 8, n), 0, None)
    df = pd.DataFrame({
        "temperature_four": temp,
        "pression_vis": rng.normal(4.5, 0.15, n),
        "humidite_kraft": rng.normal(7.5, 0.6, n),
        "defauts_ppm": defauts,
    })
    return df, tmp_path


class TestPDF:
    def test_generation_basique(self, scenario_complet):
        df, tmp = scenario_complet
        selected = ["temperature_four", "pression_vis", "humidite_kraft"]
        cleaned = clean_data(df, selected)
        results = analyze_all(cleaned)
        out = tmp / "rapport.pdf"
        generate_pdf(results, str(out), source_file="test.xlsx")
        assert out.exists()
        assert out.stat().st_size > 1000  # PDF non trivial

    def test_generation_avec_pertes(self, scenario_complet):
        df, tmp = scenario_complet
        selected = ["temperature_four", "pression_vis", "humidite_kraft"]
        cleaned = clean_data(df, selected)
        results = analyze_all(cleaned)
        losses = analyze_losses(
            df, param_columns=selected,
            defaut_column="defauts_ppm",
            volume_total=500_000,
        )
        out = tmp / "rapport_complet.pdf"
        generate_pdf(
            results, str(out),
            source_file="prod_mars.xlsx",
            losses_results=losses,
        )
        assert out.exists()
        assert out.stat().st_size > 1500

        # Test lisibilite : au moins le nombre de pages attendu
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(out))
            assert len(reader.pages) >= 1
            text = reader.pages[0].extract_text()
            assert "Rapport" in text or "LI90" in text
        except ImportError:
            # pypdf pas installe : on se contente de la taille
            pass

    def test_generation_avec_shifts(self, scenario_complet):
        """Rapport direction complet : SPC + pertes + shifts."""
        df, tmp = scenario_complet
        df = df.copy()
        df["shift"] = np.tile(["A", "B", "C"], len(df) // 3 + 1)[:len(df)]
        selected = ["temperature_four", "pression_vis"]
        cleaned = clean_data(df, selected)
        results = analyze_all(cleaned)
        losses = analyze_losses(
            df, param_columns=selected,
            defaut_column="defauts_ppm",
            volume_total=500_000,
        )
        shifts = analyze_shifts(df, param_cols=selected, shift_col="shift")
        out = tmp / "rapport_complet_shifts.pdf"
        generate_pdf(
            results, str(out),
            source_file="prod_mars.xlsx",
            losses_results=losses,
            shifts_results=shifts,
        )
        assert out.exists()
        assert out.stat().st_size > 2000

    def test_generation_nombreux_parametres(self, scenario_complet):
        """Simule 10 parametres (stress du tableau)."""
        df, tmp = scenario_complet
        rng = np.random.default_rng(9)
        for i in range(7):
            df[f"param_{i}"] = rng.normal(10, 1, len(df))
        selected = [c for c in df.columns if c != "defauts_ppm"]
        cleaned = clean_data(df, selected)
        results = analyze_all(cleaned)
        out = tmp / "many_params.pdf"
        generate_pdf(results, str(out))
        assert out.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
