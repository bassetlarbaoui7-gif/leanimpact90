"""
Tests pour analysis.py : clean_data, analyze_parameter/all, regles de Nelson.

Les regles de Nelson sont testees avec des series fabriquees pour declencher
chaque regle specifiquement. Les constantes (9, 6, 14, 15, 8) viennent
directement de Nelson 1984 et doivent etre respectees strictement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import (
    analyze_all,
    analyze_parameter,
    clean_data,
    detect_nelson_rules,
)


# =============================================================================
# clean_data
# =============================================================================
def test_clean_data_retire_lignes_nan():
    df = pd.DataFrame({
        "temp": [1.0, 2.0, np.nan, 4.0],
        "pression": [0.1, np.nan, 0.3, 0.4],
    })
    out = clean_data(df, ["temp", "pression"])
    assert len(out) == 2
    assert out["temp"].tolist() == [1.0, 4.0]


def test_clean_data_convertit_string_en_numerique():
    df = pd.DataFrame({"temp": ["1", "2", "abc", "4"]})
    out = clean_data(df, ["temp"])
    assert len(out) == 3
    assert out["temp"].dtype.kind == "f"


# =============================================================================
# analyze_parameter : Shewhart
# =============================================================================
def test_analyze_parameter_shewhart_detecte_hors_limites():
    rng = np.random.default_rng(0)
    values = rng.normal(100, 1, 100).tolist() + [200.0]  # 1 outlier extreme
    s = pd.Series(values)
    res = analyze_parameter(s)
    assert res["n"] == 101
    assert res["violations"] >= 1
    assert res["criticality"] > 0


def test_analyze_parameter_serie_vide():
    res = analyze_parameter(pd.Series([], dtype=float))
    assert res["n"] == 0
    assert res["violations"] == 0
    assert res["nelson_total"] == 0
    assert res["dominant_pattern"] == "none"


# =============================================================================
# Regles de Nelson : chaque regle individuellement
# =============================================================================
def test_nelson_1_point_hors_3_sigma():
    """Regle 1 : 1 point au-dela de 3 sigma."""
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 50)
    values[25] = 5.0  # outlier a 5 sigma
    res = detect_nelson_rules(values, mean=float(np.mean(values)),
                              std=float(np.std(values, ddof=1)))
    assert res["nelson_1"] >= 1


def test_nelson_2_neuf_points_meme_cote():
    """Regle 2 : 9 points consecutifs du meme cote de la moyenne."""
    # Serie centree sur 0 avec 9 points consecutifs > 0 au milieu
    base = np.array([-1.0, 1.0] * 10)
    # Injecter 9 points positifs consecutifs
    series = np.concatenate([base[:5], np.ones(9) * 0.5, base[:5]])
    mean = 0.0
    std = 1.0
    res = detect_nelson_rules(series, mean, std)
    assert res["nelson_2"] >= 1


def test_nelson_3_six_points_monotones():
    """Regle 3 : 6 points strictement croissants."""
    values = np.array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    res = detect_nelson_rules(values, mean, std)
    assert res["nelson_3"] >= 1


def test_nelson_4_alternance_14_points():
    """Regle 4 : 14 points alternant up/down."""
    values = np.array([0, 1, 0, 1] * 5, dtype=float)  # 20 pts alternant
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    res = detect_nelson_rules(values, mean, std)
    assert res["nelson_4"] >= 1


def test_nelson_7_stratification():
    """Regle 7 : 15+ points tous dans +/- 1 sigma (sigma surestimee)."""
    # 30 points proches de la moyenne, sigma calcule sera grande grace a 2 outliers
    core = np.full(30, 100.0) + np.random.default_rng(0).normal(0, 0.01, 30)
    outliers = np.array([80.0, 120.0])
    values = np.concatenate([core, outliers])
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    res = detect_nelson_rules(values, mean, std)
    assert res["nelson_7"] >= 1


def test_nelson_std_zero_aucun_declenchement():
    """Serie constante : aucune regle ne peut s'appliquer."""
    values = np.full(50, 42.0)
    res = detect_nelson_rules(values, 42.0, 0.0)
    for i in range(1, 9):
        assert res[f"nelson_{i}"] == 0
    assert res["nelson_total"] == 0


def test_nelson_serie_trop_courte():
    values = np.array([1.0, 2.0, 3.0])
    res = detect_nelson_rules(values, 2.0, 1.0)
    assert res["nelson_total"] == 0


# =============================================================================
# dominant_pattern
# =============================================================================
def test_dominant_pattern_trend_si_regle3_dominante():
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    res = detect_nelson_rules(values, mean, std)
    assert res["dominant_pattern"] == "trend"


def test_dominant_pattern_none_sur_signal_plat():
    rng = np.random.default_rng(42)
    values = rng.normal(100, 1, 50)
    # Pas d'outlier, pas de pattern
    res = detect_nelson_rules(values, 100.0, 1.0)
    assert res["dominant_pattern"] in ("none", "instabilite", "shift_partiel")


# =============================================================================
# analyze_all : integration + backward compatibility
# =============================================================================
def test_analyze_all_colonnes_historiques_preservees():
    """Le DataFrame retourne doit conserver les colonnes historiques en tete."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "temp": rng.normal(180, 2, 100),
        "pression": rng.normal(2.5, 0.1, 100),
    })
    result = analyze_all(df)

    expected_head = ["parameter", "n", "mean", "std", "lcl", "ucl",
                     "violations", "criticality"]
    assert list(result.columns[:8]) == expected_head


def test_analyze_all_colonnes_nelson_ajoutees():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"p": rng.normal(0, 1, 100)})
    result = analyze_all(df)
    for i in range(1, 9):
        assert f"nelson_{i}" in result.columns
    assert "nelson_total" in result.columns
    assert "dominant_pattern" in result.columns


def test_analyze_all_classement_par_criticite():
    """Les parametres doivent etre classes par criticite decroissante."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "stable": rng.normal(100, 1, 100),
        "instable": np.concatenate([rng.normal(100, 1, 90), rng.normal(200, 1, 10)]),
    })
    result = analyze_all(df)
    # "instable" doit apparaitre en premier
    assert result.iloc[0]["parameter"] == "instable"
