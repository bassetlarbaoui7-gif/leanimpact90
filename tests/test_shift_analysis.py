"""
Tests unitaires pour shift_analysis.py.

On teste :
  - Detection automatique de la colonne shift (par nom + par profil)
  - Score de stabilite par shift (CV)
  - Comparaison inter-shift (Kruskal-Wallis)
  - Robustesse : pas de shift, 1 seul shift, colonnes absentes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shift_analysis import (
    analyze_shifts,
    compare_shifts,
    compute_shift_stability,
    detect_shift_column,
)


# =============================================================================
# Detection de colonne shift
# =============================================================================
def test_detect_shift_par_nom_explicite():
    df = pd.DataFrame({
        "temp": [180, 181, 182, 183] * 10,
        "shift": ["A", "B", "C"] * 13 + ["A"],
    })
    assert detect_shift_column(df) == "shift"


def test_detect_shift_nom_equipe():
    df = pd.DataFrame({
        "valeur": range(30),
        "equipe": ["matin", "aprem", "nuit"] * 10,
    })
    assert detect_shift_column(df) == "equipe"


def test_detect_shift_par_profil_categoriel():
    """Colonne sans nom explicite mais avec profil A/B/C."""
    df = pd.DataFrame({
        "mesure": range(30),
        "groupe": ["A", "B", "C"] * 10,
    })
    assert detect_shift_column(df) == "groupe"


def test_detect_shift_aucune_colonne():
    df = pd.DataFrame({
        "temp": np.random.default_rng(0).normal(100, 1, 50),
        "pression": np.random.default_rng(1).normal(2, 0.1, 50),
    })
    assert detect_shift_column(df) is None


def test_detect_shift_ignore_numerique():
    """Une colonne numerique ne doit pas etre prise pour un shift."""
    df = pd.DataFrame({"valeurs": [1.2, 3.4, 5.6] * 10})
    assert detect_shift_column(df) is None


# =============================================================================
# Stabilite par shift (CV)
# =============================================================================
def test_compute_shift_stability_basique():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "shift": ["A"] * 30 + ["B"] * 30,
        # Shift A stable, Shift B instable
        "temp": np.concatenate([rng.normal(180, 1, 30), rng.normal(180, 5, 30)]),
    })
    stab = compute_shift_stability(df, "shift", ["temp"])
    assert len(stab) == 2
    cv_a = stab[stab["shift"] == "A"]["cv"].iloc[0]
    cv_b = stab[stab["shift"] == "B"]["cv"].iloc[0]
    # Shift B doit avoir un CV beaucoup plus eleve
    assert cv_b > cv_a * 2


def test_compute_shift_stability_n_correct():
    df = pd.DataFrame({
        "shift": ["A"] * 10 + ["B"] * 20,
        "temp": list(range(30)),
    })
    stab = compute_shift_stability(df, "shift", ["temp"])
    row_a = stab[stab["shift"] == "A"].iloc[0]
    row_b = stab[stab["shift"] == "B"].iloc[0]
    assert row_a["n"] == 10
    assert row_b["n"] == 20


# =============================================================================
# Comparaison inter-shift (Kruskal-Wallis)
# =============================================================================
def test_compare_shifts_ecart_significatif():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "shift": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        "temp": np.concatenate([
            rng.normal(180, 1, 50),
            rng.normal(185, 1, 50),   # shift B decale
            rng.normal(180, 1, 50),
        ]),
    })
    comp = compare_shifts(df, "shift", ["temp"])
    row = comp.iloc[0]
    assert row["parameter"] == "temp"
    assert bool(row["significatif_5pct"]) is True
    assert row["p_value"] < 0.05


def test_compare_shifts_pas_de_difference():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({
        "shift": ["A"] * 50 + ["B"] * 50,
        "temp": rng.normal(100, 1, 100),
    })
    comp = compare_shifts(df, "shift", ["temp"])
    assert bool(comp.iloc[0]["significatif_5pct"]) is False


def test_compare_shifts_un_seul_shift():
    df = pd.DataFrame({
        "shift": ["A"] * 20,
        "temp": range(20),
    })
    comp = compare_shifts(df, "shift", ["temp"])
    assert comp.empty


# =============================================================================
# Orchestrateur analyze_shifts
# =============================================================================
def test_analyze_shifts_cas_nominal():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "shift": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "temp": rng.normal(180, 2, 90),
    })
    res = analyze_shifts(df, ["temp"])
    assert res["detected"] is True
    assert res["shift_column"] == "shift"
    assert res["n_shifts"] == 3
    assert not res["stability"].empty
    assert not res["comparison"].empty


def test_analyze_shifts_pas_de_shift_retourne_dict_propre():
    df = pd.DataFrame({
        "temp": np.random.default_rng(0).normal(100, 1, 50),
    })
    res = analyze_shifts(df, ["temp"])
    assert res["detected"] is False
    assert res["shift_column"] is None
    assert "reason" in res
    assert res["stability"].empty


def test_analyze_shifts_un_seul_shift_signale_propre():
    """Quand on passe explicitement une colonne shift avec 1 seule modalite,
    analyze_shifts signale proprement l'impossibilite de comparer."""
    df = pd.DataFrame({
        "shift": ["A"] * 20,
        "temp": list(range(20)),
    })
    res = analyze_shifts(df, ["temp"], shift_col="shift")
    assert res["detected"] is False
    assert "modalite" in res["reason"].lower()
