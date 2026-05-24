"""Tests unitaires pour losses.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from losses import (
    analyze_losses,
    compute_correlation,
    compute_cpk,
    compute_ppm,
)


# -----------------------------------------------------------------------------
# PPM
# -----------------------------------------------------------------------------
def test_ppm_cas_nominal():
    # 50 defauts sur 10_000 unites -> 5000 PPM
    assert compute_ppm(50, 10_000) == pytest.approx(5000.0)


def test_ppm_volume_nul_rejete():
    with pytest.raises(ValueError, match="volume_total"):
        compute_ppm(10, 0)


def test_ppm_defauts_negatifs_rejetes():
    with pytest.raises(ValueError, match="defauts_total"):
        compute_ppm(-1, 1000)


# -----------------------------------------------------------------------------
# Cpk
# -----------------------------------------------------------------------------
def test_cpk_capable():
    rng = np.random.default_rng(42)
    # Process centre sur 100, std faible, specs larges -> capable
    s = pd.Series(rng.normal(100, 0.5, 500))
    res = compute_cpk(s, lsl=95, usl=105)
    assert res["verdict"] == "CAPABLE"
    assert res["cpk"] >= 1.33


def test_cpk_non_capable():
    rng = np.random.default_rng(42)
    # Process avec std grand par rapport aux specs -> non capable
    s = pd.Series(rng.normal(100, 5, 500))
    res = compute_cpk(s, lsl=95, usl=105)
    assert res["verdict"] == "NON_CAPABLE"
    assert res["cpk"] < 1.0


def test_cpk_constante_dans_specs():
    # vitesse_vis STRICTEMENT constante a 85 (np.full, aucun bruit).
    # std doit etre exactement 0, donc sous le seuil CONST_STD_THRESHOLD.
    s = pd.Series(np.full(500, 85.0))
    res = compute_cpk(s, lsl=80, usl=90)
    assert res["verdict"] == "CONSTANTE_DANS_SPECS"
    assert res["std"] < 1e-9
    assert math.isinf(res["cpk"])


def test_cpk_constante_hors_specs():
    # Constante a 120, specs [80, 90] -> hors specs.
    # Point critique : avant la correction, ce cas retournait CAPABLE
    # a cause d'un cpk=inf calcule sans verifier la position de la moyenne.
    s = pd.Series(np.full(500, 120.0))
    res = compute_cpk(s, lsl=80, usl=90)
    assert res["verdict"] == "CONSTANTE_HORS_SPECS"
    assert res["cpk"] == 0.0


def test_cpk_lsl_superieur_usl_rejete():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="lsl"):
        compute_cpk(s, lsl=10, usl=5)


# -----------------------------------------------------------------------------
# Correlation
# -----------------------------------------------------------------------------
def test_correlation_retient_pearson_sur_donnees_normales():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 10, 200)
    # Defauts correles lineairement au parametre
    y = 0.5 * x + rng.normal(0, 2, 200)
    res = compute_correlation(pd.Series(x), pd.Series(y), param_name="temp")
    assert res["test_retenu"] == "pearson"
    assert res["shapiro_p"] > 0.05
    assert abs(res["pearson_r"]) > 0.8


def test_correlation_retient_spearman_sur_donnees_non_normales():
    rng = np.random.default_rng(0)
    # Exponentielle : clairement non normale (Shapiro rejette).
    x = rng.exponential(1.0, 300)
    y = x + rng.normal(0, 0.1, 300)
    res = compute_correlation(pd.Series(x), pd.Series(y), param_name="pression")
    assert res["test_retenu"] == "spearman"
    assert res["shapiro_p"] < 0.05


def test_correlation_parametre_constant_leve_erreur():
    x = pd.Series(np.full(100, 85.0))  # strictement constante
    y = pd.Series(np.random.default_rng(0).normal(0, 1, 100))
    with pytest.raises(ValueError, match="constant"):
        compute_correlation(x, y, param_name="vitesse_vis")


# -----------------------------------------------------------------------------
# analyze_losses : orchestrateur
# -----------------------------------------------------------------------------
def test_analyze_losses_pipeline_complet():
    """
    Smoke test end-to-end avec :
      - temperature normale (Shapiro OK -> Pearson)
      - pression normale
      - vitesse_vis strictement constante (np.full, non calculable)
      - defaut en comptage ENTIER via Poisson (pas binaire 0/1)
    """
    rng = np.random.default_rng(7)
    n = 300
    volume_par_ligne = 1000

    temperature = rng.normal(180, 5, n)
    pression = rng.normal(2.5, 0.2, n)
    vitesse_vis = np.full(n, 85.0)  # CONSTANTE stricte, aucun bruit
    prob_defaut = np.clip((temperature - 180) / 200 + 0.02, 0, 0.2)
    defaut = rng.poisson(prob_defaut * volume_par_ligne / 100)

    df = pd.DataFrame({
        "temperature": temperature,
        "pression": pression,
        "vitesse_vis": vitesse_vis,
        "defaut": defaut,
    })

    res = analyze_losses(
        df,
        param_columns=["temperature", "pression", "vitesse_vis"],
        defaut_column="defaut",
        volume_total=n * volume_par_ligne,
        specs={
            "temperature": (170, 190),
            "vitesse_vis": (80, 90),
        },
    )

    # -- Correlations : 3 lignes, vitesse_vis doit etre NON_CALCULABLE
    corr = res["correlations"]
    assert len(corr) == 3
    vis_row = corr[corr["param"] == "vitesse_vis"].iloc[0]
    assert vis_row["test_retenu"] == "NON_CALCULABLE"
    assert "motif" in vis_row and "constant" in str(vis_row["motif"])
    others = corr[corr["param"] != "vitesse_vis"]
    assert (others["test_retenu"] != "NON_CALCULABLE").all()

    # -- Cpk : 2 lignes (temperature + vitesse_vis)
    cpk = res["cpk"]
    assert len(cpk) == 2
    vis_cpk = cpk[cpk["param"] == "vitesse_vis"].iloc[0]
    assert vis_cpk["verdict"] == "CONSTANTE_DANS_SPECS"

    # -- PPM : positif et fini (defauts integers Poisson)
    assert res["ppm"] is not None
    assert res["ppm"] > 0
    assert math.isfinite(res["ppm"])

    # -- Sanity : defaut est bien en comptage entier
    assert pd.api.types.is_integer_dtype(df["defaut"])


def test_analyze_losses_colonne_non_numerique_traitee_proprement():
    """
    Une colonne 100% non numerique doit produire NON_CALCULABLE
    (jamais de fillna(0) silencieux qui donnerait une correlation bidon).
    """
    df = pd.DataFrame({
        "param_texte": ["a", "b", "c", "d", "e"],
        "defaut": [0, 1, 2, 3, 4],
    })
    res = analyze_losses(df, param_columns=["param_texte"],
                        defaut_column="defaut")
    assert len(res["correlations"]) == 1
    row = res["correlations"].iloc[0]
    assert row["test_retenu"] == "NON_CALCULABLE"
    assert "motif" in row
