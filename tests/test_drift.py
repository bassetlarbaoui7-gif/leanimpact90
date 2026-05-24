"""
Tests unitaires pour drift.py (EWMA + CUSUM).

On teste :
  - Detection d'une derive injectee (EWMA doit signaler)
  - Detection d'un saut de moyenne (CUSUM doit signaler plus tot)
  - Absence de signal sur processus stable
  - Robustesse : serie trop courte, serie constante, NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from drift import compute_cusum, compute_ewma, detect_drift


# =============================================================================
# EWMA
# =============================================================================
def test_ewma_derive_lente_detectee():
    """Derive de +1 sigma injectee sur la seconde moitie : EWMA doit signaler."""
    rng = np.random.default_rng(42)
    stable = rng.normal(100, 1, 100)
    drift = rng.normal(101.0, 1, 100)  # derive de 1 sigma, bien detectable
    values = np.concatenate([stable, drift])
    res = compute_ewma(values, lambda_=0.2, L=3.0)
    assert res["valid"] is True
    assert res["violations"] > 0
    # Le signal doit tomber majoritairement dans la zone de derive
    # (quelques points pre-drift acceptables a cause de l'inertie EWMA)
    assert res["first_violation_idx"] >= 90


def test_ewma_processus_stable_aucun_signal():
    """Processus sous controle : EWMA ne doit pas signaler (ou tres peu)."""
    rng = np.random.default_rng(0)
    values = rng.normal(100, 1, 200)
    res = compute_ewma(values, lambda_=0.2, L=3.0)
    assert res["valid"] is True
    # En theorie 0, en pratique 1-2 faux positifs tolerables sur 200 points
    assert res["violations"] <= 3


def test_ewma_serie_trop_courte_invalide():
    res = compute_ewma(np.array([1.0, 2.0, 3.0]))
    assert res["valid"] is False
    assert "n=3" in res["reason"]


def test_ewma_serie_constante_invalide():
    values = np.full(50, 100.0)
    res = compute_ewma(values)
    assert res["valid"] is False
    assert "constante" in res["reason"].lower()


# =============================================================================
# CUSUM
# =============================================================================
def test_cusum_saut_de_moyenne_detecte():
    """Saut brutal de moyenne : CUSUM doit signaler (up ou down peu importe,
    l'objectif est qu'une derive soit detectee). La moyenne de reference
    etant calculee globalement, un saut peut declencher les deux signaux
    en miroir — c'est un artefact connu du CUSUM sans baseline Phase I."""
    rng = np.random.default_rng(1)
    before = rng.normal(100, 1, 100)
    after = rng.normal(103, 1, 100)
    values = np.concatenate([before, after])
    res = compute_cusum(values, k=0.5, h=4.0)
    assert res["valid"] is True
    # Au moins un des deux signaux doit etre declenche
    assert res["signal_up"] > 0 or res["signal_down"] > 0
    assert res["first_signal_idx"] is not None


def test_cusum_processus_stable_aucun_signal():
    rng = np.random.default_rng(2)
    values = rng.normal(100, 1, 200)
    res = compute_cusum(values, k=0.5, h=4.0)
    assert res["valid"] is True
    assert res["signal_up"] == 0
    assert res["signal_down"] == 0


def test_cusum_derive_vers_bas():
    rng = np.random.default_rng(3)
    before = rng.normal(100, 1, 50)
    after = rng.normal(98, 1, 50)
    values = np.concatenate([before, after])
    res = compute_cusum(values)
    assert res["signal_down"] > 0


def test_cusum_serie_constante_invalide():
    res = compute_cusum(np.full(30, 50.0))
    assert res["valid"] is False


# =============================================================================
# Orchestrateur detect_drift
# =============================================================================
def test_detect_drift_dataframe_mixte():
    """DataFrame avec 1 colonne stable + 1 colonne qui derive clairement."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "stable": rng.normal(50, 1, 200),
        "derive": np.concatenate([rng.normal(50, 1, 100), rng.normal(52, 1, 100)]),
    })
    result = detect_drift(df)
    assert len(result) == 2
    derive_row = result[result["parameter"] == "derive"].iloc[0]
    # La colonne qui derive doit etre signalee
    assert bool(derive_row["drift_detected"]) is True


def test_detect_drift_ignore_colonne_texte():
    """detect_drift ne doit pas planter sur une colonne non-numerique."""
    df = pd.DataFrame({
        "num": np.random.default_rng(5).normal(10, 1, 50),
        "texte": ["A"] * 50,
    })
    # Par defaut : seules les colonnes numeriques sont traitees
    result = detect_drift(df)
    assert "num" in result["parameter"].values
    assert "texte" not in result["parameter"].values


def test_detect_drift_robuste_avec_nan():
    rng = np.random.default_rng(6)
    vals = rng.normal(100, 1, 100)
    vals[::10] = np.nan  # NaN disperses
    df = pd.DataFrame({"p": vals})
    result = detect_drift(df)
    # Ne doit pas planter, doit produire une ligne
    assert len(result) == 1


def test_detect_drift_colonne_invalide_ne_plante_pas():
    df = pd.DataFrame({"p": [1.0, 2.0, 3.0]})  # 3 points, trop court
    result = detect_drift(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert bool(row["drift_detected"]) is False
    assert "EWMA" in row["reason"] or "CUSUM" in row["reason"]
