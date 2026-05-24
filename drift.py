"""
Detection de derives lentes et cumulatives.

Deux methodes complementaires a Shewhart (qui detecte les points extremes) :

- EWMA (Exponentially Weighted Moving Average, Roberts 1959) :
  detecte les petites derives de moyenne (ex : +0.5 sigma sur 50 points)
  que Shewhart rate completement.

- CUSUM (Cumulative Sum, Page 1954) :
  detecte la somme cumulative des ecarts a la cible. Signale plus tot
  qu'EWMA pour les derives brutales et soutenues.

Reference : Montgomery D.C., "Introduction to Statistical Quality Control",
chapitre 9. Parametres par defaut : lambda_=0.2, L=3 (EWMA) ; k=0.5, h=4 (CUSUM)
issus de la litterature (detection optimale pour derive de 1 sigma).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Parametres par defaut (standards industriels)
EWMA_LAMBDA = 0.2        # poids lissage (0 < lambda <= 1)
EWMA_L = 3.0             # largeur des limites (en sigma)
CUSUM_K = 0.5            # seuil de reference (en sigma)
CUSUM_H = 4.0            # seuil de decision (en sigma)


def compute_ewma(
    values: np.ndarray,
    lambda_: float = EWMA_LAMBDA,
    L: float = EWMA_L,
) -> dict:
    """
    Carte EWMA pour detection de derive lente.

    Retourne :
      - ewma   : serie EWMA calculee
      - ucl    : limite superieure (asymptotique)
      - lcl    : limite inferieure (asymptotique)
      - mean   : moyenne de reference
      - violations : nombre de points hors limites
      - first_violation_idx : index du premier signal (None si aucun)

    Condition : au moins 10 points pour etre significatif.
    """
    n = len(values)
    if n < 10:
        return {"ewma": np.array([]), "ucl": 0.0, "lcl": 0.0, "mean": 0.0,
                "violations": 0, "first_violation_idx": None,
                "valid": False, "reason": f"n={n} < 10"}

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std < 1e-12:
        return {"ewma": np.full(n, mean), "ucl": mean, "lcl": mean,
                "mean": mean, "violations": 0, "first_violation_idx": None,
                "valid": False, "reason": "ecart-type nul (serie constante)"}

    # Calcul EWMA
    ewma = np.empty(n)
    ewma[0] = mean
    for i in range(1, n):
        ewma[i] = lambda_ * values[i] + (1 - lambda_) * ewma[i - 1]

    # Limites asymptotiques (approximation pour i grand)
    sigma_z = std * np.sqrt(lambda_ / (2 - lambda_))
    ucl = mean + L * sigma_z
    lcl = mean - L * sigma_z

    out_of_control = (ewma > ucl) | (ewma < lcl)
    violations = int(np.sum(out_of_control))
    first_idx = int(np.argmax(out_of_control)) if violations > 0 else None

    return {
        "ewma": ewma,
        "ucl": ucl,
        "lcl": lcl,
        "mean": mean,
        "violations": violations,
        "first_violation_idx": first_idx,
        "valid": True,
        "reason": "",
    }


def compute_cusum(
    values: np.ndarray,
    k: float = CUSUM_K,
    h: float = CUSUM_H,
) -> dict:
    """
    CUSUM pour detection de derive cumulative.

    k et h sont en unites de sigma (standard Montgomery).

    Retourne :
      - c_plus, c_minus : series cumulatives
      - signal_up   : nombre de signaux de derive positive
      - signal_down : nombre de signaux de derive negative
      - first_signal_idx : premier index ou C+ ou C- depasse h
    """
    n = len(values)
    if n < 10:
        return {"c_plus": np.array([]), "c_minus": np.array([]),
                "signal_up": 0, "signal_down": 0, "first_signal_idx": None,
                "valid": False, "reason": f"n={n} < 10"}

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std < 1e-12:
        return {"c_plus": np.zeros(n), "c_minus": np.zeros(n),
                "signal_up": 0, "signal_down": 0, "first_signal_idx": None,
                "valid": False, "reason": "ecart-type nul (serie constante)"}

    K = k * std  # seuil de reference absolu
    H = h * std  # seuil de decision absolu

    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    for i in range(1, n):
        c_plus[i] = max(0.0, c_plus[i - 1] + values[i] - mean - K)
        c_minus[i] = max(0.0, c_minus[i - 1] + mean - K - values[i])

    signal_up_mask = c_plus > H
    signal_down_mask = c_minus > H
    signal_up = int(np.sum(signal_up_mask))
    signal_down = int(np.sum(signal_down_mask))

    first_idx = None
    any_signal = signal_up_mask | signal_down_mask
    if any_signal.any():
        first_idx = int(np.argmax(any_signal))

    return {
        "c_plus": c_plus,
        "c_minus": c_minus,
        "mean": mean,
        "H": H,
        "signal_up": signal_up,
        "signal_down": signal_down,
        "first_signal_idx": first_idx,
        "valid": True,
        "reason": "",
    }


def detect_drift(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Applique EWMA + CUSUM sur chaque colonne numerique.

    Retourne un DataFrame : une ligne par parametre, colonnes =
      parameter, ewma_violations, cusum_signal_up, cusum_signal_down,
      ewma_first_idx, cusum_first_idx, drift_detected, reason.

    Robuste : ne leve pas d'exception. Les colonnes invalides sont
    reportees avec reason rempli mais drift_detected=False.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for col in columns:
        try:
            values = df[col].dropna().to_numpy()
            ewma_res = compute_ewma(values)
            cusum_res = compute_cusum(values)

            drift = (ewma_res.get("violations", 0) > 0
                     or cusum_res.get("signal_up", 0) > 0
                     or cusum_res.get("signal_down", 0) > 0)

            reason_parts = []
            if not ewma_res.get("valid", False):
                reason_parts.append(f"EWMA: {ewma_res.get('reason', '')}")
            if not cusum_res.get("valid", False):
                reason_parts.append(f"CUSUM: {cusum_res.get('reason', '')}")

            rows.append({
                "parameter": col,
                "ewma_violations": ewma_res.get("violations", 0),
                "cusum_signal_up": cusum_res.get("signal_up", 0),
                "cusum_signal_down": cusum_res.get("signal_down", 0),
                "ewma_first_idx": ewma_res.get("first_violation_idx"),
                "cusum_first_idx": cusum_res.get("first_signal_idx"),
                "drift_detected": bool(drift),
                "reason": " | ".join(reason_parts),
            })
        except Exception as e:
            # Principe robustesse : on log, on continue
            rows.append({
                "parameter": col,
                "ewma_violations": 0,
                "cusum_signal_up": 0,
                "cusum_signal_down": 0,
                "ewma_first_idx": None,
                "cusum_first_idx": None,
                "drift_detected": False,
                "reason": f"erreur calcul: {e}",
            })

    return pd.DataFrame(rows)
