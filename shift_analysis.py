"""
Analyse de la variabilite inter-shift (equipes A/B/C).

Priorite #1 pour Gascogne : 3 shifts = 3 facons de faire.
L'ecart entre shifts est tres souvent la premiere source de variation
d'une ligne ancienne. Detecter et quantifier cet ecart permet de :
  - cibler la formation du shift le plus instable
  - homogeneiser les pratiques (Standard Work, Liker ch.12)
  - proteger contre les "heros de shift" (un operateur qui fait tourner
    la ligne differemment des autres).

Methodes utilisees :
  - Detection de la colonne shift (nom + valeurs)
  - Score de stabilite : CV (coefficient of variation) par shift
  - Comparaison inter-shift : Kruskal-Wallis (non parametrique, robuste)

Robuste : si pas de colonne shift, retourne un resultat vide sans planter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SHIFT_NAME_CANDIDATES = {
    "shift", "shifts", "equipe", "team", "poste", "quart", "rotation",
}

# Un shift a typiquement 2 a 8 modalites distinctes (A/B/C, 1/2/3, matin/aprem/nuit...)
SHIFT_MAX_UNIQUE = 8
SHIFT_MIN_UNIQUE = 2


def detect_shift_column(df: pd.DataFrame) -> str | None:
    """
    Detecte automatiquement une colonne representant le shift.

    Heuristique :
      1. Le nom de la colonne contient un mot-cle (shift, equipe, poste, ...)
      2. OU la colonne est categorielle avec 2 a 8 valeurs uniques courtes
    """
    # Priorite 1 : nom explicite
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(kw in col_lower for kw in SHIFT_NAME_CANDIDATES):
            n_unique = df[col].dropna().nunique()
            if SHIFT_MIN_UNIQUE <= n_unique <= SHIFT_MAX_UNIQUE:
                return str(col)

    # Priorite 2 : profil categoriel plausible
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue  # on ne veut pas les parametres numeriques
        n_unique = s.nunique()
        if not (SHIFT_MIN_UNIQUE <= n_unique <= SHIFT_MAX_UNIQUE):
            continue
        # Toutes les modalites doivent etre courtes (A, B, C, ou "matin"...)
        max_len = s.astype(str).str.len().max()
        if max_len <= 15:
            return str(col)

    return None


def compute_shift_stability(
    df: pd.DataFrame,
    shift_col: str,
    param_cols: list[str],
) -> pd.DataFrame:
    """
    Pour chaque (shift, parametre), calcule moyenne, ecart-type, et CV.

    CV (coefficient of variation) = std / |mean|. C'est le score
    d'instabilite : plus le CV est grand, plus le shift est instable
    sur ce parametre.

    Retourne un DataFrame long : shift, parameter, n, mean, std, cv.
    """
    if shift_col not in df.columns:
        return pd.DataFrame(columns=["shift", "parameter", "n", "mean", "std", "cv"])

    rows = []
    shifts = df[shift_col].dropna().unique()
    for shift_val in shifts:
        sub = df[df[shift_col] == shift_val]
        for param in param_cols:
            if param not in sub.columns:
                continue
            vals = pd.to_numeric(sub[param], errors="coerce").dropna().to_numpy()
            n = len(vals)
            if n == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            cv = float(std / abs(mean)) if abs(mean) > 1e-12 else np.nan
            rows.append({
                "shift": str(shift_val),
                "parameter": param,
                "n": n,
                "mean": mean,
                "std": std,
                "cv": cv,
            })

    return pd.DataFrame(rows)


def compare_shifts(
    df: pd.DataFrame,
    shift_col: str,
    param_cols: list[str],
) -> pd.DataFrame:
    """
    Pour chaque parametre, test de Kruskal-Wallis entre shifts.

    Kruskal-Wallis (non parametrique) est choisi plutot qu'ANOVA
    car il ne suppose pas la normalite des donnees — hypothese
    rarement verifiee en production industrielle.

    Retourne : parameter, n_shifts, H_statistic, p_value, significatif_5pct.
    """
    rows = []
    shifts = df[shift_col].dropna().unique()
    if len(shifts) < 2:
        return pd.DataFrame(columns=["parameter", "n_shifts", "H_statistic",
                                     "p_value", "significatif_5pct"])

    for param in param_cols:
        if param not in df.columns:
            continue
        groups = []
        for shift_val in shifts:
            sub = df[df[shift_col] == shift_val]
            vals = pd.to_numeric(sub[param], errors="coerce").dropna().to_numpy()
            if len(vals) >= 3:  # Kruskal necessite un minimum par groupe
                groups.append(vals)

        if len(groups) < 2:
            rows.append({"parameter": param, "n_shifts": len(groups),
                         "H_statistic": np.nan, "p_value": np.nan,
                         "significatif_5pct": False})
            continue

        try:
            h_stat, p_val = stats.kruskal(*groups)
            rows.append({
                "parameter": param,
                "n_shifts": len(groups),
                "H_statistic": float(h_stat),
                "p_value": float(p_val),
                "significatif_5pct": bool(p_val < 0.05),
            })
        except Exception:
            rows.append({"parameter": param, "n_shifts": len(groups),
                         "H_statistic": np.nan, "p_value": np.nan,
                         "significatif_5pct": False})

    return pd.DataFrame(rows)


def analyze_shifts(
    df: pd.DataFrame,
    param_cols: list[str],
    shift_col: str | None = None,
) -> dict:
    """
    Orchestrateur : detecte la colonne shift si non fournie, puis
    calcule stabilite et comparaison inter-shift.

    Retourne toujours un dict (jamais None) pour simplifier la gestion
    en aval. Si aucun shift detecte, on a un dict avec detected=False.
    """
    result: dict = {
        "detected": False,
        "shift_column": None,
        "stability": pd.DataFrame(),
        "comparison": pd.DataFrame(),
        "n_shifts": 0,
        "reason": "",
    }

    if shift_col is None:
        shift_col = detect_shift_column(df)

    if shift_col is None:
        result["reason"] = "Aucune colonne shift detectee."
        return result

    n_shifts = df[shift_col].dropna().nunique()
    if n_shifts < 2:
        result["reason"] = (
            f"Colonne '{shift_col}' detectee mais {n_shifts} modalite(s) "
            f"uniquement — comparaison impossible."
        )
        return result

    try:
        stability = compute_shift_stability(df, shift_col, param_cols)
        comparison = compare_shifts(df, shift_col, param_cols)
    except Exception as e:
        result["reason"] = f"Erreur analyse shift : {e}"
        return result

    result.update({
        "detected": True,
        "shift_column": shift_col,
        "stability": stability,
        "comparison": comparison,
        "n_shifts": int(n_shifts),
    })
    return result
