"""
Analyse des pertes : correlation parametres/defauts, capabilite Cpk, PPM.
MVP v0.1 - simple, explicite, sans abstractions inutiles.

Regles non negociables :
- Aucun fillna(0) silencieux : une colonne non numerique leve ValueError.
- Les erreurs par colonne ne tuent pas l'orchestrateur : on remplit
  test_retenu = NON_CALCULABLE et on continue.
- Cpk avec std = 0 : on distingue explicitement CONSTANTE_DANS_SPECS
  et CONSTANTE_HORS_SPECS (jamais CAPABLE par accident).
- Verdict de correlation : Pearson si normalite non rejetee (Shapiro),
  sinon Spearman. Choix base sur la donnee, pas hardcode.
"""
from __future__ import annotations

import math

import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# Constantes metier
# -----------------------------------------------------------------------------
CONST_STD_THRESHOLD = 1e-9       # en dessous : colonne consideree constante
NORMALITY_P_THRESHOLD = 0.05     # seuil Shapiro-Wilk pour rejeter la normalite
CPK_CAPABLE = 1.33               # reference Six Sigma
CPK_MARGINAL = 1.0
SHAPIRO_MAX_N = 5000             # au-dela, Shapiro devient trop puissant


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _require_numeric(series: pd.Series, name: str) -> pd.Series:
    """
    Convertit une Serie en numerique. Erreur claire si non convertible.
    Ne remplace JAMAIS silencieusement par 0.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all() and not series.isna().all():
        exemple = series.dropna().iloc[0]
        raise ValueError(
            f"Colonne '{name}' non convertible en numerique "
            f"(exemple de valeur : {exemple!r})."
        )
    return s.dropna()


# -----------------------------------------------------------------------------
# PPM
# -----------------------------------------------------------------------------
def compute_ppm(defauts_total: float, volume_total: float) -> float:
    """PPM = defauts / volume * 1_000_000."""
    if volume_total <= 0:
        raise ValueError("volume_total doit etre strictement positif.")
    if defauts_total < 0:
        raise ValueError("defauts_total ne peut pas etre negatif.")
    return (float(defauts_total) / float(volume_total)) * 1_000_000.0


# -----------------------------------------------------------------------------
# Cpk
# -----------------------------------------------------------------------------
def compute_cpk(series: pd.Series, lsl: float, usl: float,
                name: str = "parametre") -> dict:
    """
    Indice de capabilite Cpk avec gestion explicite du cas std = 0.

    Verdicts possibles :
      CAPABLE                 (cpk >= 1.33)
      MARGINAL                (1.0 <= cpk < 1.33)
      NON_CAPABLE             (cpk < 1.0)
      CONSTANTE_DANS_SPECS    (std ~ 0 ET moyenne dans [lsl, usl])
      CONSTANTE_HORS_SPECS    (std ~ 0 ET moyenne hors [lsl, usl])
      DONNEES_INSUFFISANTES   (moins de 2 points)
    """
    if lsl >= usl:
        raise ValueError(
            f"lsl ({lsl}) doit etre strictement inferieur a usl ({usl}).")

    s = _require_numeric(series, name)
    n = len(s)
    if n < 2:
        return {"n": n, "mean": None, "std": None, "cpk": None,
                "verdict": "DONNEES_INSUFFISANTES"}

    mean = float(s.mean())
    std = float(s.std(ddof=1))

    if std < CONST_STD_THRESHOLD:
        # Cpk mathematiquement indefini : on distingue en fonction
        # de la position de la moyenne par rapport aux specs.
        in_specs = (lsl <= mean <= usl)
        return {
            "n": n, "mean": mean, "std": std,
            "cpk": math.inf if in_specs else 0.0,
            "verdict": "CONSTANTE_DANS_SPECS" if in_specs
                       else "CONSTANTE_HORS_SPECS",
        }

    cpu = (usl - mean) / (3.0 * std)
    cpl = (mean - lsl) / (3.0 * std)
    cpk = float(min(cpu, cpl))

    if cpk >= CPK_CAPABLE:
        verdict = "CAPABLE"
    elif cpk >= CPK_MARGINAL:
        verdict = "MARGINAL"
    else:
        verdict = "NON_CAPABLE"

    return {"n": n, "mean": mean, "std": std, "cpk": cpk, "verdict": verdict}


# -----------------------------------------------------------------------------
# Correlation
# -----------------------------------------------------------------------------
def compute_correlation(param: pd.Series, defaut: pd.Series,
                        param_name: str = "param") -> dict:
    """
    Pearson + Spearman + Shapiro-Wilk (sur le parametre).
    Choix du test retenu :
      - Pearson si Shapiro p-value > 0.05 (normalite non rejetee)
      - Spearman sinon (y compris si Shapiro n'a pas pu etre calcule)

    Leve ValueError si les donnees ne sont pas exploitables :
      parametre ou defauts constants, moins de 3 points communs, etc.
    """
    p = _require_numeric(param, param_name)
    d = _require_numeric(defaut, "defaut")

    joined = pd.concat([p.rename("p"), d.rename("d")],
                       axis=1, join="inner").dropna()
    if len(joined) < 3:
        raise ValueError(
            f"'{param_name}' : moins de 3 observations communes exploitables.")

    if joined["p"].std(ddof=1) < CONST_STD_THRESHOLD:
        raise ValueError(
            f"'{param_name}' : parametre constant, correlation non definie.")
    if joined["d"].std(ddof=1) < CONST_STD_THRESHOLD:
        raise ValueError(
            f"'{param_name}' : defauts constants, correlation non definie.")

    pearson_r, pearson_p = stats.pearsonr(joined["p"], joined["d"])
    spearman_res = stats.spearmanr(joined["p"], joined["d"])
    # Compat scipy : selon version, resultat tuple ou objet avec .statistic/.pvalue
    _s_stat = getattr(spearman_res, "statistic", None)
    _s_pval = getattr(spearman_res, "pvalue", None)
    spearman_r = float(_s_stat if _s_stat is not None else spearman_res[0])
    spearman_p = float(_s_pval if _s_pval is not None else spearman_res[1])

    if 3 <= len(joined) <= SHAPIRO_MAX_N:
        _, shapiro_p_raw = stats.shapiro(joined["p"])
        shapiro_p = float(shapiro_p_raw)
    else:
        shapiro_p = float("nan")

    # Choix du test : base sur la donnee, pas hardcode.
    if not math.isnan(shapiro_p) and shapiro_p > NORMALITY_P_THRESHOLD:
        chosen = "pearson"
        chosen_r, chosen_p = float(pearson_r), float(pearson_p)
    else:
        chosen = "spearman"
        chosen_r, chosen_p = spearman_r, spearman_p

    return {
        "param": param_name,
        "n": int(len(joined)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "shapiro_p": shapiro_p,
        "test_retenu": chosen,
        "r": chosen_r,
        "p_value": chosen_p,
    }


# -----------------------------------------------------------------------------
# Orchestrateur
# -----------------------------------------------------------------------------
def analyze_losses(
    df: pd.DataFrame,
    param_columns: list[str],
    defaut_column: str,
    volume_total: float | None = None,
    specs: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """
    Pipeline complet d'analyse des pertes.

    Retourne un dict avec :
      - 'correlations' : DataFrame trie par |r| descendant
      - 'cpk' : DataFrame (une ligne par parametre present dans specs)
      - 'ppm' : float ou None (si volume_total non fourni)

    Les erreurs par colonne ne tuent pas l'orchestrateur :
    la ligne concernee porte test_retenu = NON_CALCULABLE avec 'motif'.
    """
    if defaut_column not in df.columns:
        raise ValueError(
            f"Colonne defaut '{defaut_column}' absente du DataFrame.")
    missing = [c for c in param_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes parametres absentes : {missing}")

    specs = specs or {}

    # -- Correlations (tolerant aux erreurs par colonne)
    corr_rows = []
    for col in param_columns:
        try:
            corr_rows.append(
                compute_correlation(df[col], df[defaut_column], param_name=col))
        except ValueError as exc:
            corr_rows.append({
                "param": col,
                "n": 0,
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_r": float("nan"),
                "spearman_p": float("nan"),
                "shapiro_p": float("nan"),
                "test_retenu": "NON_CALCULABLE",
                "r": float("nan"),
                "p_value": float("nan"),
                "motif": str(exc),
            })

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["abs_r"] = corr_df["r"].abs()
        corr_df = (corr_df
                   .sort_values("abs_r", ascending=False, na_position="last")
                   .drop(columns=["abs_r"])
                   .reset_index(drop=True))

    # -- Cpk (tolerant par colonne)
    cpk_rows = []
    for col, (lsl, usl) in specs.items():
        if col not in df.columns:
            cpk_rows.append({"param": col, "lsl": lsl, "usl": usl,
                             "n": 0, "mean": None, "std": None, "cpk": None,
                             "verdict": "COLONNE_ABSENTE"})
            continue
        try:
            res = compute_cpk(df[col], lsl, usl, name=col)
            cpk_rows.append({"param": col, "lsl": lsl, "usl": usl, **res})
        except ValueError as exc:
            cpk_rows.append({"param": col, "lsl": lsl, "usl": usl,
                             "n": 0, "mean": None, "std": None, "cpk": None,
                             "verdict": f"ERREUR: {exc}"})
    cpk_df = (pd.DataFrame(cpk_rows)
              if cpk_rows
              else pd.DataFrame(columns=["param", "lsl", "usl", "n", "mean",
                                         "std", "cpk", "verdict"]))

    # -- PPM global
    ppm: float | None = None
    if volume_total is not None:
        defauts_series = _require_numeric(df[defaut_column], defaut_column)
        defauts_total = float(defauts_series.sum())
        ppm = compute_ppm(defauts_total, float(volume_total))

    return {"correlations": corr_df, "cpk": cpk_df, "ppm": ppm}
