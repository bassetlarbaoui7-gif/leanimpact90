"""
Nettoyage des donnees + detection de derives + classement.

Methodes :
  - Shewhart 3-sigma (Montgomery ch.5) : points hors limites
  - Regles de Nelson (Nelson 1984) : 8 patterns de causes speciales
    (trend, shift, stratification, over-control, ...) qui completent
    Shewhart — lequel rate 80% des derives lentes.

Le DataFrame retourne est additif : les colonnes Shewhart historiques
sont preservees, on ajoute seulement des colonnes nelson_* et
dominant_pattern pour l'interpretation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Mapping regle Nelson -> categorie industrielle (pour interpretation)
NELSON_PATTERN_TYPE = {
    1: "hors_controle",    # 1 point au-dela de 3 sigma
    2: "shift",            # 9 points du meme cote (decalage de moyenne)
    3: "trend",            # 6 points monotones (derive)
    4: "instabilite",      # 14 points alternant
    5: "shift_partiel",    # 2 sur 3 au-dela de 2 sigma
    6: "shift_partiel",    # 4 sur 5 au-dela de 1 sigma
    7: "stratification",   # 15 points dans +/-1 sigma (sous-estimation sigma)
    8: "instabilite",      # 8 points au-dela de 1 sigma (les deux cotes)
}


def clean_data(
    df: pd.DataFrame,
    columns: list[str],
    min_valid_ratio: float = 0.05,
) -> pd.DataFrame:
    """Garde les colonnes choisies, force en numerique, supprime les lignes vides.

    Defense industrielle : les colonnes dont moins de `min_valid_ratio` des
    valeurs sont numeriques (ex. capteur offline, colonne texte) sont
    ecartees AVANT le dropna pour eviter qu'un seul mauvais canal ne detruise
    tout le jeu de donnees.

    Le nom des colonnes ecartees est annote en attribut `.dropped_columns`
    sur le DataFrame retourne pour que l'UI puisse les afficher.
    """
    if not columns:
        raise ValueError("Aucune colonne selectionnee.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes absentes : {missing}")

    working = df[columns].copy()
    for col in columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    n_total = max(len(working), 1)
    dropped: list[str] = []
    keep: list[str] = []
    for col in columns:
        valid_ratio = working[col].notna().sum() / n_total
        if valid_ratio < min_valid_ratio:
            dropped.append(col)
        else:
            keep.append(col)

    cleaned = working[keep].dropna() if keep else working.iloc[0:0]
    # Metadonnees pour l'UI (sans casser l'API : c'est un DataFrame)
    try:
        cleaned.attrs["dropped_columns"] = dropped
    except Exception:
        pass
    return cleaned


# -----------------------------------------------------------------------------
# Regles de Nelson
# -----------------------------------------------------------------------------

def detect_nelson_rules(values: np.ndarray, mean: float, std: float) -> dict:
    """
    Applique les 8 regles de Nelson sur une serie.

    Retourne un dict {nelson_1: count, ..., nelson_8: count, total: N,
    dominant_pattern: str}.

    Si std == 0 (serie constante), aucune regle ne peut s'appliquer
    (la notion de sigma n'a pas de sens) -> tout a 0.
    """
    n = len(values)
    result = {f"nelson_{i}": 0 for i in range(1, 9)}
    result["nelson_total"] = 0
    result["dominant_pattern"] = "none"

    if n < 9 or std < 1e-12:
        return result

    # Normalisation en sigma
    z = (values - mean) / std
    above = z > 0  # cote positif

    # Regle 1 : 1 point au-dela de 3 sigma
    result["nelson_1"] = int(np.sum(np.abs(z) > 3))

    # Regle 2 : 9 points d'affilee du meme cote
    result["nelson_2"] = _count_consecutive_same_side(above, 9)

    # Regle 3 : 6 points monotones (croissants ou decroissants)
    result["nelson_3"] = _count_monotonic(values, 6)

    # Regle 4 : 14 points alternant up/down
    result["nelson_4"] = _count_alternating(values, 14)

    # Regle 5 : 2 sur 3 au-dela de 2 sigma du meme cote
    result["nelson_5"] = _count_k_of_n_beyond(z, k=2, n_win=3, threshold=2.0)

    # Regle 6 : 4 sur 5 au-dela de 1 sigma du meme cote
    result["nelson_6"] = _count_k_of_n_beyond(z, k=4, n_win=5, threshold=1.0)

    # Regle 7 : 15 points d'affilee dans +/- 1 sigma (hugging centerline)
    inside_1sigma = np.abs(z) < 1.0
    result["nelson_7"] = _count_consecutive_true(inside_1sigma, 15)

    # Regle 8 : 8 points d'affilee tous au-dela de +/- 1 sigma (des deux cotes)
    outside_1sigma = np.abs(z) > 1.0
    result["nelson_8"] = _count_consecutive_true(outside_1sigma, 8)

    total = sum(result[f"nelson_{i}"] for i in range(1, 9))
    result["nelson_total"] = int(total)

    # Pattern dominant : la regle avec le plus grand nombre de declenchements
    if total > 0:
        best_rule = max(range(1, 9), key=lambda i: result[f"nelson_{i}"])
        if result[f"nelson_{best_rule}"] > 0:
            result["dominant_pattern"] = NELSON_PATTERN_TYPE[best_rule]

    return result


def _count_consecutive_same_side(above: np.ndarray, k: int) -> int:
    """Nombre de fois qu'on observe >= k points consecutifs du meme cote."""
    if len(above) < k:
        return 0
    count = 0
    run = 1
    for i in range(1, len(above)):
        if above[i] == above[i - 1]:
            run += 1
            if run == k:
                count += 1
        else:
            run = 1
    return count


def _count_consecutive_true(mask: np.ndarray, k: int) -> int:
    """Nombre de fois qu'on observe >= k True consecutifs."""
    if len(mask) < k:
        return 0
    count = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            if run == k:
                count += 1
        else:
            run = 0
    return count


def _count_monotonic(values: np.ndarray, k: int) -> int:
    """Nombre de fois qu'on observe k points strictement monotones (up OU down)."""
    if len(values) < k:
        return 0
    diffs = np.diff(values)
    count = 0
    run_up = 1
    run_down = 1
    for d in diffs:
        if d > 0:
            run_up += 1
            run_down = 1
            if run_up == k:
                count += 1
        elif d < 0:
            run_down += 1
            run_up = 1
            if run_down == k:
                count += 1
        else:
            run_up = 1
            run_down = 1
    return count


def _count_alternating(values: np.ndarray, k: int) -> int:
    """Nombre de fois qu'on observe k points alternant up/down."""
    if len(values) < k:
        return 0
    diffs = np.diff(values)
    signs = np.sign(diffs)
    count = 0
    run = 1
    for i in range(1, len(signs)):
        if signs[i] != 0 and signs[i] == -signs[i - 1]:
            run += 1
            if run == k:
                count += 1
        else:
            run = 1
    return count


def _count_k_of_n_beyond(z: np.ndarray, k: int, n_win: int,
                          threshold: float) -> int:
    """
    Compte les fenetres glissantes de n_win points ou au moins k points
    sont au-dela de +threshold OU au-dela de -threshold (meme cote).
    """
    n = len(z)
    if n < n_win:
        return 0
    count = 0
    for i in range(n - n_win + 1):
        window = z[i:i + n_win]
        n_above = int(np.sum(window > threshold))
        n_below = int(np.sum(window < -threshold))
        if n_above >= k or n_below >= k:
            count += 1
    return count


# -----------------------------------------------------------------------------
# Analyse par parametre (Shewhart + Nelson)
# -----------------------------------------------------------------------------

def analyze_parameter(series: pd.Series) -> dict:
    """Statistiques de base + limites Shewhart + regles Nelson."""
    values = series.to_numpy()
    n = len(values)
    if n == 0:
        out = {"n": 0, "mean": 0.0, "std": 0.0, "ucl": 0.0, "lcl": 0.0,
               "violations": 0, "criticality": 0.0}
        out.update({f"nelson_{i}": 0 for i in range(1, 9)})
        out["nelson_total"] = 0
        out["dominant_pattern"] = "none"
        return out

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    violations = int(np.sum((values > ucl) | (values < lcl)))
    criticality = (violations / n) * 100.0

    result = {
        "n": n,
        "mean": mean,
        "std": std,
        "ucl": ucl,
        "lcl": lcl,
        "violations": violations,
        "criticality": criticality,
    }
    # Enrichissement Nelson (additif)
    result.update(detect_nelson_rules(values, mean, std))
    return result


def analyze_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse chaque colonne : Shewhart + Nelson.
    Classement par criticite decroissante.

    Compatibilite : les colonnes historiques sont en tete
    (parameter, n, mean, std, lcl, ucl, violations, criticality).
    Les colonnes Nelson sont ajoutees apres — le code existant continue
    de fonctionner tel quel.
    """
    rows = []
    for col in df.columns:
        s = analyze_parameter(df[col])
        s["parameter"] = col
        rows.append(s)

    result = pd.DataFrame(rows)
    result = result.sort_values("criticality", ascending=False).reset_index(drop=True)

    base_cols = ["parameter", "n", "mean", "std", "lcl", "ucl",
                 "violations", "criticality"]
    nelson_cols = [f"nelson_{i}" for i in range(1, 9)] + \
                  ["nelson_total", "dominant_pattern"]
    return result[base_cols + nelson_cols]
