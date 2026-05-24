"""
Audit de qualite des donnees avant analyse.
Reflexe industriel : on n'analyse pas des donnees qu'on n'a pas inspectees.

Produit un DataFrame synthetique par colonne avec :
  - n_total, n_valid, pct_missing
  - type detecte (numeric / datetime / categorical / text)
  - plus_grand_trou (NaN consecutifs max)
  - duplicates_temporels (pour colonne datetime)
  - alerte (ok / warn / danger)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityVerdict:
    level: str  # ok | warn | danger
    message: str


import re

_DATE_HINT = re.compile(r"[-/:]\d|\d[-/:]")


def _detect_type(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.notna().sum() / max(len(s), 1) > 0.5:
        return "numeric"
    # datetime heuristique rapide : on teste seulement si un echantillon
    # contient des separateurs date classiques (-, /, :) - sinon on saute
    # le parse couteux de dateutil.
    sample = s.dropna().astype(str).head(5)
    if not sample.empty and sample.str.contains(_DATE_HINT).any():
        try:
            parsed = pd.to_datetime(s.head(20), errors="coerce")
            if parsed.notna().sum() >= 10:
                return "datetime"
        except Exception:
            pass
    n_unique = s.nunique(dropna=True)
    if n_unique <= 10:
        return "categorical"
    return "text"


def _max_gap(s: pd.Series) -> int:
    """Plus longue sequence de valeurs manquantes consecutives."""
    if s.empty:
        return 0
    mask = s.isna().to_numpy()
    if not mask.any():
        return 0
    max_run = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def column_quality(s: pd.Series) -> dict:
    """Stats detailees pour une colonne."""
    n_total = len(s)
    n_valid = int(s.notna().sum())
    pct_missing = 100.0 * (n_total - n_valid) / max(n_total, 1)
    col_type = _detect_type(s)
    max_gap = _max_gap(s)
    n_unique = int(s.nunique(dropna=True))

    # Verdict metier
    if pct_missing >= 50:
        verdict = QualityVerdict(
            "danger",
            f"{pct_missing:.0f}% manquantes - colonne non exploitable"
        )
    elif pct_missing >= 20:
        verdict = QualityVerdict(
            "warn",
            f"{pct_missing:.0f}% manquantes - investigation recommandee"
        )
    elif max_gap >= 20:
        verdict = QualityVerdict(
            "warn",
            f"Trou capteur detecte ({max_gap} NaN consecutifs)"
        )
    elif n_unique <= 1 and col_type == "numeric":
        verdict = QualityVerdict(
            "warn",
            "Valeur constante - aucune information"
        )
    else:
        verdict = QualityVerdict("ok", "Conforme")

    return {
        "column": s.name,
        "type": col_type,
        "n_total": n_total,
        "n_valid": n_valid,
        "pct_missing": round(pct_missing, 2),
        "max_gap_nan": max_gap,
        "n_unique": n_unique,
        "verdict": verdict.level,
        "commentaire": verdict.message,
    }


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Audit complet : une ligne par colonne."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["column", "type", "n_total", "n_valid",
                     "pct_missing", "max_gap_nan", "n_unique",
                     "verdict", "commentaire"]
        )
    rows = [column_quality(df[c]) for c in df.columns]
    return pd.DataFrame(rows)


def summary_stats(df: pd.DataFrame) -> dict:
    """Synthese globale pour header."""
    audit = audit_dataframe(df)
    return {
        "n_columns": len(df.columns) if df is not None else 0,
        "n_rows": len(df) if df is not None else 0,
        "n_ok": int((audit["verdict"] == "ok").sum()) if not audit.empty else 0,
        "n_warn": int((audit["verdict"] == "warn").sum())
                  if not audit.empty else 0,
        "n_danger": int((audit["verdict"] == "danger").sum())
                    if not audit.empty else 0,
    }
