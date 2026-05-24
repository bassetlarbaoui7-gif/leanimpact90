"""
F3 - Ishikawa 5M + detection IA des causes racines.

Architecture :
- Stockage des analyses historiques en SQLite (table rca).
- V1 : recherche par similarite TF-IDF + cosine similarity (demarrage sans historique).
- Interface pluggable : RootCauseEngine pourra etre remplace par LightGBM/SHAP
  des que l'historique labellise le permet.
- Structure 5M : Machine, Methode, Matiere, Main-d'oeuvre, Milieu.

Pas de dependance cloud. Tout tourne en local, compatible PyInstaller.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Constantes 5M
# ---------------------------------------------------------------------------
BRANCHES_5M: tuple[str, ...] = (
    "Machine",
    "Methode",
    "Matiere",
    "Main-d'oeuvre",
    "Milieu",
)

# Auto-remplissage possible (donnees quantitatives) vs saisie humaine
AUTO_BRANCHES = {"Machine", "Matiere"}
SEMI_AUTO_BRANCHES = {"Methode"}
MANUAL_BRANCHES = {"Main-d'oeuvre", "Milieu"}


# ---------------------------------------------------------------------------
# Modele de donnees
# ---------------------------------------------------------------------------
@dataclass
class RcaCase:
    """Un cas d'analyse RCA complet."""

    defect_type: str
    context: str  # description textuelle libre
    parameters: dict[str, float] = field(default_factory=dict)
    causes_5m: dict[str, list[str]] = field(default_factory=dict)
    validated_root_cause: str = ""
    validated_branch: str = ""
    operator: str = ""
    created_at: str = ""

    def to_text(self) -> str:
        """Serialise le cas en texte pour vectorisation TF-IDF."""
        parts = [self.defect_type, self.context]
        for k, v in self.parameters.items():
            parts.append(f"{k}={v}")
        for branch, causes in self.causes_5m.items():
            parts.append(f"{branch}: " + ", ".join(causes))
        return " | ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Stockage SQLite
# ---------------------------------------------------------------------------
DEFAULT_DB = Path("li90.db")


def init_db(db_path: Path = DEFAULT_DB) -> None:
    """Cree le schema si besoin."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rca (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            defect_type TEXT NOT NULL,
            context TEXT,
            parameters_json TEXT,
            causes_5m_json TEXT,
            validated_root_cause TEXT,
            validated_branch TEXT,
            operator TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_rca(case: RcaCase, db_path: Path = DEFAULT_DB) -> int:
    """Persiste un cas RCA et retourne son id."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rca (
            defect_type, context, parameters_json, causes_5m_json,
            validated_root_cause, validated_branch, operator, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case.defect_type,
            case.context,
            json.dumps(case.parameters, ensure_ascii=False),
            json.dumps(case.causes_5m, ensure_ascii=False),
            case.validated_root_cause,
            case.validated_branch,
            case.operator,
            case.created_at or datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def load_all_rca(db_path: Path = DEFAULT_DB) -> list[RcaCase]:
    """Charge tous les cas RCA pour l'indexation IA."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """SELECT defect_type, context, parameters_json, causes_5m_json,
                  validated_root_cause, validated_branch, operator, created_at
           FROM rca"""
    )
    cases: list[RcaCase] = []
    for row in cur.fetchall():
        cases.append(
            RcaCase(
                defect_type=row[0],
                context=row[1] or "",
                parameters=json.loads(row[2] or "{}"),
                causes_5m=json.loads(row[3] or "{}"),
                validated_root_cause=row[4] or "",
                validated_branch=row[5] or "",
                operator=row[6] or "",
                created_at=row[7] or "",
            )
        )
    conn.close()
    return cases


# ---------------------------------------------------------------------------
# Moteur IA - V1 TF-IDF + cosine similarity
# ---------------------------------------------------------------------------
@dataclass
class CauseSuggestion:
    branch: str
    cause: str
    confidence: float  # 0..1
    source_case_id: int | None = None
    explanation: str = ""


class RootCauseEngine:
    """
    Moteur IA pluggable.
    V1 : TF-IDF + cosine. V2 : remplacera par LightGBM + SHAP + ONNX
    des que >= 200 cas historiques labellises sont disponibles.
    """

    def __init__(self, min_history: int = 3) -> None:
        self.min_history = min_history
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._cases: list[RcaCase] = []

    def fit(self, cases: Sequence[RcaCase]) -> None:
        self._cases = list(cases)
        if len(self._cases) < self.min_history:
            self._vectorizer = None
            self._matrix = None
            return
        texts = [c.to_text() for c in self._cases]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def is_trained(self) -> bool:
        return self._vectorizer is not None and self._matrix is not None

    def suggest(
        self,
        defect_type: str,
        context: str,
        parameters: dict[str, float],
        top_k: int = 5,
    ) -> list[CauseSuggestion]:
        """Retourne les top-K causes les plus probables, triees par confiance."""
        if not self.is_trained():
            return []
        query_case = RcaCase(
            defect_type=defect_type, context=context, parameters=parameters
        )
        q_vec = self._vectorizer.transform([query_case.to_text()])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        # Tri decroissant
        order = np.argsort(-sims)
        suggestions: list[CauseSuggestion] = []
        seen_causes: set[tuple[str, str]] = set()
        for idx in order:
            if sims[idx] <= 0:
                break
            c = self._cases[idx]
            if c.validated_root_cause and c.validated_branch:
                key = (c.validated_branch, c.validated_root_cause)
                if key not in seen_causes:
                    suggestions.append(
                        CauseSuggestion(
                            branch=c.validated_branch,
                            cause=c.validated_root_cause,
                            confidence=float(sims[idx]),
                            source_case_id=idx,
                            explanation=(
                                f"Cas similaire (defect: {c.defect_type}) "
                                f"resolu le {c.created_at[:10]}"
                            ),
                        )
                    )
                    seen_causes.add(key)
            if len(suggestions) >= top_k:
                break
        return suggestions


# ---------------------------------------------------------------------------
# Auto-remplissage Machine / Matiere a partir des donnees courantes
# ---------------------------------------------------------------------------
def auto_fill_machine_branch(
    parameters: dict[str, float],
    cibles: dict[str, tuple[float, float]],
) -> list[str]:
    """
    Branche Machine : detecte les parametres hors plage cible.
    cibles : {param: (min_cible, max_cible)}
    """
    suspects = []
    for param, value in parameters.items():
        if param not in cibles:
            continue
        lo, hi = cibles[param]
        if value < lo:
            suspects.append(f"{param} bas ({value:.2f} < {lo})")
        elif value > hi:
            suspects.append(f"{param} haut ({value:.2f} > {hi})")
    return suspects


def auto_fill_matiere_branch(
    current_lot: dict[str, float],
    historical_lots: pd.DataFrame,
    z_threshold: float = 2.0,
) -> list[str]:
    """
    Branche Matiere : compare le lot courant aux lots precedents.
    Signale les parametres atypiques (z-score > seuil).
    """
    suspects = []
    for key, val in current_lot.items():
        if key not in historical_lots.columns:
            continue
        col = historical_lots[key].dropna()
        if len(col) < 5 or col.std() == 0:
            continue
        z = abs((val - col.mean()) / col.std())
        if z > z_threshold:
            suspects.append(f"Lot atypique sur {key} (z={z:.1f})")
    return suspects


# ---------------------------------------------------------------------------
# Helpers pour le rendu UI
# ---------------------------------------------------------------------------
def empty_5m_structure() -> dict[str, list[str]]:
    return {b: [] for b in BRANCHES_5M}


def branch_mode(branch: str) -> str:
    if branch in AUTO_BRANCHES:
        return "auto"
    if branch in SEMI_AUTO_BRANCHES:
        return "semi-auto"
    return "manuel"
