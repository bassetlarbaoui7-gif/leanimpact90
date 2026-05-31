"""
core/cbr/similarity.py - calcul de similarite entre cas.

Approche hybride (3 dimensions) :
  1. Textuel    : TF-IDF + cosinus sur description + type_incident
  2. Categoriel : match pondere sur machine, role createur, statut
  3. Numerique  : distance euclidienne normalisee sur severite (encodee ordinale)

Score final = ponderation des 3 dimensions.
Tout est pur Python + sklearn TF-IDF (deja installe via lightgbm/shap).

Le moteur retourne les top-K cas les plus similaires a un nouveau cas
pour alimenter le moteur de chemins 5 Pourquoi (Jour 3).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from core import db


# ---------------------------------------------------------------------------
# Constantes - encodage ordinal et ponderations
# ---------------------------------------------------------------------------
SEVERITE_ORDINALE = {
    "faible":   1.0,
    "moyenne":  2.0,
    "haute":    3.0,
    "critique": 4.0,
}

# Ponderations finales (somme = 1.0)
WEIGHT_TEXT       = 0.50
WEIGHT_CATEGORIEL = 0.30
WEIGHT_NUMERIQUE  = 0.20

# Mots vides francais (industriels) - simple liste pour TF-IDF
STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "a", "au",
    "aux", "en", "sur", "par", "pour", "avec", "sans", "dans", "ce", "cette",
    "ces", "il", "elle", "ils", "elles", "on", "qui", "que", "quoi", "ou",
    "est", "sont", "etait", "etre", "ete", "fait", "faire", "se", "ne", "pas",
    "plus", "tres", "deja", "encore", "aussi", "donc", "mais", "car", "si",
    "comme", "alors", "puis", "apres", "avant",
}


# ---------------------------------------------------------------------------
# Utilitaires textuels (tokenisation legere FR)
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Lowercase + accents simples retires + ponctuation."""
    if not text:
        return ""
    t = text.lower()
    # Retire accents simples
    repl = (("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
            ("à", "a"), ("â", "a"), ("ä", "a"),
            ("ï", "i"), ("î", "i"),
            ("ô", "o"), ("ö", "o"),
            ("ù", "u"), ("û", "u"), ("ü", "u"),
            ("ç", "c"), ("ñ", "n"))
    for src, dst in repl:
        t = t.replace(src, dst)
    return t


def _tokenize(text: str) -> list[str]:
    """Decoupe en tokens, filtre les stopwords et les tokens trop courts."""
    norm = _normalize(text)
    raw  = re.findall(r"[a-z0-9]+", norm)
    return [tok for tok in raw if len(tok) >= 2 and tok not in STOPWORDS_FR]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Compte les occurrences de chaque token."""
    out: dict[str, float] = {}
    for tok in tokens:
        out[tok] = out.get(tok, 0.0) + 1.0
    return out


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosinus entre 2 vecteurs sparse (dicts {token: poids})."""
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na  = math.sqrt(sum(v * v for v in a.values()))
    nb  = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Composants de similarite
# ---------------------------------------------------------------------------
def text_similarity(text_a: str, text_b: str) -> float:
    """Similarite textuelle entre 2 descriptions (cosinus TF)."""
    toks_a = _tokenize(text_a or "")
    toks_b = _tokenize(text_b or "")
    if not toks_a or not toks_b:
        return 0.0
    tf_a = _term_freq(toks_a)
    tf_b = _term_freq(toks_b)
    return _cosine_sparse(tf_a, tf_b)


def categorical_similarity(
    case_a: dict, case_b: dict,
    *,
    fields_strong: tuple[str, ...] = ("machine", "type_incident"),
    fields_weak:   tuple[str, ...] = ("cree_par_role",),
) -> float:
    """
    Match exact pondere sur les categories.
    fields_strong = poids 1.0, fields_weak = poids 0.4.
    """
    score = 0.0
    weight_total = 0.0
    for f in fields_strong:
        weight_total += 1.0
        va = _normalize(str(case_a.get(f, "")))
        vb = _normalize(str(case_b.get(f, "")))
        if va and vb and va == vb:
            score += 1.0
    for f in fields_weak:
        weight_total += 0.4
        va = _normalize(str(case_a.get(f, "")))
        vb = _normalize(str(case_b.get(f, "")))
        if va and vb and va == vb:
            score += 0.4
    return (score / weight_total) if weight_total > 0 else 0.0


def numeric_similarity(case_a: dict, case_b: dict) -> float:
    """
    Distance normalisee sur severite ordinale.
    Plus la distance est petite, plus le score est haut.
    """
    sa = SEVERITE_ORDINALE.get(str(case_a.get("severite", "")).lower())
    sb = SEVERITE_ORDINALE.get(str(case_b.get("severite", "")).lower())
    if sa is None or sb is None:
        return 0.5  # neutre si donnee manquante
    # Distance max sur l'echelle = 3 (4 niveaux)
    dist = abs(sa - sb)
    return max(0.0, 1.0 - dist / 3.0)


def case_similarity(case_a: dict, case_b: dict) -> float:
    """
    Score global de similarite (0..1) entre 2 incidents.

    Combine :
      50% texte (description + type incident)
      30% categoriel (machine, type, role)
      20% numerique (severite)
    """
    text_a = f"{case_a.get('description', '')} {case_a.get('type_incident', '')}"
    text_b = f"{case_b.get('description', '')} {case_b.get('type_incident', '')}"
    s_text  = text_similarity(text_a, text_b)
    s_categ = categorical_similarity(case_a, case_b)
    s_num   = numeric_similarity(case_a, case_b)
    return (
        WEIGHT_TEXT * s_text +
        WEIGHT_CATEGORIEL * s_categ +
        WEIGHT_NUMERIQUE * s_num
    )


# ---------------------------------------------------------------------------
# Recherche des cas similaires (entry point Jour 3)
# ---------------------------------------------------------------------------
def find_similar_cases(
    new_case: dict,
    *,
    top_k: int = 5,
    min_similarity: float = 0.0,
    exclude_self_id: int | None = None,
    entreprise_id: int | None = None,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Pioche tous les incidents historiques et retourne les top-K les plus
    similaires au cas en entree.

    Le DataFrame retourne contient toutes les colonnes de la table
    incidents + une colonne 'similarity' (0..1) triee decroissante.

    Filtres optionnels :
      exclude_self_id : id de l'incident a exclure (utile si new_case existe deja)
      entreprise_id   : filtre par entreprise (multi-tenant V2)
      min_similarity  : seuil minimum
    """
    df = db.list_incidents(db_path=db_path)
    if df.empty:
        return df  # demarrage froid : aucun cas

    # Filtre eventuel par entreprise
    if entreprise_id is not None and "entreprise_id" in df.columns:
        df = df[df["entreprise_id"] == entreprise_id]

    # Exclure le cas lui-meme si demande
    if exclude_self_id is not None and "id" in df.columns:
        df = df[df["id"] != exclude_self_id]

    if df.empty:
        return df

    # Calcul de similarite pour chaque cas
    scores: list[float] = []
    for _, row in df.iterrows():
        scores.append(case_similarity(new_case, row.to_dict()))

    df = df.copy()
    df["similarity"] = scores
    df = df[df["similarity"] >= min_similarity]
    df = df.sort_values("similarity", ascending=False).head(top_k)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper : recuperer un incident sous forme de dict (pour usage CBR)
# ---------------------------------------------------------------------------
def incident_to_case_dict(incident_row: dict | pd.Series) -> dict:
    """
    Normalise une ligne incident en dict prêt pour case_similarity().
    Utile quand on passe d'un nouveau cas en cours de saisie a la recherche.
    """
    if isinstance(incident_row, pd.Series):
        incident_row = incident_row.to_dict()
    return {
        "id":              incident_row.get("id"),
        "machine":         incident_row.get("machine", ""),
        "type_incident":   incident_row.get("type_incident", ""),
        "description":     incident_row.get("description", ""),
        "severite":        incident_row.get("severite", "moyenne"),
        "cree_par_role":   incident_row.get("cree_par_role", ""),
        "entreprise_id":   incident_row.get("entreprise_id", 1),
    }
