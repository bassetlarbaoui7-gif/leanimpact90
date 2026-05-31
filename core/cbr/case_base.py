"""
core/cbr/case_base.py - Base de cas et helpers CRUD pour le moteur CBR.

3 tables principales du moteur CBR :
  - chemins_pourquoi   : l'arbre 5 Pourquoi par projet et branche M
  - evidence           : preuves attachees a chaque noeud (capteur/photo/...)
  - feedback_chemins   : validation humaine +1 / -1 / 0 (ajuste)

Plus 1 table de preparation multi-tenant :
  - entreprises        : 1 ligne par client (Gascogne pour le MVP V1)

Toutes les tables ont entreprise_id pour preparer la V2 multi-clients.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.db import DB_PATH, connect


# ---------------------------------------------------------------------------
# Schemas SQL des tables CBR
# ---------------------------------------------------------------------------
SCHEMA_ENTREPRISES = """
CREATE TABLE IF NOT EXISTS entreprises (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    nom           TEXT NOT NULL UNIQUE,
    secteur       TEXT,                       -- emballage, plastique, agro...
    actif         INTEGER DEFAULT 1,
    cree_le       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_entreprises_secteur ON entreprises(secteur);
"""

SCHEMA_CHEMINS_POURQUOI = """
CREATE TABLE IF NOT EXISTS chemins_pourquoi (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    projet_id         INTEGER NOT NULL,
    branche_m         TEXT NOT NULL,          -- Machine / Matiere / Methode / Main-oeuvre / Milieu
    niveau            INTEGER NOT NULL,        -- 1 a N (typiquement 1 a 5)
    question          TEXT NOT NULL,
    reponse           TEXT NOT NULL,
    parent_id         INTEGER,                 -- chaine parent -> enfant
    type_noeud        TEXT DEFAULT 'cause_directe',
                                              -- symptome | condition | cause_directe | cause_racine
    est_cause_racine  INTEGER DEFAULT 0,
    source_cas_id     INTEGER,                 -- cas historique d'ou provient ce chemin (CBR)
    similarite_cas    REAL,                    -- 0.0 a 1.0
    confidence        REAL DEFAULT 0.5,
    entreprise_id     INTEGER DEFAULT 1,
    cree_le           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (projet_id)  REFERENCES projets_ac(id),
    FOREIGN KEY (parent_id)  REFERENCES chemins_pourquoi(id),
    FOREIGN KEY (entreprise_id) REFERENCES entreprises(id)
);
CREATE INDEX IF NOT EXISTS idx_chemins_projet   ON chemins_pourquoi(projet_id);
CREATE INDEX IF NOT EXISTS idx_chemins_branche  ON chemins_pourquoi(branche_m);
CREATE INDEX IF NOT EXISTS idx_chemins_racine   ON chemins_pourquoi(est_cause_racine);
"""

SCHEMA_EVIDENCE = """
CREATE TABLE IF NOT EXISTS evidence (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    chemin_id         INTEGER NOT NULL,
    type              TEXT NOT NULL,           -- sensor | photo | document | testimony
    contenu           TEXT,                    -- valeur mesuree, chemin fichier, texte temoin
    description       TEXT,
    confidence        REAL DEFAULT 0.5,
    timestamp_source  TIMESTAMP,
    cree_le           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chemin_id) REFERENCES chemins_pourquoi(id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_chemin  ON evidence(chemin_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type    ON evidence(type);
"""

SCHEMA_FEEDBACK_CHEMINS = """
CREATE TABLE IF NOT EXISTS feedback_chemins (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chemin_id       INTEGER NOT NULL,
    decision        INTEGER NOT NULL,          -- +1 valide / -1 refus / 0 ajuste
    commentaire     TEXT,
    valide_par      TEXT,
    role_valideur   TEXT,                       -- ac_manager / technician / production / ...
    valide_le       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chemin_id) REFERENCES chemins_pourquoi(id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_chemin    ON feedback_chemins(chemin_id);
CREATE INDEX IF NOT EXISTS idx_feedback_decision  ON feedback_chemins(decision);
"""

ALL_CBR_SCHEMAS = (
    SCHEMA_ENTREPRISES,
    SCHEMA_CHEMINS_POURQUOI,
    SCHEMA_EVIDENCE,
    SCHEMA_FEEDBACK_CHEMINS,
)


# ---------------------------------------------------------------------------
# Init + extension multi-tenant des tables existantes
# ---------------------------------------------------------------------------
def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = {r["name"] for r in rows}
    return col in cols


def _ensure_entreprise_id_columns(conn: sqlite3.Connection) -> None:
    """Ajoute entreprise_id sur incidents et projets_ac (idempotent)."""
    for table in ("incidents", "projets_ac"):
        # La table existe ?
        try:
            if not _column_exists(conn, table, "entreprise_id"):
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN "
                    "entreprise_id INTEGER DEFAULT 1"
                )
        except sqlite3.OperationalError:
            # La table n'existe pas encore - sera creee par db.init_db()
            pass


def _seed_default_entreprise(conn: sqlite3.Connection) -> None:
    """Insere 'gascogne' comme entreprise par defaut (id=1)."""
    row = conn.execute(
        "SELECT id FROM entreprises WHERE id = 1"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO entreprises (id, nom, secteur, actif) "
            "VALUES (1, 'Gascogne Sacs', 'emballage_papier', 1)"
        )


def init_cbr_tables(db_path: Path | str | None = None) -> None:
    """Cree toutes les tables CBR et ajoute entreprise_id si manquant.

    Idempotent : peut etre appelee a chaque demarrage du logiciel.
    """
    with connect(db_path) as conn:
        for schema in ALL_CBR_SCHEMAS:
            conn.executescript(schema)
        _ensure_entreprise_id_columns(conn)
        _seed_default_entreprise(conn)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------
def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def _rows_to_df(rows: Iterable[sqlite3.Row]) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# CRUD - CHEMINS POURQUOI (l'arbre 5 Pourquoi)
# ---------------------------------------------------------------------------
def add_chemin(
    projet_id: int,
    branche_m: str,
    niveau: int,
    question: str,
    reponse: str,
    *,
    parent_id: int | None = None,
    type_noeud: str = "cause_directe",
    est_cause_racine: bool = False,
    source_cas_id: int | None = None,
    similarite_cas: float | None = None,
    confidence: float = 0.5,
    entreprise_id: int = 1,
    db_path: Path | str | None = None,
) -> int:
    """Insere un noeud dans l'arbre 5 Pourquoi. Retourne l'id."""
    sql = """
        INSERT INTO chemins_pourquoi
        (projet_id, branche_m, niveau, question, reponse, parent_id,
         type_noeud, est_cause_racine, source_cas_id, similarite_cas,
         confidence, entreprise_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, (
            projet_id, branche_m, niveau, question.strip(), reponse.strip(),
            parent_id, type_noeud, 1 if est_cause_racine else 0,
            source_cas_id, similarite_cas, confidence, entreprise_id,
        ))
        return int(cur.lastrowid)


def get_chemin(
    chemin_id: int,
    *,
    db_path: Path | str | None = None,
) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM chemins_pourquoi WHERE id = ?", (chemin_id,),
        ).fetchone()
    return _row_to_dict(row)


def list_chemins_projet(
    projet_id: int,
    *,
    branche_m: str | None = None,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Liste tous les noeuds d'un projet, tries par branche puis niveau."""
    sql = "SELECT * FROM chemins_pourquoi WHERE projet_id = ?"
    params: list = [projet_id]
    if branche_m:
        sql += " AND branche_m = ?"
        params.append(branche_m)
    sql += " ORDER BY branche_m, niveau, id"
    with connect(db_path) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _rows_to_df(rows)


def list_causes_racines_historiques(
    *,
    branche_m: str | None = None,
    entreprise_id: int | None = None,
    db_path: Path | str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """
    Retourne toutes les causes racines deja validees sur les projets
    passes (sera utilise par le moteur CBR pour proposer des chemins).
    """
    sql = """
        SELECT c.*
        FROM chemins_pourquoi c
        WHERE c.est_cause_racine = 1
    """
    params: list = []
    if branche_m:
        sql += " AND c.branche_m = ?"
        params.append(branche_m)
    if entreprise_id is not None:
        sql += " AND c.entreprise_id = ?"
        params.append(entreprise_id)
    sql += " ORDER BY c.cree_le DESC LIMIT ?"
    params.append(limit)
    with connect(db_path) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _rows_to_df(rows)


def update_chemin(
    chemin_id: int,
    *,
    db_path: Path | str | None = None,
    **fields,
) -> bool:
    """Update partiel d'un noeud (champs whitelistes)."""
    allowed = {
        "question", "reponse", "type_noeud", "est_cause_racine",
        "confidence", "similarite_cas",
    }
    safe = {k: v for k, v in fields.items() if k in allowed}
    if not safe:
        return False
    # Cast booleen pour SQLite
    if "est_cause_racine" in safe:
        safe["est_cause_racine"] = 1 if safe["est_cause_racine"] else 0
    set_clause = ", ".join(f"{k} = ?" for k in safe.keys())
    values = list(safe.values()) + [chemin_id]
    sql = f"UPDATE chemins_pourquoi SET {set_clause} WHERE id = ?"
    with connect(db_path) as conn:
        cur = conn.execute(sql, values)
        return cur.rowcount > 0


def delete_chemins_projet(
    projet_id: int,
    *,
    branche_m: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Supprime tous les noeuds d'un projet (utile pour regenerer l'arbre)."""
    sql = "DELETE FROM chemins_pourquoi WHERE projet_id = ?"
    params: list = [projet_id]
    if branche_m:
        sql += " AND branche_m = ?"
        params.append(branche_m)
    with connect(db_path) as conn:
        cur = conn.execute(sql, tuple(params))
        return cur.rowcount


# ---------------------------------------------------------------------------
# CRUD - EVIDENCE (preuves attachees a un noeud)
# ---------------------------------------------------------------------------
def add_evidence(
    chemin_id: int,
    type_evidence: str,
    contenu: str = "",
    *,
    description: str = "",
    confidence: float = 0.5,
    timestamp_source: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """type_evidence in {sensor, photo, document, testimony}."""
    sql = """
        INSERT INTO evidence
        (chemin_id, type, contenu, description, confidence, timestamp_source)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, (
            chemin_id, type_evidence, contenu, description.strip(),
            confidence, timestamp_source,
        ))
        return int(cur.lastrowid)


def list_evidence_for_chemin(
    chemin_id: int,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM evidence WHERE chemin_id = ? ORDER BY cree_le",
            (chemin_id,),
        ).fetchall()
    return _rows_to_df(rows)


# ---------------------------------------------------------------------------
# CRUD - FEEDBACK (boucle d'apprentissage)
# ---------------------------------------------------------------------------
def add_feedback(
    chemin_id: int,
    decision: int,
    *,
    commentaire: str = "",
    valide_par: str = "",
    role_valideur: str = "",
    db_path: Path | str | None = None,
) -> int:
    """
    decision : +1 (valide) / -1 (refus) / 0 (ajuste).
    Chaque feedback est trace pour permettre le ranking CBR.
    """
    if decision not in (-1, 0, 1):
        raise ValueError(f"decision doit etre -1, 0 ou +1 (recu {decision})")
    sql = """
        INSERT INTO feedback_chemins
        (chemin_id, decision, commentaire, valide_par, role_valideur)
        VALUES (?, ?, ?, ?, ?)
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, (
            chemin_id, decision, commentaire, valide_par, role_valideur,
        ))
        return int(cur.lastrowid)


def get_feedback_score(
    chemin_id: int,
    *,
    db_path: Path | str | None = None,
) -> dict:
    """
    Retourne {valid: n, refus: n, ajuste: n, net: signed_sum} pour
    pondering les chemins lors du ranking CBR.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT decision FROM feedback_chemins WHERE chemin_id = ?",
            (chemin_id,),
        ).fetchall()
    decisions = [int(r["decision"]) for r in rows]
    return {
        "valid":  sum(1 for d in decisions if d == 1),
        "refus":  sum(1 for d in decisions if d == -1),
        "ajuste": sum(1 for d in decisions if d == 0),
        "net":    sum(decisions),
    }


def list_feedback_for_chemin(
    chemin_id: int,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM feedback_chemins WHERE chemin_id = ? "
            "ORDER BY valide_le DESC",
            (chemin_id,),
        ).fetchall()
    return _rows_to_df(rows)


# ---------------------------------------------------------------------------
# CRUD - ENTREPRISES (multi-tenant V2 prep)
# ---------------------------------------------------------------------------
def list_entreprises(
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM entreprises WHERE actif = 1 ORDER BY nom"
        ).fetchall()
    return _rows_to_df(rows)


def get_entreprise(
    entreprise_id: int,
    *,
    db_path: Path | str | None = None,
) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM entreprises WHERE id = ?", (entreprise_id,),
        ).fetchone()
    return _row_to_dict(row)
