"""
core/db.py - acces base SQLite pour le workflow Vue B.

4 tables ajoutees a li90.db (la base existante reste intacte) :
  - incidents          : signalements bruts (operateur)
  - projets_ac         : projets d'amelioration continue (Resp. AC)
  - validations        : validations distribuees (Resp. Prod + Tech N+1)
  - actions            : plan d'action + suivi + ROI

Tous les helpers retournent soit un int (id cree), soit un dict
(get_one), soit un pandas.DataFrame (list_*), pour rester simple a
brancher dans les pages Streamlit.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Path & connexion
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent.parent / "li90.db"


@contextmanager
def connect(db_path: Path | str | None = None):
    """Connexion SQLite avec PRAGMA foreign_keys ON, commit/rollback auto."""
    path = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schemas SQL des 4 tables Vue B
# ---------------------------------------------------------------------------
SCHEMA_INCIDENTS = """
CREATE TABLE IF NOT EXISTS incidents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    operateur_nom       TEXT,
    machine             TEXT NOT NULL,
    type_incident       TEXT,
    description         TEXT,
    audio_transcript    TEXT,
    photo_path          TEXT,
    severite            TEXT DEFAULT 'moyenne',
    contexte_json       TEXT,
    statut              TEXT NOT NULL DEFAULT 'brut',
    cree_par_role       TEXT,
    cree_le             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    maj_le              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_incidents_statut ON incidents(statut);
"""

SCHEMA_PROJETS_AC = """
CREATE TABLE IF NOT EXISTS projets_ac (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id         INTEGER,
    titre               TEXT NOT NULL,
    qqoqcp_qui          TEXT,
    qqoqcp_quoi         TEXT,
    qqoqcp_ou           TEXT,
    qqoqcp_quand        TEXT,
    qqoqcp_comment      TEXT,
    qqoqcp_pourquoi     TEXT,
    causes_ia_json      TEXT,
    causes_validees     TEXT,
    solution_proposee   TEXT,
    cout_estime         REAL,
    temps_estime_jours  REAL,
    gain_estime_eur     REAL,
    gain_productivite   REAL,
    statut              TEXT NOT NULL DEFAULT 'cadrage',
    cree_par            TEXT,
    cree_par_role       TEXT,
    cree_le             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    maj_le              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (incident_id) REFERENCES incidents(id)
);
CREATE INDEX IF NOT EXISTS idx_projets_ac_statut ON projets_ac(statut);
"""

SCHEMA_VALIDATIONS = """
CREATE TABLE IF NOT EXISTS validations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    projet_id           INTEGER NOT NULL,
    role_valideur       TEXT NOT NULL,
    nom_valideur        TEXT,
    decision            TEXT,
    commentaire         TEXT,
    cree_le             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (projet_id) REFERENCES projets_ac(id)
);
CREATE INDEX IF NOT EXISTS idx_validations_projet ON validations(projet_id);
"""

SCHEMA_ACTIONS = """
CREATE TABLE IF NOT EXISTS actions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    projet_id           INTEGER NOT NULL,
    titre               TEXT NOT NULL,
    description         TEXT,
    assignee            TEXT,
    priorite            INTEGER DEFAULT 2,
    statut              TEXT NOT NULL DEFAULT 'a_faire',
    echeance            DATE,
    termine_le          TIMESTAMP,
    roi_reel_eur        REAL,
    cree_le             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (projet_id) REFERENCES projets_ac(id)
);
CREATE INDEX IF NOT EXISTS idx_actions_projet ON actions(projet_id);
CREATE INDEX IF NOT EXISTS idx_actions_statut ON actions(statut);
"""

ALL_SCHEMAS = (
    SCHEMA_INCIDENTS,
    SCHEMA_PROJETS_AC,
    SCHEMA_VALIDATIONS,
    SCHEMA_ACTIONS,
)


def init_db(db_path: Path | str | None = None) -> None:
    """Cree toutes les tables si elles n'existent pas. Idempotent.

    Cree :
      - 4 tables de base (incidents, projets_ac, validations, actions)
      - 4 tables CBR (entreprises, chemins_pourquoi, evidence, feedback_chemins)
        + ajoute entreprise_id sur incidents/projets_ac
    """
    with connect(db_path) as conn:
        for schema in ALL_SCHEMAS:
            conn.executescript(schema)
    # Import local pour eviter une circular dependency au module-level
    from core.cbr.case_base import init_cbr_tables
    init_cbr_tables(db_path)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------
def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def _rows_to_df(rows: Iterable[sqlite3.Row]) -> pd.DataFrame:
    data = [dict(r) for r in rows]
    return pd.DataFrame(data)


def _json_dump(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(v, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# INCIDENTS - CRUD
# ---------------------------------------------------------------------------
def create_incident(
    machine: str,
    description: str = "",
    *,
    operateur_nom: str = "",
    type_incident: str = "",
    severite: str = "moyenne",
    audio_transcript: str = "",
    photo_path: str = "",
    contexte: dict | None = None,
    cree_par_role: str = "",
    db_path: Path | str | None = None,
) -> int:
    """Cree un incident brut. Retourne l'id."""
    sql = """
        INSERT INTO incidents
        (machine, description, operateur_nom, type_incident, severite,
         audio_transcript, photo_path, contexte_json, statut, cree_par_role)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'brut', ?)
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, (
            machine.strip(),
            description.strip(),
            operateur_nom.strip(),
            type_incident.strip(),
            severite,
            audio_transcript,
            photo_path,
            _json_dump(contexte),
            cree_par_role,
        ))
        return int(cur.lastrowid)


def list_incidents(
    statut: str | None = None,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Liste les incidents (optionnel filtre statut). Tri recent d'abord."""
    sql = "SELECT * FROM incidents"
    params: tuple = ()
    if statut:
        sql += " WHERE statut = ?"
        params = (statut,)
    sql += " ORDER BY cree_le DESC"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return _rows_to_df(rows)


def get_incident(
    incident_id: int,
    *,
    db_path: Path | str | None = None,
) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM incidents WHERE id = ?", (incident_id,),
        ).fetchone()
    return _row_to_dict(row)


def update_incident_statut(
    incident_id: int, nouveau_statut: str,
    *,
    db_path: Path | str | None = None,
) -> bool:
    with connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE incidents SET statut = ?, maj_le = CURRENT_TIMESTAMP"
            " WHERE id = ?",
            (nouveau_statut, incident_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# PROJETS AC - CRUD
# ---------------------------------------------------------------------------
def create_projet_ac(
    titre: str,
    *,
    incident_id: int | None = None,
    cree_par: str = "",
    cree_par_role: str = "",
    db_path: Path | str | None = None,
) -> int:
    sql = """
        INSERT INTO projets_ac
        (incident_id, titre, statut, cree_par, cree_par_role)
        VALUES (?, ?, 'cadrage', ?, ?)
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, (
            incident_id, titre.strip(), cree_par, cree_par_role,
        ))
        return int(cur.lastrowid)


def list_projets_ac(
    statut: str | None = None,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM projets_ac"
    params: tuple = ()
    if statut:
        sql += " WHERE statut = ?"
        params = (statut,)
    sql += " ORDER BY cree_le DESC"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return _rows_to_df(rows)


def get_projet_ac(
    projet_id: int,
    *,
    db_path: Path | str | None = None,
) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM projets_ac WHERE id = ?", (projet_id,),
        ).fetchone()
    return _row_to_dict(row)


_ALLOWED_PROJET_FIELDS = {
    "titre", "qqoqcp_qui", "qqoqcp_quoi", "qqoqcp_ou", "qqoqcp_quand",
    "qqoqcp_comment", "qqoqcp_pourquoi", "causes_ia_json",
    "causes_validees", "solution_proposee", "cout_estime",
    "temps_estime_jours", "gain_estime_eur", "gain_productivite", "statut",
}


def update_projet_ac(
    projet_id: int,
    *,
    db_path: Path | str | None = None,
    **fields,
) -> bool:
    """Update partiel sur projets_ac. Seuls les champs whitelistes passent."""
    safe = {k: v for k, v in fields.items() if k in _ALLOWED_PROJET_FIELDS}
    if not safe:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in safe.keys())
    values = list(safe.values()) + [projet_id]
    sql = (
        f"UPDATE projets_ac SET {set_clause}, maj_le = CURRENT_TIMESTAMP"
        " WHERE id = ?"
    )
    with connect(db_path) as conn:
        cur = conn.execute(sql, values)
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# VALIDATIONS - CRUD
# ---------------------------------------------------------------------------
def add_validation(
    projet_id: int,
    role_valideur: str,
    decision: str,
    *,
    nom_valideur: str = "",
    commentaire: str = "",
    db_path: Path | str | None = None,
) -> int:
    sql = """
        INSERT INTO validations
        (projet_id, role_valideur, nom_valideur, decision, commentaire)
        VALUES (?, ?, ?, ?, ?)
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, (
            projet_id, role_valideur, nom_valideur, decision, commentaire,
        ))
        return int(cur.lastrowid)


def list_validations(
    projet_id: int | None = None,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM validations"
    params: tuple = ()
    if projet_id is not None:
        sql += " WHERE projet_id = ?"
        params = (projet_id,)
    sql += " ORDER BY cree_le DESC"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return _rows_to_df(rows)


# ---------------------------------------------------------------------------
# ACTIONS - CRUD
# ---------------------------------------------------------------------------
def add_action(
    projet_id: int,
    titre: str,
    *,
    description: str = "",
    assignee: str = "",
    priorite: int = 2,
    echeance: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    sql = """
        INSERT INTO actions
        (projet_id, titre, description, assignee, priorite, echeance, statut)
        VALUES (?, ?, ?, ?, ?, ?, 'a_faire')
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, (
            projet_id, titre.strip(), description.strip(), assignee,
            priorite, echeance,
        ))
        return int(cur.lastrowid)


def list_actions(
    projet_id: int | None = None,
    statut: str | None = None,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM actions WHERE 1=1"
    params: list = []
    if projet_id is not None:
        sql += " AND projet_id = ?"
        params.append(projet_id)
    if statut:
        sql += " AND statut = ?"
        params.append(statut)
    sql += " ORDER BY priorite ASC, cree_le DESC"
    with connect(db_path) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _rows_to_df(rows)


_ALLOWED_ACTION_FIELDS = {
    "titre", "description", "assignee", "priorite", "statut",
    "echeance", "termine_le", "roi_reel_eur",
}


def update_action(
    action_id: int,
    *,
    db_path: Path | str | None = None,
    **fields,
) -> bool:
    safe = {k: v for k, v in fields.items() if k in _ALLOWED_ACTION_FIELDS}
    if not safe:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in safe.keys())
    values = list(safe.values()) + [action_id]
    sql = f"UPDATE actions SET {set_clause} WHERE id = ?"
    with connect(db_path) as conn:
        cur = conn.execute(sql, values)
        return cur.rowcount > 0
