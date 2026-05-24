-- =============================================================
-- LI90 - F3 Causes Racines - Schéma BDD v1
-- Cible : SQLite 3.35+ (stdlib Python)
-- Statut : SPECIFICATION - non encore appliqué
-- Date   : 2026-05-05
-- =============================================================

-- Pragmas obligatoires à appliquer à chaque connexion (NON inclus ici,
-- gérés dans db.py au moment de la connexion) :
--   PRAGMA foreign_keys = ON;
--   PRAGMA journal_mode = WAL;
--   PRAGMA synchronous  = NORMAL;

-- -------------------------------------------------------------
-- Table système : suivi des migrations
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- -------------------------------------------------------------
-- 1. defauts : saisie opérateur (1 ligne = 1 défaut constaté)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS defauts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    operator_id         TEXT    NOT NULL,
    ligne               TEXT    NOT NULL,
    type_defaut         TEXT    NOT NULL,
    severite            INTEGER NOT NULL CHECK (severite BETWEEN 1 AND 5),
    commentaire         TEXT,
    raw_data_snapshot   TEXT          -- JSON sérialisé
);

CREATE INDEX IF NOT EXISTS idx_defauts_timestamp ON defauts(timestamp);
CREATE INDEX IF NOT EXISTS idx_defauts_ligne     ON defauts(ligne);

-- -------------------------------------------------------------
-- 2. model_versions : audit ISO / 8D
--    1 modèle par axe M (Machine, Matiere, Methode)
--    Jamais MO ni Milieu (saisie manuelle uniquement)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    m_axis              TEXT    NOT NULL CHECK (m_axis IN ('Machine','Matiere','Methode')),
    version             TEXT    NOT NULL,
    training_date       TEXT    NOT NULL,
    training_data_hash  TEXT    NOT NULL,
    nb_samples          INTEGER NOT NULL,
    top3_accuracy       REAL    NOT NULL CHECK (top3_accuracy BETWEEN 0 AND 1),
    onnx_path           TEXT    NOT NULL,
    active              INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0,1))
);

-- Garantit qu'un seul modèle est actif par axe à la fois
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_active_per_axis
    ON model_versions(m_axis)
    WHERE active = 1;

-- -------------------------------------------------------------
-- 3. predictions : top-3 IA pour un défaut donné, par axe M
--    1 défaut peut avoir jusqu'à 3 prédictions par axe (= 9 lignes max)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    defaut_id           INTEGER NOT NULL,
    timestamp           TEXT    NOT NULL,
    model_version_id    INTEGER NOT NULL,
    m_axis              TEXT    NOT NULL CHECK (m_axis IN ('Machine','Matiere','Methode')),
    rank                INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 3),
    cause_label         TEXT    NOT NULL,
    probabilite         REAL    NOT NULL CHECK (probabilite BETWEEN 0 AND 1),
    shap_features       TEXT    NOT NULL,                   -- JSON top-5 features
    inference_time_ms   INTEGER NOT NULL,
    statut              TEXT    NOT NULL CHECK (statut IN ('proposee','inconnu_low_conf')),
    FOREIGN KEY (defaut_id)        REFERENCES defauts(id)        ON DELETE CASCADE,
    FOREIGN KEY (model_version_id) REFERENCES model_versions(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_predictions_defaut    ON predictions(defaut_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_ver ON predictions(model_version_id);

-- -------------------------------------------------------------
-- 4. feedback : retour opérateur sur la justesse d'une prédiction
--    prediction_id peut être NULL si l'opérateur signale une cause
--    qui n'était pas dans le top-3 proposé
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER,
    defaut_id       INTEGER NOT NULL,
    timestamp       TEXT    NOT NULL,
    operator_id     TEXT    NOT NULL,
    was_correct     INTEGER NOT NULL CHECK (was_correct IN (0,1)),
    vraie_cause     TEXT,
    commentaire     TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL,
    FOREIGN KEY (defaut_id)     REFERENCES defauts(id)     ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_defaut     ON feedback(defaut_id);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON feedback(prediction_id);

-- -------------------------------------------------------------
-- Marquer la version courante du schéma
-- -------------------------------------------------------------
INSERT OR IGNORE INTO schema_version (version, applied_at)
VALUES (1, datetime('now'));
