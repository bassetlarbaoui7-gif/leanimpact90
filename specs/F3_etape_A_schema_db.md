# F3 — Étape A : Schéma BDD + Contrat API

**Statut** : spécification — non implémenté
**Version** : 0.1 — 2026-05-05
**Module cible** : `li90_mvp/db.py`
**Dépend de** : Contrat d'Inversion F3 (cf. discussion architecte)

---

## 1. Contraintes de design (validées par Castro)

- **Top-3 accuracy cible** : 80 % au démarrage, montée par feedback loop
- **L'IA ne propose jamais une cause humaine** (`Main d'œuvre`, `Milieu` restent en saisie manuelle)
- **Priorité absolue** : simplicité, robustesse, portabilité PC industriel sans dépendances lourdes
- **Adoption terrain > sophistication technique**

---

## 2. Décisions techniques fondatrices

| Décision | Choix | Justification |
|---|---|---|
| Moteur BDD | SQLite (fichier `.db`) | Zéro install, embarqué Python stdlib, fonctionne offline |
| Driver | `sqlite3` stdlib uniquement | Aucune dépendance externe |
| Concurrence | Mode WAL | Streamlit ouvre plusieurs connexions sans locks |
| Migrations | Table `schema_version` + scripts SQL versionnés | Pas d'Alembic, lisible et auditable |
| Validation | `dataclass` Python + contraintes `CHECK` SQL | Double barrière de validation |

**Conséquence** : `db.py` ne dépend que de la stdlib Python.

---

## 3. Schéma BDD

### 3.1 Vue d'ensemble

```
schema_version   (système, migrations)
defauts          (saisie opérateur)
  └── predictions    (top-3 IA pour chaque défaut, par axe M)
        └── feedback (retour opérateur sur la prédiction)
model_versions   (audit ISO / 8D : quel modèle, quand, sur quelles données)
```

### 3.2 Table `defauts`

| Colonne | Type | Contrainte | Note |
|---|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT | — |
| `timestamp` | TEXT | NOT NULL | ISO 8601 |
| `operator_id` | TEXT | NOT NULL | Identifiant libre |
| `ligne` | TEXT | NOT NULL | Ligne de production |
| `type_defaut` | TEXT | NOT NULL | Catégorie normalisée |
| `severite` | INTEGER | CHECK BETWEEN 1 AND 5 | Échelle Gascogne |
| `commentaire` | TEXT | nullable | Texte libre opérateur |
| `raw_data_snapshot` | TEXT (JSON) | nullable | Params machine au moment T |

### 3.3 Table `predictions`

| Colonne | Type | Contrainte | Note |
|---|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT | — |
| `defaut_id` | INTEGER | FK NOT NULL → defauts | — |
| `timestamp` | TEXT | NOT NULL | Quand l'inférence a tourné |
| `model_version_id` | INTEGER | FK NOT NULL → model_versions | Audit |
| `m_axis` | TEXT | CHECK IN ('Machine','Matiere','Methode') | Jamais MO ni Milieu |
| `rank` | INTEGER | CHECK BETWEEN 1 AND 3 | Position top-3 |
| `cause_label` | TEXT | NOT NULL | Cause prédite |
| `probabilite` | REAL | CHECK BETWEEN 0 AND 1 | Calibrée Platt/isotonic |
| `shap_features` | TEXT (JSON) | NOT NULL | Top-5 features déclencheuses |
| `inference_time_ms` | INTEGER | NOT NULL | Monitoring perf |
| `statut` | TEXT | CHECK IN ('proposee','inconnu_low_conf') | « JNSP » assumé |

### 3.4 Table `feedback`

| Colonne | Type | Contrainte | Note |
|---|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT | — |
| `prediction_id` | INTEGER | FK nullable → predictions | NULL si cause hors top-3 |
| `defaut_id` | INTEGER | FK NOT NULL → defauts | — |
| `timestamp` | TEXT | NOT NULL | — |
| `operator_id` | TEXT | NOT NULL | — |
| `was_correct` | INTEGER | CHECK IN (0,1) | — |
| `vraie_cause` | TEXT | nullable | Si différente |
| `commentaire` | TEXT | nullable | — |

### 3.5 Table `model_versions`

| Colonne | Type | Contrainte | Note |
|---|---|---|---|
| `id` | INTEGER | PK AUTOINCREMENT | — |
| `m_axis` | TEXT | CHECK IN ('Machine','Matiere','Methode') | — |
| `version` | TEXT | NOT NULL | Semver |
| `training_date` | TEXT | NOT NULL | — |
| `training_data_hash` | TEXT | NOT NULL | SHA-256 du dataset |
| `nb_samples` | INTEGER | NOT NULL | — |
| `top3_accuracy` | REAL | CHECK BETWEEN 0 AND 1 | Cible ≥ 0.80 |
| `onnx_path` | TEXT | NOT NULL | Chemin relatif |
| `active` | INTEGER | CHECK IN (0,1) | UNIQUE par axe quand =1 |

### 3.6 Table système `schema_version`

| Colonne | Type | Note |
|---|---|---|
| `version` | INTEGER | Numéro courant |
| `applied_at` | TEXT | Horodatage migration |

---

## 4. Contrat API publique de `db.py`

```python
# === INITIALISATION ===
def init_db(db_path: Path) -> None
def get_schema_version() -> int
def migrate_to(target_version: int) -> None

# === DÉFAUTS ===
def insert_defaut(defaut: Defaut) -> int
def get_defaut(defaut_id: int) -> Defaut | None
def list_defauts(limit: int = 50, ligne: str | None = None) -> list[Defaut]

# === PRÉDICTIONS ===
def insert_predictions(defaut_id: int, predictions: list[Prediction]) -> list[int]
def get_predictions_for_defaut(defaut_id: int) -> list[Prediction]

# === FEEDBACK ===
def insert_feedback(feedback: Feedback) -> int
def get_feedback_stats(model_version_id: int) -> FeedbackStats

# === MODEL VERSIONS ===
def register_model_version(model: ModelVersion) -> int
def activate_model(model_version_id: int) -> None
def get_active_model(m_axis: str) -> ModelVersion | None
def list_model_versions(m_axis: str | None = None) -> list[ModelVersion]

# === MAINTENANCE ===
def backup_db(target_path: Path) -> None
def vacuum() -> None
def prune_old_data(older_than_days: int) -> int
```

5 dataclasses associées : `Defaut`, `Prediction`, `Feedback`, `FeedbackStats`, `ModelVersion`.

---

## 5. Inversion appliquée à la BDD

| Mode d'échec | Garde-fou |
|---|---|
| Corruption fichier (coupure courant) | Mode WAL + `PRAGMA synchronous=NORMAL` + backup auto avant migration |
| Concurrence Streamlit multi-onglets | WAL + connexion par requête |
| SQL injection (commentaire libre) | 100 % requêtes paramétrées, règle absolue |
| Schéma évolue mal | Migrations versionnées + tests sur chaque migration |
| Foreign keys cassées | `PRAGMA foreign_keys=ON` systématique |
| Logs qui explosent | `prune_old_data()` + index sur `timestamp` |
| Modèle « actif » dupliqué | Index unique partiel : `UNIQUE (m_axis) WHERE active=1` |
| Données invalides | `dataclass` + `CHECK` SQL (double barrière) |
| Pas de rollback en erreur | Tout insert dans `with conn:` (context manager) |
| Stockage de secrets | Aucun secret en BDD. Jamais. |

---

## 6. Tests d'inversion à écrire (étape A-bis)

Fichier `tests/test_db.py` :

- `test_init_creates_all_tables_and_schema_v1`
- `test_insert_defaut_rejects_invalid_severity`
- `test_sql_injection_blocked_in_commentaire`
- `test_only_one_active_model_per_axis`
- `test_foreign_keys_enforced`
- `test_concurrent_writes_no_corruption` (2 threads, 100 inserts)
- `test_migration_v1_to_v2_preserves_data`
- `test_backup_creates_valid_db_copy`
- `test_prune_old_data_respects_threshold`

---

## 7. Étapes suivantes

| Étape | Livrable | Statut |
|---|---|---|
| **A** | Schéma + contrat API (ce document) | À VALIDER |
| **A-bis** | Implémentation `db.py` + dataclasses + tests d'inversion | bloquée par A |
| **B** | Pipeline preprocess + audit données client | bloquée par A-bis |
| **C** | Adapter `inference.py` au contrat (seuil, refus, SHAP, audit log) | bloquée par B |
| **D** | UI `ishikawa_ui.py` (formulaire 5 champs + top-3 + feedback) | bloquée par C |
| **E** | Tests d'inversion bout-en-bout | bloquée par D |
| **F** | Script ré-entraînement + monitoring drift | bloquée par E |
