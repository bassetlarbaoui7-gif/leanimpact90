"""
F3 V2 - Pipeline d'entrainement supervise par M (Machine, Matiere, Methode).

Architecture :
  1. classify_cause_to_m()   - heuristique mots-cles, mappe une cause-racine
                                textuelle vers une ou plusieurs M (multi-label).
  2. build_per_m_dataset()   - genere (X, y) pour une M donnee a partir
                                d'un registre global de defauts.
  3. train_m_model()         - entraine LightGBM ; refuse si volume insuffisant.
  4. export_to_onnx()        - serialise le modele en ONNX (portable, .exe).
  5. train_all_models()      - orchestre les 3 entrainements + rapport global.

Generateur synthetique :
  generate_synthetic_global_history() pour demo / dev sans donnees client.

Aucune dependance UI. Aucune modification de ishikawa.py / home.py.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
M_BRANCHES: tuple[str, ...] = ("Machine", "Matiere", "Methode")

DEFAULT_MIN_CASES_PER_M = 150       # volume minimal pour entrainer
DEFAULT_MIN_CLASSES_PER_M = 3       # nombre minimal de causes distinctes
DEFAULT_MIN_EXAMPLES_PER_CLASS = 5  # par classe, sinon fusion en "_aucune_"

NONE_LABEL = "_aucune_"  # marqueur "M non responsable du defaut"


# Heuristique mots-cles (extensible). Une cause peut tomber dans plusieurs M.
KEYWORDS_BY_M: dict[str, list[str]] = {
    "Machine": [
        r"\btemp[eé]rature\b", r"\bpression\b", r"\bbus[eo]\b",
        r"\bmoteur\b", r"\bv[eé]rin\b", r"\bcapteur\b",
        r"\bvis\b", r"\busure\b", r"\bencrass[eé]\b", r"\boutil\b",
        r"\bd[eé]bit\b", r"\bjeu\b", r"\bfuite\b",
    ],
    "Matiere": [
        r"\blot\b", r"\bfournisseur\b", r"\bhumidit[eé]\b",
        r"\bdensit[eé]\b", r"\bkraft\b", r"\bcolle\b", r"\bencre\b",
        r"\bmati[eè]re\b", r"\bgrammage\b", r"\bp[âa]te\b",
        r"\bviscosit[eé]\b", r"\bqualit[eé] mati[eè]re\b",
    ],
    "Methode": [
        r"\brecette\b", r"\br[eé]glage\b", r"\bconsigne\b",
        r"\bproc[eé]dure\b", r"\bgamme\b", r"\bvitesse\b",
        r"\bm[eé]thode\b", r"\bparam[eè]trage\b", r"\bprogramme\b",
        r"\boperat[eo]ire\b",
    ],
}


# ---------------------------------------------------------------------------
# Dataclass de rapport
# ---------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    n_total: int = 0
    n_real_cases: int = 0      # cas non "_aucune_"
    n_classes: int = 0
    accuracy: float = 0.0
    top3_accuracy: float = 0.0
    classes: list[str] = field(default_factory=list)


@dataclass
class TrainingReport:
    timestamp: str = ""
    history_size: int = 0
    models_trained: dict[str, ModelMetrics] = field(default_factory=dict)
    models_skipped: dict[str, str] = field(default_factory=dict)  # m -> raison
    output_dir: str = ""

    def summary(self) -> str:
        lines = [
            f"Entrainement F3 V2 - {self.timestamp}",
            f"  Historique : {self.history_size} cas",
            f"  Modeles entraines : {len(self.models_trained)} / {len(M_BRANCHES)}",
        ]
        for m, mm in self.models_trained.items():
            lines.append(
                f"    [{m}] {mm.n_real_cases} cas, {mm.n_classes} causes, "
                f"accuracy={mm.accuracy:.2%}, top3={mm.top3_accuracy:.2%}"
            )
        for m, raison in self.models_skipped.items():
            lines.append(f"    [{m}] SKIP - {raison}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Classification cause-racine -> M
# ---------------------------------------------------------------------------
def classify_cause_to_m(cause_text: str) -> set[str]:
    """
    Mappe un texte de cause-racine vers les M concernees (multi-label).
    Retourne un set vide si aucune correspondance.
    """
    if not cause_text or not isinstance(cause_text, str):
        return set()
    text = cause_text.lower()
    found: set[str] = set()
    for m, patterns in KEYWORDS_BY_M.items():
        for pat in patterns:
            if re.search(pat, text):
                found.add(m)
                break
    return found


# ---------------------------------------------------------------------------
# 2. Construction du dataset par M
# ---------------------------------------------------------------------------
def build_per_m_dataset(
    history_df: pd.DataFrame,
    m: str,
    feature_prefix: str | None = None,
    label_col: str = "cause_racine",
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Pour la branche `m`, retourne (X, y, feature_cols) :
      - X        = colonnes numeriques au prefix `M_{m.lower()}_*`
      - y        = cause-racine textuelle si la cause concerne cette M,
                   sinon NONE_LABEL
      - features = liste des colonnes utilisees
    """
    if m not in M_BRANCHES:
        raise ValueError(f"M inconnue : {m}")
    if feature_prefix is None:
        feature_prefix = f"M_{m.lower()}_"

    feature_cols = [c for c in history_df.columns
                    if c.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(
            f"Aucune colonne avec prefix '{feature_prefix}' "
            f"dans l'historique."
        )
    if label_col not in history_df.columns:
        raise ValueError(f"Colonne label '{label_col}' absente.")

    causes = history_df[label_col].fillna("").astype(str)
    y = np.array([
        cause if m in classify_cause_to_m(cause) else NONE_LABEL
        for cause in causes
    ])
    X = history_df[feature_cols].copy()
    # Cast en numerique (resilience aux strings)
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # On ne dropna pas ici : LightGBM gere les NaN nativement
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# 3. Entrainement d'un modele par M
# ---------------------------------------------------------------------------
def train_m_model(
    X: pd.DataFrame,
    y: np.ndarray,
    m_name: str,
    min_cases: int = DEFAULT_MIN_CASES_PER_M,
    min_classes: int = DEFAULT_MIN_CLASSES_PER_M,
    min_examples_per_class: int = DEFAULT_MIN_EXAMPLES_PER_CLASS,
    random_state: int = 42,
) -> tuple[lgb.LGBMClassifier | None, LabelEncoder | None,
           ModelMetrics | None, str]:
    """
    Entraine LightGBM sur (X, y).
    Retourne (model, encoder, metrics, raison_skip_si_skip).
    Si volume insuffisant : retourne (None, None, None, raison).
    """
    counts = pd.Series(y).value_counts()

    # Fusion des classes trop rares dans NONE_LABEL pour stabilite
    rare_real = counts[(counts < min_examples_per_class)
                        & (counts.index != NONE_LABEL)].index.tolist()
    if rare_real:
        y = np.array([NONE_LABEL if yi in rare_real else yi for yi in y])

    n_real_cases = int(np.sum(y != NONE_LABEL))
    real_classes = sorted(set(y) - {NONE_LABEL})
    n_real_classes = len(real_classes)

    if n_real_cases < min_cases:
        return None, None, None, (
            f"Volume insuffisant : {n_real_cases} cas reels "
            f"(min requis {min_cases})"
        )
    if n_real_classes < min_classes:
        return None, None, None, (
            f"Classes distinctes insuffisantes : {n_real_classes} "
            f"(min requis {min_classes})"
        )

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Stratify si possible
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc, test_size=0.2,
            stratify=y_enc, random_state=random_state,
        )
    except ValueError:
        # certaines classes < 2 -> fallback sans stratify
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc, test_size=0.2, random_state=random_state,
        )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=5,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    accuracy = float(accuracy_score(y_te, y_pred))

    # Top-3 accuracy (k borne par le nombre de classes)
    proba = model.predict_proba(X_te)
    k = min(3, len(le.classes_))
    try:
        top3 = float(top_k_accuracy_score(
            y_te, proba, k=k, labels=np.arange(len(le.classes_))
        ))
    except Exception:
        top3 = float(accuracy)

    metrics = ModelMetrics(
        n_total=len(y),
        n_real_cases=n_real_cases,
        n_classes=len(le.classes_),
        accuracy=accuracy,
        top3_accuracy=top3,
        classes=list(le.classes_),
    )
    return model, le, metrics, ""


# ---------------------------------------------------------------------------
# 4. Export ONNX
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: lgb.LGBMClassifier,
    feature_cols: list[str],
    output_path: str | Path,
) -> None:
    """Convertit un LGBMClassifier en ONNX et l'ecrit sur disque."""
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = convert_lightgbm(
        model, initial_types=initial_types,
        zipmap=False,        # output direct (proba), pas le wrapper Map
        target_opset=12,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onnx_model.SerializeToString())


# ---------------------------------------------------------------------------
# 5. Orchestrateur
# ---------------------------------------------------------------------------
def train_all_models(
    history_df: pd.DataFrame,
    output_dir: str | Path = "./models",
    label_col: str = "cause_racine",
    min_cases: int = DEFAULT_MIN_CASES_PER_M,
) -> TrainingReport:
    """
    Boucle sur les 3 M, entraine ce qui peut l'etre, ecrit les ONNX
    dans output_dir, retourne un TrainingReport.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = TrainingReport(
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        history_size=len(history_df),
        output_dir=str(out.resolve()),
    )

    metadata = {
        "trained_at": report.timestamp,
        "history_size": report.history_size,
        "models": {},
    }

    for m in M_BRANCHES:
        try:
            X, y, feats = build_per_m_dataset(
                history_df, m=m, label_col=label_col,
            )
        except ValueError as e:
            report.models_skipped[m] = f"Dataset non constructible : {e}"
            continue

        model, le, metrics, skip_reason = train_m_model(
            X, y, m_name=m, min_cases=min_cases,
        )
        if model is None:
            report.models_skipped[m] = skip_reason
            continue

        # Export ONNX (inference rapide, embarquable .exe)
        onnx_path = out / f"model_{m.lower()}.onnx"
        export_to_onnx(model, feats, onnx_path)

        # Export booster pickle (necessaire a SHAP TreeExplainer)
        # Optionnel a l'inference : si absent, fallback feature_importances.
        booster_path = out / f"model_{m.lower()}.pkl"
        with booster_path.open("wb") as f:
            pickle.dump({"model": model, "encoder": le}, f)

        # Statistiques par feature pour imputation et explication
        # (median = valeur a substituer si donnee manquante a l'inference)
        feature_stats = {}
        for col in feats:
            series = pd.to_numeric(X[col], errors="coerce").dropna()
            feature_stats[col] = {
                "median": float(series.median()) if len(series) else 0.0,
                "std": float(series.std()) if len(series) > 1 else 1.0,
            }

        # Importances globales (fallback si SHAP indisponible)
        importances = dict(zip(feats, model.feature_importances_.tolist()))

        report.models_trained[m] = metrics

        metadata["models"][m] = {
            "onnx_file": onnx_path.name,
            "booster_file": booster_path.name,
            "features": feats,
            "feature_stats": feature_stats,
            "feature_importances": importances,
            "classes": list(le.classes_),
            "metrics": {
                "accuracy": metrics.accuracy,
                "top3_accuracy": metrics.top3_accuracy,
                "n_real_cases": metrics.n_real_cases,
                "n_classes": metrics.n_classes,
            },
        }

    # Ecriture du metadata.json (necessaire a l'inference pour decoder
    # les classes et savoir quelles features attendre)
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return report


# ---------------------------------------------------------------------------
# 6. Generateur synthetique (pour demo client sans historique reel)
# ---------------------------------------------------------------------------
SYNTH_CAUSES = {
    "Machine": [
        "Temperature colle basse",
        "Pression vis trop elevee",
        "Buse encrassee",
        "Capteur defaillant",
        "Usure outil de coupe",
    ],
    "Matiere": [
        "Lot kraft humide fournisseur A",
        "Densite lot atypique",
        "Grammage hors specification",
        "Colle viscosite anormale",
    ],
    "Methode": [
        "Mauvaise recette appliquee",
        "Vitesse consigne incorrecte",
        "Procedure de reglage non suivie",
    ],
    "Multi": [
        "Lot kraft humide + temperature colle basse",
        "Buse encrassee + mauvaise recette appliquee",
        "Pression vis trop elevee + grammage hors specification",
    ],
}


def generate_synthetic_global_history(
    n: int = 400, seed: int = 42,
) -> pd.DataFrame:
    """
    Genere un registre global realiste pour demo F3 V2.
    Schema :
      defect_id, defect_type, timestamp, cause_racine,
      M_machine_*, M_matiere_*, M_methode_*

    Distribution :
      40% Machine, 30% Matiere, 15% Methode, 15% Multi
    """
    rng = np.random.default_rng(seed)
    cats = rng.choice(
        ["Machine", "Matiere", "Methode", "Multi"],
        size=n, p=[0.40, 0.30, 0.15, 0.15],
    )

    rows = []
    base_ts = pd.Timestamp("2025-01-01")
    for i, cat in enumerate(cats):
        cause = str(rng.choice(SYNTH_CAUSES[cat]))
        cause_lower = cause.lower()

        # Parametres Machine (3 colonnes)
        temp_machine = rng.normal(180, 3)
        pression_machine = rng.normal(4.5, 0.15)
        vitesse_vis = rng.normal(120, 5)
        if "temperature colle basse" in cause_lower:
            temp_machine = rng.normal(170, 2)
        if "pression vis" in cause_lower and "elevee" in cause_lower:
            pression_machine = rng.normal(5.3, 0.15)
        if "buse encrassee" in cause_lower:
            pression_machine = rng.normal(5.0, 0.25)
            vitesse_vis = rng.normal(105, 3)
        if "capteur defaillant" in cause_lower:
            temp_machine = rng.normal(150, 20)  # bruit fort
        if "usure outil" in cause_lower:
            vitesse_vis = rng.normal(95, 4)

        # Parametres Matiere (3 colonnes)
        humidite_kraft = rng.normal(7.0, 0.5)
        densite_lot = rng.normal(120, 3)
        grammage = rng.normal(80, 2)
        if "humide" in cause_lower:
            humidite_kraft = rng.normal(11.5, 0.8)
        if "densite" in cause_lower:
            densite_lot = rng.normal(135, 2)
        if "grammage" in cause_lower:
            grammage = rng.normal(90, 3)
        if "viscosite" in cause_lower:
            densite_lot = rng.normal(130, 2)

        # Parametres Methode (3 colonnes)
        vitesse_consigne = rng.normal(1200, 30)
        temp_cible = rng.normal(180, 1)
        pression_cible = rng.normal(4.5, 0.05)
        if "recette" in cause_lower:
            temp_cible = rng.normal(165, 1.5)
        if "vitesse consigne" in cause_lower:
            vitesse_consigne = rng.normal(1500, 40)
        if "procedure" in cause_lower or "reglage" in cause_lower:
            pression_cible = rng.normal(4.2, 0.1)

        rows.append({
            "defect_id": i + 1,
            "defect_type": str(rng.choice([
                "colle_insuffisante", "tache", "deformation", "rupture",
            ])),
            "timestamp": base_ts + pd.Timedelta(hours=i),
            "cause_racine": cause,
            "M_machine_temperature": temp_machine,
            "M_machine_pression": pression_machine,
            "M_machine_vitesse_vis": vitesse_vis,
            "M_matiere_humidite_kraft": humidite_kraft,
            "M_matiere_densite_lot": densite_lot,
            "M_matiere_grammage": grammage,
            "M_methode_vitesse_consigne": vitesse_consigne,
            "M_methode_temperature_cible": temp_cible,
            "M_methode_pression_cible": pression_cible,
        })

    return pd.DataFrame(rows)
