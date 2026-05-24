"""
F3 V2 - Moteur d'inference supervise multi-M.

Architecture :
  - Charge les artefacts produits par training.train_all_models() :
      * model_<m>.onnx       (inference rapide, embarquable PyInstaller)
      * model_<m>.pkl        (booster + LabelEncoder, necessaire a SHAP)
      * metadata.json        (features, classes, stats, importances)
  - Pour chaque defect_context, retourne une prediction par M (Machine,
    Matiere, Methode) avec top-3 causes triees + score de confiance.
  - Explainability via SHAP TreeExplainer si .pkl disponible, sinon
    fallback sur feature_importances_ ponderees par |z-score|.
  - Robustesse :
      * M dont l'ONNX manque -> indispo, pas de crash
      * features manquantes  -> imputation mediane + warning
      * metadata corrompu    -> erreur au chargement (fail fast)
      * NONE_LABEL filtree   -> non remontee dans le top-3

Aucune modification de ishikawa.py / home.py. La classe expose
RootCauseEngine pour pouvoir remplacer la V1 sans changer l'UI.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ishikawa import RcaCase, RootCauseEngine
from training import M_BRANCHES, NONE_LABEL


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modeles de donnees retournes a l'UI
# ---------------------------------------------------------------------------
@dataclass
class CausePrediction:
    """Une cause-racine candidate avec son score."""
    cause: str
    confidence: float          # probabilite [0, 1]


@dataclass
class ShapContribution:
    """Contribution d'une feature a une prediction."""
    feature: str
    value: float               # valeur observee
    contribution: float        # signe = sens, |.| = ampleur
    median: float              # valeur de reference (mediane historique)


@dataclass
class MPrediction:
    """Prediction complete pour une branche M."""
    m: str                              # "Machine" | "Matiere" | "Methode"
    available: bool                     # False si modele non charge
    top_causes: list[CausePrediction] = field(default_factory=list)
    is_none: bool = False               # cause dominante == NONE_LABEL
    explanation: list[ShapContribution] = field(default_factory=list)
    reason_unavailable: str = ""        # message si available == False


@dataclass
class MultiMPrediction:
    """Reponse complete : 3 predictions par M + meta."""
    predictions: dict[str, MPrediction] = field(default_factory=dict)

    def available_ms(self) -> list[str]:
        return [m for m, p in self.predictions.items() if p.available]


# ---------------------------------------------------------------------------
# Chargement d'un modele M (lazy, encapsule)
# ---------------------------------------------------------------------------
class _MModel:
    """Wrapper interne autour d'un modele M charge depuis disque."""

    def __init__(
        self,
        m: str,
        models_dir: Path,
        meta: dict,
    ) -> None:
        self.m = m
        self.features: list[str] = list(meta["features"])
        self.classes: list[str] = list(meta["classes"])
        self.feature_stats: dict[str, dict[str, float]] = meta.get(
            "feature_stats", {}
        )
        self.feature_importances: dict[str, float] = meta.get(
            "feature_importances", {}
        )

        # Session ONNX (toujours requise pour la prediction)
        import onnxruntime as ort
        onnx_path = models_dir / meta["onnx_file"]
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX manquant : {onnx_path}")
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        # Detection des noms d'I/O (peut varier selon onnxmltools)
        self._input_name = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        # Recherche de la sortie probabilites (vecteur > 1 col)
        self._proba_output = None
        for o in outputs:
            shape = o.shape or []
            if len(shape) == 2 and (shape[-1] is None
                                    or int(shape[-1] or 0) >= 2
                                    or len(self.classes) >= 2):
                self._proba_output = o.name
        if self._proba_output is None and outputs:
            # fallback sur le dernier output (souvent les probas)
            self._proba_output = outputs[-1].name

        # Booster pickle (optionnel, pour SHAP)
        self._booster = None
        self._encoder = None
        booster_file = meta.get("booster_file")
        if booster_file:
            booster_path = models_dir / booster_file
            if booster_path.exists():
                try:
                    with booster_path.open("rb") as f:
                        bundle = pickle.load(f)
                    self._booster = bundle.get("model")
                    self._encoder = bundle.get("encoder")
                except Exception as e:
                    logger.warning(
                        "Booster %s illisible (%s) - SHAP desactive pour %s",
                        booster_path, e, m,
                    )

        # Explainer SHAP cree a la demande (lazy, lourd a init)
        self._shap_explainer = None

    # ------------------------------------------------------------------
    def _build_input_vector(
        self, defect_context: dict[str, float],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Construit le vecteur d'entree dans l'ordre des features attendues.
        Retourne (vecteur 1xN, liste features manquantes imputees).
        """
        row = []
        imputed: list[str] = []
        for feat in self.features:
            val = defect_context.get(feat, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                stats = self.feature_stats.get(feat, {})
                row.append(float(stats.get("median", 0.0)))
                imputed.append(feat)
            else:
                try:
                    row.append(float(val))
                except (TypeError, ValueError):
                    row.append(float(self.feature_stats.get(feat, {})
                                       .get("median", 0.0)))
                    imputed.append(feat)
        if imputed:
            logger.info(
                "M=%s : %d feature(s) imputee(s) avec la mediane historique : %s",
                self.m, len(imputed), imputed,
            )
        return np.asarray(row, dtype=np.float32).reshape(1, -1), imputed

    # ------------------------------------------------------------------
    def predict(
        self,
        defect_context: dict[str, float],
        top_k: int = 3,
    ) -> MPrediction:
        """Prediction ONNX + tri top-k."""
        x, _ = self._build_input_vector(defect_context)
        outputs = self._session.run(
            [self._proba_output], {self._input_name: x},
        )
        proba = np.asarray(outputs[0]).reshape(1, -1)[0]

        # Aligner sur self.classes (l'ONNX expose les classes dans l'ordre
        # du LabelEncoder, donc identique a self.classes).
        if len(proba) != len(self.classes):
            # Defensif : si shape inattendue, on degrade proprement.
            return MPrediction(
                m=self.m, available=True, top_causes=[],
                is_none=True,
                reason_unavailable="Shape de sortie ONNX inattendue.",
            )

        # Tri decroissant
        order = np.argsort(-proba)
        ranked: list[tuple[str, float]] = [
            (self.classes[i], float(proba[i])) for i in order
        ]

        # Cause dominante == NONE_LABEL ?
        is_none = ranked[0][0] == NONE_LABEL

        # Top-k filtree de NONE_LABEL pour l'affichage operateur
        top_causes = [
            CausePrediction(cause=c, confidence=p)
            for c, p in ranked
            if c != NONE_LABEL
        ][:top_k]

        return MPrediction(
            m=self.m,
            available=True,
            top_causes=top_causes,
            is_none=is_none,
        )

    # ------------------------------------------------------------------
    def explain(
        self,
        defect_context: dict[str, float],
        top_features: int = 5,
    ) -> list[ShapContribution]:
        """
        Top-k features contributives pour la prediction dominante.
        Strategie :
          1. Si TreeExplainer SHAP utilisable -> contributions reelles
             pour la classe dominante.
          2. Sinon -> fallback : importance globale * |z-score| (signe = sens).
        """
        x, _ = self._build_input_vector(defect_context)

        # Tentative SHAP
        if self._booster is not None:
            try:
                import shap  # import lazy
                # On passe un DataFrame nomme pour eviter le warning sklearn
                # "X does not have valid feature names" et coller au schema
                # avec lequel LightGBM a ete fit.
                x_df = pd.DataFrame(x, columns=self.features)
                if self._shap_explainer is None:
                    self._shap_explainer = shap.TreeExplainer(self._booster)
                shap_values = self._shap_explainer.shap_values(x_df)

                # Identifier la classe dominante
                proba = self._booster.predict_proba(x_df)[0]
                dom_idx = int(np.argmax(proba))

                # shap_values shape varie selon version :
                #   list[ndarray(1, n_feat)] (n_classes elements)  -> ancien
                #   ndarray(1, n_feat, n_classes)                  -> recent
                if isinstance(shap_values, list):
                    contrib = np.asarray(shap_values[dom_idx]).reshape(-1)
                elif (isinstance(shap_values, np.ndarray)
                      and shap_values.ndim == 3):
                    contrib = shap_values[0, :, dom_idx]
                else:
                    contrib = np.asarray(shap_values).reshape(-1)

                return self._rank_contributions(
                    contrib, x[0], top_features,
                )
            except Exception as e:
                logger.warning("SHAP indisponible pour %s (%s) - fallback.",
                                self.m, e)

        # Fallback : importance globale * z-score signe
        contrib = []
        for i, feat in enumerate(self.features):
            imp = float(self.feature_importances.get(feat, 0.0))
            stats = self.feature_stats.get(feat, {})
            med = float(stats.get("median", 0.0))
            std = float(stats.get("std", 1.0)) or 1.0
            z = (float(x[0, i]) - med) / std
            contrib.append(imp * z)  # signe = sens du delta
        return self._rank_contributions(
            np.asarray(contrib), x[0], top_features,
        )

    # ------------------------------------------------------------------
    def _rank_contributions(
        self,
        contributions: np.ndarray,
        values: np.ndarray,
        top_k: int,
    ) -> list[ShapContribution]:
        """Trie par |contribution| decroissante et retourne le top-k."""
        order = np.argsort(-np.abs(contributions))
        out: list[ShapContribution] = []
        for i in order[:top_k]:
            feat = self.features[int(i)]
            stats = self.feature_stats.get(feat, {})
            out.append(ShapContribution(
                feature=feat,
                value=float(values[int(i)]),
                contribution=float(contributions[int(i)]),
                median=float(stats.get("median", 0.0)),
            ))
        return out


# ---------------------------------------------------------------------------
# Engine multi-M
# ---------------------------------------------------------------------------
class SupervisedRootCauseEngine(RootCauseEngine):
    """
    Moteur F3 V2 supervise multi-M.

    Implemente l'interface RootCauseEngine pour pouvoir remplacer la V1
    TF-IDF dans ishikawa_ui.py sans changer l'UI. Les methodes specifiques
    multi-M (`predict_multi`, `explain_multi`) sont exposees en plus.
    """

    def __init__(self, models_dir: str | Path) -> None:
        super().__init__(min_history=0)  # pas de fit() requis
        self.models_dir = Path(models_dir)
        self._models: dict[str, _MModel] = {}
        self._unavailable_reasons: dict[str, str] = {}
        self._meta_global: dict = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Charge metadata.json + chaque modele M disponible."""
        meta_path = self.models_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json introuvable dans {self.models_dir}"
            )
        try:
            self._meta_global = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"metadata.json corrompu : {e}") from e

        models_meta = self._meta_global.get("models", {})

        for m in M_BRANCHES:
            if m not in models_meta:
                self._unavailable_reasons[m] = (
                    "Modele non entraine (volume insuffisant a l'historique)"
                )
                continue
            try:
                self._models[m] = _MModel(
                    m=m, models_dir=self.models_dir, meta=models_meta[m],
                )
            except Exception as e:
                logger.error("Chargement %s impossible : %s", m, e)
                self._unavailable_reasons[m] = f"Erreur de chargement : {e}"

    # ------------------------------------------------------------------
    def is_available(self) -> dict[str, bool]:
        """Pour l'UI : drapeau dispo/indispo par M."""
        return {m: (m in self._models) for m in M_BRANCHES}

    def reasons_unavailable(self) -> dict[str, str]:
        return dict(self._unavailable_reasons)

    def get_features(self, m: str) -> list[str]:
        """Liste des features attendues par le modele M (utile a l'UI)."""
        if m not in self._models:
            return []
        return list(self._models[m].features)

    # ------------------------------------------------------------------
    def predict_multi(
        self,
        defect_context: dict[str, float],
        top_k: int = 3,
    ) -> MultiMPrediction:
        """Une prediction par M disponible."""
        result = MultiMPrediction()
        for m in M_BRANCHES:
            if m in self._models:
                result.predictions[m] = self._models[m].predict(
                    defect_context, top_k=top_k,
                )
            else:
                result.predictions[m] = MPrediction(
                    m=m, available=False,
                    reason_unavailable=self._unavailable_reasons.get(
                        m, "Modele indisponible",
                    ),
                )
        return result

    def explain_multi(
        self,
        defect_context: dict[str, float],
        top_features: int = 5,
    ) -> dict[str, list[ShapContribution]]:
        """Explications SHAP (ou fallback) par M disponible."""
        out: dict[str, list[ShapContribution]] = {}
        for m, model in self._models.items():
            out[m] = model.explain(defect_context, top_features=top_features)
        return out

    # ------------------------------------------------------------------
    # Compat interface RootCauseEngine (V1)
    # ------------------------------------------------------------------
    def fit(self, cases: Sequence[RcaCase]) -> None:
        """No-op : le modele est deja entraine offline."""
        return

    def is_trained(self) -> bool:
        return len(self._models) > 0

    def suggest(
        self,
        defect_type: str,
        context: str,
        parameters: dict[str, float],
        top_k: int = 5,
    ) -> list:
        """
        Compat V1 : aplatit les predictions multi-M en CauseSuggestion.
        Permet de brancher SupervisedRootCauseEngine la ou la V1 etait
        utilisee, sans modifier l'UI.
        """
        from ishikawa import CauseSuggestion
        result = self.predict_multi(parameters, top_k=top_k)
        flat: list[CauseSuggestion] = []
        for m, pred in result.predictions.items():
            if not pred.available or pred.is_none:
                continue
            for cp in pred.top_causes:
                flat.append(CauseSuggestion(
                    branch=m,
                    cause=cp.cause,
                    confidence=cp.confidence,
                    explanation=f"Modele supervise {m} (top-3)",
                ))
        flat.sort(key=lambda s: s.confidence, reverse=True)
        return flat[:top_k]
