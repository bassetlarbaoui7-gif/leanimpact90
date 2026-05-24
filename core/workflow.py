"""
core/workflow.py - machine a etats du projet AC.

Statuts possibles :
  brut → fiabilise → cadre → analyse → validation_en_cours
                                            ↓
                                       valide → solution_propose
                                                     ↓
                                              deploiement → clos
Branche refus :
  validation_en_cours → refuse  (renvoie en cadre)

L'API expose :
  - ProjetStatus (enum)
  - TRANSITIONS (dict)
  - can_transition(src, dst) -> bool
  - next_states(src) -> list[ProjetStatus]
  - transition(projet_id, dst) -> applique en base si valide
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

from core import db


class ProjetStatus(str, Enum):
    BRUT                 = "brut"
    FIABILISE            = "fiabilise"
    CADRE                = "cadre"
    ANALYSE              = "analyse"
    VALIDATION_EN_COURS  = "validation_en_cours"
    VALIDE               = "valide"
    REFUSE               = "refuse"
    SOLUTION_PROPOSE     = "solution_propose"
    DEPLOIEMENT          = "deploiement"
    CLOS                 = "clos"


# Transitions autorisees : src -> ensemble des dst possibles
TRANSITIONS: dict[ProjetStatus, set[ProjetStatus]] = {
    ProjetStatus.BRUT:                {ProjetStatus.FIABILISE},
    ProjetStatus.FIABILISE:           {ProjetStatus.CADRE},
    ProjetStatus.CADRE:               {ProjetStatus.ANALYSE},
    ProjetStatus.ANALYSE:             {ProjetStatus.VALIDATION_EN_COURS},
    ProjetStatus.VALIDATION_EN_COURS: {ProjetStatus.VALIDE,
                                       ProjetStatus.REFUSE},
    ProjetStatus.REFUSE:              {ProjetStatus.CADRE},
    ProjetStatus.VALIDE:              {ProjetStatus.SOLUTION_PROPOSE},
    ProjetStatus.SOLUTION_PROPOSE:    {ProjetStatus.DEPLOIEMENT},
    ProjetStatus.DEPLOIEMENT:         {ProjetStatus.CLOS},
    ProjetStatus.CLOS:                set(),
}


# Libelles humains pour l'UI
LABELS: dict[ProjetStatus, str] = {
    ProjetStatus.BRUT:                "Incident brut",
    ProjetStatus.FIABILISE:           "Fiabilise par Tech N+1",
    ProjetStatus.CADRE:               "Cadre",
    ProjetStatus.ANALYSE:             "Analyse IA",
    ProjetStatus.VALIDATION_EN_COURS: "Validation en cours",
    ProjetStatus.VALIDE:              "Valide",
    ProjetStatus.REFUSE:              "Refuse (a re-cadrer)",
    ProjetStatus.SOLUTION_PROPOSE:    "Solution proposee",
    ProjetStatus.DEPLOIEMENT:         "Deploiement en cours",
    ProjetStatus.CLOS:                "Clos",
}


def _normalize(s: ProjetStatus | str) -> ProjetStatus:
    if isinstance(s, ProjetStatus):
        return s
    try:
        return ProjetStatus(s)
    except ValueError:
        raise ValueError(f"Statut inconnu : {s!r}")


def can_transition(
    src: ProjetStatus | str,
    dst: ProjetStatus | str,
) -> bool:
    """True si la transition src -> dst est autorisee."""
    try:
        src_n = _normalize(src)
        dst_n = _normalize(dst)
    except ValueError:
        return False
    return dst_n in TRANSITIONS.get(src_n, set())


def next_states(src: ProjetStatus | str) -> list[ProjetStatus]:
    """Statuts suivants possibles depuis src."""
    src_n = _normalize(src)
    return sorted(TRANSITIONS.get(src_n, set()), key=lambda s: s.value)


def transition(
    projet_id: int,
    dst: ProjetStatus | str,
    *,
    db_path: Path | str | None = None,
) -> tuple[bool, str]:
    """
    Applique la transition en base apres validation.

    Retour :
      (True, "")               si OK
      (False, "raison")        sinon
    """
    projet = db.get_projet_ac(projet_id, db_path=db_path)
    if not projet:
        return False, f"Projet {projet_id} introuvable"
    src = projet.get("statut")
    if not can_transition(src, dst):
        dst_label = dst.value if isinstance(dst, ProjetStatus) else dst
        return False, (
            f"Transition interdite : {src} → {dst_label}. "
            f"Suivants possibles : "
            f"{[s.value for s in next_states(src)] or 'aucun'}"
        )
    dst_n = _normalize(dst)
    ok = db.update_projet_ac(projet_id, statut=dst_n.value, db_path=db_path)
    if not ok:
        return False, "Echec de l'update SQL (aucune ligne modifiee)"
    return True, ""
