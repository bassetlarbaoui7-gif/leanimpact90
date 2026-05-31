"""
core/cbr/classifier.py - Classification des noeuds 5 Pourquoi.

Ce module transforme un 5 Pourquoi FAIBLE en 5 Pourquoi FORT en appliquant
les regles des meilleures methodes (Toyota authentique, Apollo RCA, TapRoot).

3 capacites cles :
  1. classify_node      : symptome / condition / cause_directe / cause_racine
  2. detect_blame_shortcut : detecte "l'operateur a fait erreur" -> force a creuser
  3. should_stop_digging: critere d'arret intelligent (cause actionnable atteinte)

Approche : regles + dictionnaire metier FR industriel (transparent, debuggable,
marche des le 1er cas). On pourra ajouter du ML par-dessus en V2.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Types de noeuds (vocabulaire methodologique)
# ---------------------------------------------------------------------------
TYPE_SYMPTOME      = "symptome"        # visible mais consequence, pas cause
TYPE_CONDITION     = "condition"       # favorise mais ne cause pas seul
TYPE_CAUSE_DIRECTE = "cause_directe"   # maillon causal verifie
TYPE_CAUSE_RACINE  = "cause_racine"    # actionnable, systemique -> STOP

TYPE_LABELS = {
    TYPE_SYMPTOME:      "Symptome",
    TYPE_CONDITION:     "Condition",
    TYPE_CAUSE_DIRECTE: "Cause directe",
    TYPE_CAUSE_RACINE:  "Cause racine",
}


# ---------------------------------------------------------------------------
# Dictionnaires de marqueurs (FR industriel)
# ---------------------------------------------------------------------------
# Marqueurs d'une CAUSE RACINE ACTIONNABLE (systemique / organisationnelle).
# Si on les trouve -> on a atteint une cause sur laquelle on peut AGIR.
MARQUEURS_RACINE = (
    "absence de", "manque de", "pas de procedure", "pas de plan",
    "pas de revue", "non formalise", "non documente", "non a jour",
    "gouvernance", "budget", "sous-dimensionne", "sous-estime",
    "pas de responsable", "pas de pilote", "pas de matrice",
    "process non", "processus non", "politique", "strategie",
    "non identifie", "non maitrise", "non assigne",
    "pas de controle", "pas de sonde", "pas de formation",
    "formation incomplete", "formation insuffisante",
    "maintenance insuffisante", "maintenance preventive",
)

# Marqueurs de SYMPTOME (ce qu'on observe, conséquence visible)
MARQUEURS_SYMPTOME = (
    "defaut", "defauts", "casse", "cassure", "panne", "arret",
    "blocage", "fuite", "bruit", "vibration apparente", "visible",
    "on observe", "on constate", "apparait", "se produit",
    "rebut", "non conforme", "hors tolerance",
)

# Marqueurs de CONDITION (favorise mais ne cause pas seul)
MARQUEURS_CONDITION = (
    "quand il fait", "en ete", "en hiver", "par temps", "lorsque",
    "humidite ambiante", "temperature ambiante", "saison",
    "en periode de", "pendant les", "conditions ambiantes",
)

# Marqueurs de BLAME HUMAIN (piege a eviter - Toyota authentique)
# Si on s'arrete la -> on traite le symptome, pas la cause systeme.
MARQUEURS_BLAME_HUMAIN = (
    "operateur a oublie", "operateur a fait", "operateur n'a pas",
    "erreur humaine", "erreur de l'operateur", "faute de",
    "negligence", "inattention", "distraction",
    "mauvaise manipulation", "mauvais geste", "n'a pas verifie",
    "n'a pas respecte", "a mal", "oubli de", "etourderie",
    "manque d'attention", "manque de rigueur", "pas concentre",
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    repl = (("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
            ("à", "a"), ("â", "a"), ("î", "i"), ("ï", "i"),
            ("ô", "o"), ("ù", "u"), ("û", "u"), ("ç", "c"))
    for a, b in repl:
        t = t.replace(a, b)
    return t


def _contains_any(text: str, markers: tuple[str, ...]) -> list[str]:
    """Retourne la liste des marqueurs trouves dans le texte."""
    t = _norm(text)
    return [m for m in markers if m in t]


# ---------------------------------------------------------------------------
# 1. Classification du type de noeud
# ---------------------------------------------------------------------------
def classify_node(reponse: str, niveau: int = 1) -> dict:
    """
    Classe un noeud du 5 Pourquoi.

    Retour :
      {
        "type":        TYPE_*,
        "confidence":  0..1,
        "is_root":     bool,    (cause racine actionnable detectee)
        "markers":     [...],   (marqueurs trouves, pour transparence)
      }

    Logique de priorite :
      1. Si marqueurs racine -> CAUSE_RACINE (actionnable)
      2. Sinon si marqueurs condition -> CONDITION
      3. Sinon si marqueurs symptome ET niveau <= 2 -> SYMPTOME
      4. Sinon -> CAUSE_DIRECTE (le cas le plus courant au milieu de la chaine)
    """
    racine_m    = _contains_any(reponse, MARQUEURS_RACINE)
    condition_m = _contains_any(reponse, MARQUEURS_CONDITION)
    symptome_m  = _contains_any(reponse, MARQUEURS_SYMPTOME)

    if racine_m:
        return {
            "type": TYPE_CAUSE_RACINE,
            "confidence": min(1.0, 0.7 + 0.1 * len(racine_m)),
            "is_root": True,
            "markers": racine_m,
        }
    if condition_m:
        return {
            "type": TYPE_CONDITION,
            "confidence": 0.6,
            "is_root": False,
            "markers": condition_m,
        }
    if symptome_m and niveau <= 2:
        return {
            "type": TYPE_SYMPTOME,
            "confidence": 0.6,
            "is_root": False,
            "markers": symptome_m,
        }
    return {
        "type": TYPE_CAUSE_DIRECTE,
        "confidence": 0.55,
        "is_root": False,
        "markers": [],
    }


# ---------------------------------------------------------------------------
# 2. Detection du piege "blame humain"
# ---------------------------------------------------------------------------
def detect_blame_shortcut(reponse: str) -> dict:
    """
    Detecte si une reponse blame l'humain sans remonter au systeme.

    C'est LE piege #1 du 5 Pourquoi industriel. Toyota authentique :
    "l'humain a fait erreur" n'est JAMAIS une cause racine - il faut
    demander pourquoi le SYSTEME a permis cette erreur.

    Retour :
      {
        "is_blame":         bool,
        "markers":          [...],
        "relance_question": str | None,   (question pour creuser plus loin)
      }
    """
    markers = _contains_any(reponse, MARQUEURS_BLAME_HUMAIN)
    if not markers:
        return {"is_blame": False, "markers": [], "relance_question": None}

    relance = (
        "Pourquoi le systeme (procedure, formation, conception du poste, "
        "poka-yoke) a-t-il permis cette erreur humaine ? "
        "Une cause racine actionnable se trouve au niveau du systeme, "
        "pas de la personne."
    )
    return {
        "is_blame": True,
        "markers": markers,
        "relance_question": relance,
    }


# ---------------------------------------------------------------------------
# 3. Critere d'arret intelligent
# ---------------------------------------------------------------------------
def should_stop_digging(reponse: str, niveau: int) -> dict:
    """
    Decide si on a atteint une cause sur laquelle on peut AGIR (STOP),
    ou s'il faut continuer a creuser.

    Retour :
      {
        "stop":   bool,
        "raison": str,
        "type":   TYPE_*,
      }

    Regles :
      - Blame humain pur -> NE PAS s'arreter (creuser le systeme)
      - Cause racine actionnable (systemique) -> STOP
      - Niveau >= 5 ET cause directe -> STOP (profondeur suffisante)
      - Sinon -> continuer
    """
    blame = detect_blame_shortcut(reponse)
    if blame["is_blame"]:
        return {
            "stop": False,
            "raison": "Blame humain detecte - creuser le systeme sous-jacent",
            "type": TYPE_CAUSE_DIRECTE,
        }

    cls = classify_node(reponse, niveau)
    if cls["is_root"]:
        return {
            "stop": True,
            "raison": "Cause racine actionnable atteinte (systemique)",
            "type": TYPE_CAUSE_RACINE,
        }
    if niveau >= 5 and cls["type"] == TYPE_CAUSE_DIRECTE:
        return {
            "stop": True,
            "raison": "Profondeur suffisante (niveau 5) - cause directe stable",
            "type": TYPE_CAUSE_DIRECTE,
        }
    return {
        "stop": False,
        "raison": "Continuer a creuser - cause intermediaire",
        "type": cls["type"],
    }


# ---------------------------------------------------------------------------
# 4. Analyse complete d'un noeud (entry point)
# ---------------------------------------------------------------------------
def analyze_node(reponse: str, niveau: int = 1) -> dict:
    """
    Analyse complete : combine classification + blame + critere d'arret.
    Utilise par le moteur pour annoter chaque noeud genere.

    Retour :
      {
        "type":             TYPE_*,
        "type_label":       "Cause racine" | ...,
        "confidence":       0..1,
        "is_root":          bool,
        "is_blame":         bool,
        "should_stop":      bool,
        "relance_question": str | None,
        "raison_arret":     str,
        "markers":          [...],
      }
    """
    cls   = classify_node(reponse, niveau)
    blame = detect_blame_shortcut(reponse)
    stop  = should_stop_digging(reponse, niveau)

    # Le blame humain force le type a cause_directe (jamais racine)
    final_type = cls["type"]
    is_root    = cls["is_root"]
    if blame["is_blame"]:
        final_type = TYPE_CAUSE_DIRECTE
        is_root = False

    return {
        "type":             final_type,
        "type_label":       TYPE_LABELS.get(final_type, final_type),
        "confidence":       cls["confidence"],
        "is_root":          is_root,
        "is_blame":         blame["is_blame"],
        "should_stop":      stop["stop"] and not blame["is_blame"],
        "relance_question": blame["relance_question"],
        "raison_arret":     stop["raison"],
        "markers":          cls["markers"] + blame["markers"],
    }
