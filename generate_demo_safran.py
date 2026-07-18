"""
generate_demo_safran.py - Prepare la demo Safran Fougeres (video POC).

Ce script fait 2 choses :

  1. Genere demo_safran_fougeres.xlsx : 30 jours de donnees process
     d'une ligne CMS (four de refusion) avec une histoire causale implantee :
       - Maintenance preventive du four le 29 juin 2026
       - Remontage incorrect de l'element chauffant zone 6 (connecteur)
       - -> temp zone 6 derive lentement (-0.55 C/jour)
       - -> temperature pic et temps au-dessus du liquidus chutent
       - -> defauts de brasage (NC AOI) x10 en 3 semaines
       - Bonus : equipe C (nuit) roule le convoyeur plus vite (ecart shifts)
     Signaux detectables par LI90 :
       - EWMA/CUSUM + Nelson (tendance) sur temp_zone6_c et temp_pic_profil_c
       - Correlation forte defauts <-> temp_zone6_c (negative)
       - Correlation defauts <-> jours_depuis_maint_four (positive)
       - Kruskal-Wallis significatif sur vitesse_convoyeur (equipe C)

  2. --seed-db : pre-charge li90.db avec 3 anciens cas RCA electronique
     valides et clos (brasage froid, voids BGA, tombstoning) pour que le
     moteur CBR propose de vrais cas similaires avec confiance elevee
     pendant la demo (au lieu des templates generiques demarrage froid).

Usage (depuis le dossier li90_mvp) :
    python generate_demo_safran.py             # genere le xlsx
    python generate_demo_safran.py --seed-db   # xlsx + seed la base CBR
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_XLSX = HERE / "demo_safran_fougeres.xlsx"
OUT_XLSX_FIX = HERE / "demo_safran_fougeres_apres_correctif.xlsx"

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Dataset process ligne CMS
# ---------------------------------------------------------------------------
START = datetime(2026, 6, 18)          # J1
N_DAYS = 30                            # 18 juin -> 17 juillet
MAINT_DAY = 11                         # index du 29 juin (maintenance four)
SHIFTS = [("A", 5), ("B", 13), ("C", 21)]   # equipe, heure de debut
LOTS_PER_SHIFT = 3                     # 1 lot toutes les ~2h30


FIX_DAY = 29                           # index du 17 juillet (re-serrage)


def gen_dataset(recovery: bool = False) -> pd.DataFrame:
    """
    recovery=False : 30 jours jusqu'au 17/07 (etat au moment de l'alerte).
    recovery=True  : 37 jours ; a partir du 17/07 le correctif est applique
                     (zone 6 revient a 235 C, defauts retombent) -> fichier
                     de la preuve avant/apres a importer en fin de video.
    """
    n_days = 37 if recovery else N_DAYS
    rows = []
    for d in range(n_days):
        day = START + timedelta(days=d)
        # Derive thermique zone 6 apres la maintenance (connecteur desserre),
        # stoppee au jour du correctif si recovery.
        if d <= MAINT_DAY or (recovery and d >= FIX_DAY):
            drift = 0.0
        else:
            drift = 0.55 * (d - MAINT_DAY)
        if recovery and d >= FIX_DAY:
            jours_maint = d - FIX_DAY
        else:
            jours_maint = (MAINT_DAY - d + 14) if d <= MAINT_DAY \
                else d - MAINT_DAY

        for equipe, h0 in SHIFTS:
            # Equipe C : consigne convoyeur non homogene (93 au lieu de 90)
            v_base = 93.0 if equipe == "C" else 90.0
            for lot in range(LOTS_PER_SHIFT):
                ts = day + timedelta(hours=h0 + 2.5 * lot)

                temp_z6 = 235.0 - drift + RNG.normal(0, 0.8)
                temp_pic = 245.0 + 0.9 * (temp_z6 - 235.0) + RNG.normal(0, 0.7)
                vitesse = v_base + RNG.normal(0, 0.6)
                tal = (62.0 + 0.75 * (temp_pic - 245.0)
                       - 0.35 * (vitesse - 90.0) + RNG.normal(0, 1.1))
                humidite = 52.0 + 4.0 * np.sin(d / 5.0) + RNG.normal(0, 2.0)

                lam = (0.35
                       + 0.55 * max(0.0, 243.0 - temp_pic)
                       + (0.35 if equipe == "C" else 0.0)
                       + 0.05 * max(0.0, humidite - 55.0))
                defauts = int(RNG.poisson(lam))
                cartes = int(np.clip(RNG.normal(118, 6), 100, 132))

                rows.append({
                    "timestamp": ts,
                    "ligne": "CMS 2",
                    "equipe": equipe,
                    "ref_carte": "CALC-3040",
                    "cartes_produites": cartes,
                    "defauts_brasage": defauts,
                    "temp_zone1_c": 140.0 + RNG.normal(0, 0.7),
                    "temp_zone2_c": 150.0 + RNG.normal(0, 0.7),
                    "temp_zone3_c": 162.0 + RNG.normal(0, 0.8),
                    "temp_zone4_c": 178.0 + RNG.normal(0, 0.8),
                    "temp_zone5_c": 198.0 + RNG.normal(0, 0.9),
                    "temp_zone6_c": round(temp_z6, 2),
                    "temp_zone7_c": 249.0 + RNG.normal(0, 0.8),
                    "temp_pic_profil_c": round(temp_pic, 2),
                    "temps_liquidus_s": round(tal, 1),
                    "vitesse_convoyeur_cm_min": round(vitesse, 2),
                    "epaisseur_pate_um": 120.0 + RNG.normal(0, 3.5),
                    "viscosite_pate_kcps": 185.0 + RNG.normal(0, 7.0),
                    "humidite_atelier_pct": round(humidite, 1),
                    "temp_atelier_c": 22.5 + RNG.normal(0, 0.8),
                    "o2_azote_ppm": 800.0 + RNG.normal(0, 120.0),
                    "age_pate_h": float(np.clip(RNG.normal(8, 5), 0.5, 24)),
                    "jours_depuis_maint_four": jours_maint,
                })
    df = pd.DataFrame(rows)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(2)
    return df


# ---------------------------------------------------------------------------
# 2. Seed CBR : 3 anciens cas electronique valides + clos
# ---------------------------------------------------------------------------
def _branch(chain: list[tuple[str, str]], conf_racine: float,
            conf_autres: float = 0.75) -> list[dict]:
    """Construit une branche 5 Pourquoi. Le dernier noeud = cause racine."""
    nodes = []
    for i, (q, r) in enumerate(chain, start=1):
        last = i == len(chain)
        nodes.append({
            "niveau": i,
            "question": q,
            "reponse": r,
            "type_noeud": "cause_racine" if last else "cause_directe",
            "est_cause_racine": last,
            "confidence": conf_racine if last else conf_autres,
        })
    return nodes


SEED_CASES = [
    {
        "machine": "Ligne CMS 1 - Four de refusion",
        "description": ("Joints brases froids detectes a l'AOI sur cartes "
                        "calculateur, hausse progressive apres maintenance "
                        "preventive du four de refusion, profil thermique "
                        "derive."),
        "type_incident": "Defaut produit",
        "titre": "RCA brasage froid four refusion CMS 1",
        "materiel": [
            ("Pourquoi des joints brases froids ?",
             "Temperature pic du profil de refusion insuffisante"),
            ("Pourquoi le pic est insuffisant ?",
             "La zone de refusion du four a derive apres la maintenance"),
            ("Pourquoi la zone a derive ?",
             "Element chauffant remonte avec un connecteur mal serre"),
            ("Pourquoi le defaut n'a pas ete vu au redemarrage ?",
             "Aucune verification du profil thermique apres maintenance"),
            ("Pourquoi aucune verification ?",
             "La checklist de redemarrage ne contient pas de passage "
             "de plaque de profilage"),
        ],
    },
    {
        "machine": "Ligne CMS 2 - Serigraphie",
        "description": ("Voids excessifs sur billes BGA calculateur detectes "
                        "en inspection rayons X, epaisseur de pate a braser "
                        "instable en sortie serigraphie."),
        "type_incident": "Defaut produit",
        "titre": "RCA voids BGA serigraphie CMS 2",
        "materiel": [
            ("Pourquoi des voids excessifs sous BGA ?",
             "Depot de pate a braser irregulier sur les plages"),
            ("Pourquoi le depot est irregulier ?",
             "Racle de serigraphie usee au-dela de la limite"),
            ("Pourquoi la racle est usee ?",
             "Remplacement declenche a la casse, pas au compteur"),
            ("Pourquoi pas de remplacement preventif ?",
             "Le compteur de cycles racle n'est pas suivi"),
            ("Pourquoi pas suivi ?",
             "Le plan de controle SPI ne couvre pas l'usure outillage"),
        ],
    },
    {
        "machine": "Ligne CMS 1 - Four de refusion",
        "description": ("Tombstoning sur resistances 0402, lot de cartes "
                        "entrees-sorties, prechauffage insuffisant apres "
                        "changement de reference de pate a braser."),
        "type_incident": "Defaut produit",
        "titre": "RCA tombstoning 0402 prechauffage",
        "materiel": [
            ("Pourquoi du tombstoning sur les 0402 ?",
             "Fusion asymetrique des deux plages du composant"),
            ("Pourquoi une fusion asymetrique ?",
             "Gradient de prechauffage trop rapide pour la nouvelle pate"),
            ("Pourquoi le gradient est inadapte ?",
             "Profil du four non requalifie apres changement de pate"),
            ("Pourquoi pas de requalification ?",
             "Le changement de reference n'a pas declenche d'essai profil"),
            ("Pourquoi pas de declenchement ?",
             "Aucune regle ne lie changement matiere et requalification "
             "du profil"),
        ],
    },
]


def seed_db() -> None:
    sys.path.insert(0, str(HERE))
    from core import db
    from core.cbr import case_base as cb
    from core.cbr.path_engine import generate_full_tree, save_tree_to_db
    from core.cbr.feedback import (
        record_validation_outcome, enrich_from_validated_projet,
    )

    db.init_db()

    for case in SEED_CASES:
        iid = db.create_incident(
            case["machine"],
            case["description"],
            operateur_nom="Demo",
            type_incident=case["type_incident"],
            severite="haute",
            cree_par_role="operator",
        )
        db.update_incident_statut(iid, "fiabilise")
        pid = db.create_projet_ac(
            case["titre"],
            incident_id=iid,
            cree_par="Castro",
            cree_par_role="ac_manager",
        )
        db.update_incident_statut(iid, "en_projet")

        # Arbre : templates + branche Materiel remplacee par le vrai chemin
        result = generate_full_tree(iid)
        tree = result["tree"]
        tree["Materiel"] = _branch(case["materiel"], conf_racine=0.90)
        save_tree_to_db(pid, tree)
        db.update_projet_ac(pid, statut="analyse")

        # Validation AC sur les causes racines
        arbre = cb.list_chemins_projet(pid)
        racines = arbre[arbre["est_cause_racine"] == 1]
        for _, r in racines.iterrows():
            cb.add_feedback(int(r["id"]), +1, role_valideur="ac_manager",
                            commentaire="Validation analyse (cas historique)")

        # Validation distribuee Tech + Prod -> valide -> clos
        db.update_projet_ac(pid, statut="validation_en_cours")
        for role, nom in (("technician", "Tech1"), ("production", "Prod1")):
            db.add_validation(pid, role, "valide", nom_valideur=nom)
            record_validation_outcome(pid, role, "valide", nom_valideur=nom)
        db.update_projet_ac(pid, statut="valide")
        enrich_from_validated_projet(pid)
        db.update_projet_ac(pid, statut="clos")
        print(f"  [seed] projet {pid} clos : {case['titre']}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-db", action="store_true",
                        help="pre-charge li90.db avec 3 cas RCA valides")
    args = parser.parse_args()

    df = gen_dataset()
    df.to_excel(OUT_XLSX, index=False, sheet_name="mesures")
    df_fix = gen_dataset(recovery=True)
    df_fix.to_excel(OUT_XLSX_FIX, index=False, sheet_name="mesures")

    # Resume de l'histoire implantee (verification rapide)
    first = df[df["timestamp"] < START + timedelta(days=10)]
    last = df[df["timestamp"] >= START + timedelta(days=23)]
    ppm_a = first["defauts_brasage"].sum() / first["cartes_produites"].sum() * 1e6
    ppm_b = last["defauts_brasage"].sum() / last["cartes_produites"].sum() * 1e6
    corr = df["defauts_brasage"].corr(df["temp_zone6_c"])

    print(f"OK : {OUT_XLSX.name} genere ({len(df)} lignes, "
          f"{len(df.columns)} colonnes)")
    print(f"  NC AOI 10 premiers jours : {ppm_a:,.0f} ppm")
    print(f"  NC AOI 7 derniers jours  : {ppm_b:,.0f} ppm  (x"
          f"{ppm_b / max(ppm_a, 1):.1f})")
    print(f"  Correlation defauts <-> temp_zone6_c : r = {corr:.2f}")
    print("  Cause implantee : derive four zone 6 apres maintenance du "
          "29 juin (connecteur element chauffant).")

    rec = df_fix[df_fix["timestamp"] >= START + timedelta(days=FIX_DAY + 2)]
    ppm_r = (rec["defauts_brasage"].sum()
             / rec["cartes_produites"].sum() * 1e6)
    print(f"OK : {OUT_XLSX_FIX.name} genere ({len(df_fix)} lignes)")
    print(f"  NC AOI apres correctif du 17/07 : {ppm_r:,.0f} ppm "
          "(preuve avant/apres pour la fin de la video)")

    if args.seed_db:
        print("Seed de la base CBR...")
        seed_db()
        print("OK : 3 cas historiques valides + clos dans li90.db")


if __name__ == "__main__":
    main()
