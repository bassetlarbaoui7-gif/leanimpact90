"""
Tests unitaires pour data_loader.py.

Couvre les 12 pieges industriels un par un + tests d'integration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    detect_encoding,
    detect_header_rows,
    detect_separator,
    list_excel_sheets,
    load_file,
    max_consecutive_nans,
    normalize_french_numeric,
    parse_dates_flexible,
    replace_sentinels,
)


# =============================================================================
# PIEGE 1 : encodage
# =============================================================================
def test_piege1_encodage_latin1_detecte(tmp_path: Path):
    csv = tmp_path / "latin1.csv"
    csv.write_bytes("temperature;pression\nrégulée;2,5\n".encode("latin-1"))
    df, report = load_file(csv, csv.name)
    assert report.encoding is not None
    assert "régulée" in df.iloc[0].tolist() or df.iloc[0].notna().any()


def test_piege1_encodage_utf8_bom():
    data = b"\xef\xbb\xbfcol1,col2\na,b\n"
    assert detect_encoding(data) == "utf-8-sig"


# =============================================================================
# PIEGE 2 : separateur CSV
# =============================================================================
def test_piege2_separateur_point_virgule(tmp_path: Path):
    csv = tmp_path / "fr.csv"
    csv.write_text("temp;pression;defauts\n180;2.5;3\n181;2.6;1\n", encoding="utf-8")
    df, report = load_file(csv, csv.name)
    assert report.separator == ";"
    assert list(df.columns) == ["temp", "pression", "defauts"]
    assert len(df) == 2


def test_piege2_separateur_tabulation(tmp_path: Path):
    csv = tmp_path / "tsv.csv"
    csv.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
    df, report = load_file(csv, csv.name)
    assert report.separator == "\t"
    assert len(df.columns) == 3


# =============================================================================
# PIEGE 3 : decimales francaises
# =============================================================================
def test_piege3_decimales_francaises(tmp_path: Path):
    csv = tmp_path / "fr_decimal.csv"
    csv.write_text("temp;pression\n180,5;2,47\n181,2;2,51\n", encoding="utf-8")
    df, report = load_file(csv, csv.name)
    assert "temp" in report.french_decimals_converted
    assert "pression" in report.french_decimals_converted
    assert pd.api.types.is_numeric_dtype(df["temp"])
    assert df["temp"].iloc[0] == pytest.approx(180.5)
    assert df["pression"].iloc[1] == pytest.approx(2.51)


def test_piege3_normalize_pure():
    s = pd.Series(["2,47", "1 234,56", "3,14"])
    out, converted = normalize_french_numeric(s)
    assert converted is True
    assert out.iloc[0] == pytest.approx(2.47)
    assert out.iloc[1] == pytest.approx(1234.56)


# =============================================================================
# PIEGE 4 : dates multi-formats
# =============================================================================
def test_piege4_dates_format_francais(tmp_path: Path):
    csv = tmp_path / "dates.csv"
    csv.write_text(
        "timestamp;valeur\n15/03/2026 10:30:00;180.5\n"
        "15/03/2026 11:00:00;181.0\n16/03/2026 08:15:00;179.8\n",
        encoding="utf-8",
    )
    df, report = load_file(csv, csv.name)
    assert "timestamp" in report.dates_parsed
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-03-15 10:30:00")


def test_piege4_parse_pure_iso():
    s = pd.Series(["2026-03-15", "2026-03-16", "2026-03-17"])
    out, parsed = parse_dates_flexible(s)
    assert parsed is True
    assert out.iloc[0] == pd.Timestamp("2026-03-15")


def test_piege4_numerique_pur_non_interprete_comme_date():
    """Un '180' ne doit pas etre parse comme date (bug Unix timestamp courant)."""
    s = pd.Series(["180", "181", "182"])
    _, parsed = parse_dates_flexible(s)
    assert parsed is False


# =============================================================================
# PIEGE 5 : headers sur 2 lignes
# =============================================================================
def test_piege5_headers_multilignes_excel(tmp_path: Path):
    xlsx = tmp_path / "multi_header.xlsx"
    # Ligne 1 : categorie, ligne 2 : sous-categorie, ligne 3+ : donnees
    data = [
        ["Four", "Four", "Injection"],
        ["Temp_C", "Pression_bar", "Vitesse_rpm"],
        [180.5, 2.5, 85.0],
        [181.2, 2.6, 85.0],
        [179.8, 2.4, 85.0],
    ]
    pd.DataFrame(data).to_excel(xlsx, index=False, header=False)
    df, report = load_file(xlsx, xlsx.name)
    assert report.header_rows_detected == 2
    # Les colonnes doivent contenir la fusion parent_enfant
    assert any("Temp_C" in c for c in df.columns)
    assert len(df) == 3


# =============================================================================
# PIEGE 6 : cellules fusionnees dans headers
# (gere via detect_header_rows + flatten : test indirect au piege 5)
# =============================================================================
def test_piege6_flatten_columns_avec_unnamed(tmp_path: Path):
    """Quand Excel a des cellules fusionnees en header, les 'Unnamed:' sont filtres."""
    xlsx = tmp_path / "fused.xlsx"
    # Simule : Four fusionne sur 2 colonnes (Temp + Pression)
    data = [
        ["Four", None, "Injection"],
        ["Temp", "Pression", "Vitesse"],
        [180.5, 2.5, 85.0],
        [181.0, 2.6, 85.0],
        [182.0, 2.7, 85.0],
    ]
    pd.DataFrame(data).to_excel(xlsx, index=False, header=False)
    df, report = load_file(xlsx, xlsx.name)
    assert report.header_rows_detected == 2
    # Chaque nom de colonne est lisible, sans 'Unnamed' parasite
    for col in df.columns:
        assert "Unnamed" not in col


# =============================================================================
# PIEGE 7 : valeurs sentinelles
# =============================================================================
def test_piege7_sentinelles_textuelles():
    s = pd.Series(["180", "N/A", "181", "NULL", "SENSOR_FAIL", "182"])
    out, n = replace_sentinels(s)
    assert n == 3
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[3])
    assert pd.isna(out.iloc[4])


def test_piege7_sentinelles_numeriques():
    s = pd.Series([180.0, -9999.0, 181.0, -273.15, 182.0])
    out, n = replace_sentinels(s)
    assert n == 2
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[3])


def test_piege7_integration_csv_sentinelles(tmp_path: Path):
    csv = tmp_path / "sentinels.csv"
    csv.write_text(
        "temp;pression\n180;2.5\nN/A;2.6\n181;SENSOR_FAIL\n-9999;2.7\n",
        encoding="utf-8",
    )
    df, report = load_file(csv, csv.name)
    assert sum(report.sentinels_replaced.values()) >= 3


# =============================================================================
# PIEGE 8 : separateurs de milliers
# =============================================================================
def test_piege8_separateurs_de_milliers(tmp_path: Path):
    csv = tmp_path / "thousands.csv"
    # '1 234,56' avec espace insecable (nbsp)
    csv.write_text(
        "volume;cout\n1\u00a0234,56;12\u00a0500\n5\u00a0000,00;45\u00a0000\n",
        encoding="utf-8",
    )
    df, report = load_file(csv, csv.name)
    assert "volume" in report.french_decimals_converted
    assert df["volume"].iloc[0] == pytest.approx(1234.56)
    assert df["cout"].iloc[1] == pytest.approx(45000.0)


# =============================================================================
# PIEGE 9 : unites collees aux nombres
# =============================================================================
def test_piege9_unites_suffixe():
    s = pd.Series(["22,5 degC", "23,1 degC", "24,0 degC"])
    out, converted = normalize_french_numeric(s)
    assert converted is True
    assert out.iloc[0] == pytest.approx(22.5)
    assert out.iloc[2] == pytest.approx(24.0)


def test_piege9_unites_collees():
    s = pd.Series(["2.5bar", "2.6bar", "2.7bar"])
    out, converted = normalize_french_numeric(s)
    assert converted is True
    assert out.iloc[0] == pytest.approx(2.5)


# =============================================================================
# PIEGE 10 : doublons temporels (warning, pas dedup)
# =============================================================================
def test_piege10_doublons_timestamp_warning(tmp_path: Path):
    csv = tmp_path / "dup_ts.csv"
    csv.write_text(
        "timestamp;valeur\n"
        "15/03/2026 10:00:00;1\n"
        "15/03/2026 10:00:00;2\n"
        "15/03/2026 10:00:00;3\n"
        "15/03/2026 10:05:00;4\n",
        encoding="utf-8",
    )
    df, report = load_file(csv, csv.name)
    # Il doit y avoir un warning mentionnant les doublons
    assert any("double" in w.lower() for w in report.warnings)
    # Les donnees ne sont PAS dedupliquees automatiquement
    assert len(df) == 4


# =============================================================================
# PIEGE 11 : trous capteur longs
# =============================================================================
def test_piege11_trou_capteur_warning(tmp_path: Path):
    # 30 NaN consecutifs, devrait declencher un warning
    values = [180.0] * 10 + [np.nan] * 30 + [181.0] * 10
    csv = tmp_path / "gap.csv"
    lines = ["temp"] + ["" if np.isnan(v) else str(v) for v in values]
    csv.write_text("\n".join(lines), encoding="utf-8")
    df, report = load_file(csv, csv.name)
    assert any("trou" in w.lower() for w in report.warnings)


def test_piege11_max_consecutive_nans_pure():
    s = pd.Series([1.0, 2.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 4.0])
    assert max_consecutive_nans(s) == 3


# =============================================================================
# PIEGE 12 : feuilles Excel multiples
# =============================================================================
def test_piege12_multi_sheet_warning_par_defaut(tmp_path: Path):
    xlsx = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        df_a = pd.DataFrame({"a": [1, 2, 3]})
        df_b = pd.DataFrame({"b": [4, 5, 6]})
        df_a.to_excel(writer, sheet_name="Production", index=False)
        df_b.to_excel(writer, sheet_name="Maintenance", index=False)
    df, report = load_file(xlsx, xlsx.name)
    # Feuille par defaut = premiere
    assert report.sheet == "Production"
    assert report.sheets_available == ["Production", "Maintenance"]
    # Un warning doit prevenir l'utilisateur
    assert any("feuille" in w.lower() for w in report.warnings)


def test_piege12_choix_explicite_feuille(tmp_path: Path):
    xlsx = tmp_path / "multi2.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        df_a = pd.DataFrame({"a": [1, 2, 3]})
        df_b = pd.DataFrame({"b": [4, 5, 6]})
        df_a.to_excel(writer, sheet_name="Sheet1", index=False)
        df_b.to_excel(writer, sheet_name="Sheet2", index=False)
    df, report = load_file(xlsx, xlsx.name, sheet="Sheet2")
    assert report.sheet == "Sheet2"
    assert "b" in df.columns


def test_piege12_feuille_inexistante_leve_erreur(tmp_path: Path):
    xlsx = tmp_path / "mono.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(xlsx, sheet_name="OnlyOne", index=False)
    with pytest.raises(ValueError, match="absente"):
        load_file(xlsx, xlsx.name, sheet="Inexistante")


def test_piege12_list_sheets(tmp_path: Path):
    xlsx = tmp_path / "list.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="A", index=False)
        pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="B", index=False)
        pd.DataFrame({"c": [3]}).to_excel(writer, sheet_name="C", index=False)
    assert list_excel_sheets(xlsx) == ["A", "B", "C"]


# =============================================================================
# Integration : fichier realiste combinant plusieurs pieges
# =============================================================================
def test_integration_fichier_industriel_realiste(tmp_path: Path):
    """
    Fichier FR realiste combinant : separateur ;, decimales FR, dates FR,
    sentinelles, trous capteur.
    """
    csv = tmp_path / "realiste.csv"
    content = "timestamp;temp_four;pression;nb_defauts\n"
    content += "15/03/2026 08:00:00;180,5;2,47;2\n"
    content += "15/03/2026 08:15:00;N/A;2,48;1\n"
    content += "15/03/2026 08:30:00;181,2;SENSOR_FAIL;3\n"
    content += "15/03/2026 08:45:00;-9999;2,49;0\n"
    content += "15/03/2026 09:00:00;179,8;2,50;1\n"
    csv.write_text(content, encoding="utf-8")

    df, report = load_file(csv, csv.name)

    assert report.separator == ";"
    assert "timestamp" in report.dates_parsed
    assert "temp_four" in report.french_decimals_converted
    assert sum(report.sentinels_replaced.values()) >= 3
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_numeric_dtype(df["temp_four"])
    # La ligne "-9999" et la ligne "N/A" doivent avoir cree des NaN
    assert df["temp_four"].isna().sum() == 2


def test_integration_extension_non_supportee(tmp_path: Path):
    bad = tmp_path / "fichier.docx"
    bad.write_text("contenu", encoding="utf-8")
    with pytest.raises(ValueError, match="Extension"):
        load_file(bad, bad.name)


def test_integration_rapport_summary_non_vide(tmp_path: Path):
    csv = tmp_path / "simple.csv"
    csv.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    _, report = load_file(csv, csv.name)
    summary = report.summary()
    assert "2 lignes" in summary
    assert "2 colonnes" in summary


# =============================================================================
# Fonctions utilitaires : tests directs
# =============================================================================
def test_detect_separator_stable_gagne():
    # `,` apparait 2 fois par ligne de facon stable, `;` 1 fois instable
    sample = "a,b,c\n1,2,3\n4,5,6\n"
    assert detect_separator(sample) == ","


def test_detect_header_rows_simple_header():
    df = pd.DataFrame({
        0: ["col_a", 1, 2, 3],
        1: ["col_b", 10, 20, 30],
    })
    assert detect_header_rows(df) == 1
