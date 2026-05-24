"""
Chargement robuste de fichiers Excel/CSV industriels.

Gere explicitement les 12 pieges courants en production :

  1. Encodage (latin-1, windows-1252, utf-8-bom)
  2. Separateur CSV (;, ,, tab, pipe)
  3. Decimales francaises (',' au lieu de '.')
  4. Dates multi-formats (dd/mm/yyyy, yyyy-mm-dd, etc.)
  5. Headers sur 2 lignes (Excel industriel)
  6. Cellules fusionnees dans les en-tetes
  7. Valeurs sentinelles (-9999, NULL, #N/A, SENSOR_FAIL, -273.15, etc.)
  8. Separateurs de milliers ('1 234,56' avec espace insecable)
  9. Unites collees aux nombres ('22,5 degC', '2.5bar')
 10. Doublons temporels (warning, pas de dedup automatique)
 11. Trous capteur longs (warning si >= 20 NaN consecutifs)
 12. Feuilles Excel multiples (liste exposee, choix explicite)

Retourne (DataFrame, LoadReport) : le rapport documente chaque transformation
pour tracabilite (important pour l'audit qualite industriel).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chardet
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

# Valeurs textuelles couramment utilisees comme "donnee manquante"
SENTINEL_TEXTS = {
    "", "-", "--", "---", "N/A", "#N/A", "NA", "NULL", "null", "None",
    "ERR", "ERROR", "SENSOR_FAIL", "OVER", "UNDER",
    "#VALUE!", "#DIV/0!", "#REF!", "#NAME?",
}

# Valeurs numeriques sentinelles (codes d'erreur capteur classiques)
SENTINEL_NUMBERS = {-9999.0, -999.0, -273.15, 9999.0, 65535.0}

# Formats de dates essayes dans l'ordre (du plus specifique au plus general)
DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%Y/%m/%d",
    "%d.%m.%Y",
]

CSV_SEPARATORS = [";", ",", "\t", "|"]

GAP_ALERT_THRESHOLD = 20  # NaN consecutifs declencheurs d'avertissement
DATE_HEURISTIC_CHARS = re.compile(r"[-/:]")


# -----------------------------------------------------------------------------
# Rapport de chargement
# -----------------------------------------------------------------------------

@dataclass
class LoadReport:
    """Rapport de tracabilite du chargement (audit qualite)."""
    encoding: str | None = None
    separator: str | None = None
    sheet: str | None = None
    sheets_available: list[str] = field(default_factory=list)
    header_rows_detected: int = 1
    n_rows: int = 0
    n_cols: int = 0
    sentinels_replaced: dict[str, int] = field(default_factory=dict)
    french_decimals_converted: list[str] = field(default_factory=list)
    dates_parsed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Rapport texte concis pour affichage UI."""
        lines = []
        if self.encoding:
            lines.append(f"Encodage : {self.encoding}")
        if self.separator:
            sep_label = {"\t": "TAB", ";": ";", ",": ",", "|": "|"}.get(
                self.separator, self.separator)
            lines.append(f"Separateur : '{sep_label}'")
        if self.sheet:
            lines.append(f"Feuille : {self.sheet}")
        if self.header_rows_detected > 1:
            lines.append(f"Headers : {self.header_rows_detected} lignes fusionnees")
        if self.sentinels_replaced:
            total = sum(self.sentinels_replaced.values())
            lines.append(f"Sentinelles remplacees : {total} valeurs")
        if self.french_decimals_converted:
            n = len(self.french_decimals_converted)
            lines.append(f"Decimales FR converties : {n} colonne(s)")
        if self.dates_parsed:
            lines.append(f"Dates parsees : {len(self.dates_parsed)} colonne(s)")
        lines.append(f"Resultat : {self.n_rows} lignes x {self.n_cols} colonnes")
        return " | ".join(lines)


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def detect_encoding(data: bytes, default: str = "utf-8") -> str:
    """Detecte l'encodage d'un contenu binaire."""
    if not data:
        return default
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        result = chardet.detect(data[:10_000])
        enc = (result.get("encoding") or default).lower()
        # chardet retourne parfois 'ascii' pour du UTF-8 sans accents
        if enc == "ascii":
            enc = "utf-8"
        return enc
    except Exception:
        return default


def detect_separator(sample: str) -> str:
    """Detecte le separateur CSV en analysant la stabilite par ligne."""
    lines = [ln for ln in sample.split("\n") if ln.strip()][:15]
    if not lines:
        return ","
    best_sep = ","
    best_score = -1
    for sep in CSV_SEPARATORS:
        counts = [ln.count(sep) for ln in lines]
        if max(counts) == 0:
            continue
        # Un bon separateur produit le meme nombre de colonnes a chaque ligne
        stable = len(set(counts)) == 1
        score = (max(counts) * 10) if stable else max(counts)
        if score > best_score:
            best_score = score
            best_sep = sep
    return best_sep


def _row_numeric_ratio(row: pd.Series) -> float:
    """Fraction d'une ligne qui ressemble a du numerique."""
    if len(row) == 0:
        return 0.0
    n_numeric = 0
    for v in row:
        if pd.isna(v):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            n_numeric += 1
        elif isinstance(v, str):
            # heuristique : contient au moins un chiffre, pas plus de 2 lettres
            s = v.strip()
            n_digits = sum(c.isdigit() for c in s)
            n_letters = sum(c.isalpha() for c in s)
            if n_digits > 0 and n_letters <= 2:
                n_numeric += 1
    return n_numeric / len(row)


def detect_header_rows(df_raw: pd.DataFrame) -> int:
    """
    Retourne 1 ou 2 selon la probabilite de header multi-lignes.
    Heuristique : 2 premieres lignes textuelles + 3eme numerique => 2 headers.
    """
    if len(df_raw) < 3 or len(df_raw.columns) == 0:
        return 1
    r0 = _row_numeric_ratio(df_raw.iloc[0])
    r1 = _row_numeric_ratio(df_raw.iloc[1])
    r2 = _row_numeric_ratio(df_raw.iloc[2])
    if r0 < 0.3 and r1 < 0.3 and r2 >= 0.5:
        return 2
    return 1


# -----------------------------------------------------------------------------
# Nettoyage colonne par colonne
# -----------------------------------------------------------------------------

def replace_sentinels(series: pd.Series,
                      extra: list[str] | None = None
                      ) -> tuple[pd.Series, int]:
    """Remplace les valeurs sentinelles par NaN. Retourne (serie, n_remplacees)."""
    if series.empty:
        return series, 0

    # Sentinelles numeriques (si deja typee numerique)
    if series.dtype.kind in ("i", "f"):
        mask = series.isin(SENTINEL_NUMBERS)
        n = int(mask.sum())
        return series.where(~mask, np.nan), n

    # Sentinelles textuelles (colonnes object)
    sentinels = set(SENTINEL_TEXTS)
    if extra:
        sentinels.update(extra)
    sentinels_lower = {s.lower() for s in sentinels}

    # Ajoute les formes numeriques sentinelles sous forme texte
    for num in SENTINEL_NUMBERS:
        sentinels_lower.add(str(num))
        sentinels_lower.add(str(int(num)) if num == int(num) else str(num))

    def is_sentinel(v: Any) -> bool:
        if pd.isna(v):
            return False
        return str(v).strip().lower() in sentinels_lower

    mask = series.apply(is_sentinel)
    n = int(mask.sum())
    return series.where(~mask, np.nan), n


def _looks_like_date(series: pd.Series) -> bool:
    """Heuristique rapide : majorite des valeurs contiennent - / ou :."""
    sample = series.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    hits = sum(1 for v in sample if DATE_HEURISTIC_CHARS.search(v))
    return hits / len(sample) >= 0.5


def parse_dates_flexible(series: pd.Series) -> tuple[pd.Series, bool]:
    """Essaie plusieurs formats de dates. Retourne (serie, True_si_parse)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, False
    if series.dtype.kind in ("i", "f"):
        return series, False
    if not _looks_like_date(series):
        return series, False

    n_non_null = int(series.notna().sum())
    if n_non_null == 0:
        return series, False

    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().sum() / n_non_null >= 0.8:
            return parsed, True

    # Dernier recours : parser permissif avec dayfirst (convention FR)
    try:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed.notna().sum() / n_non_null >= 0.8:
            return parsed, True
    except Exception:
        pass

    return series, False


def normalize_french_numeric(series: pd.Series) -> tuple[pd.Series, bool]:
    """
    Convertit '2,47' -> 2.47 ; '1 234,56' -> 1234.56 ; '22,5 degC' -> 22.5.
    Retourne (serie, True_si_conversion_utile).
    """
    if series.dtype.kind in ("i", "f"):
        return series, False
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, False

    def clean(v: Any) -> Any:
        if pd.isna(v):
            return v
        s = str(v).strip()
        # Suppression d'espaces (separateurs de milliers) + espaces insecables
        s = s.replace("\u00a0", "").replace(" ", "")
        # Suppression des suffixes d'unites (lettres, %, °) en fin de chaine
        # Ne touche pas aux exposants scientifiques ("1e5" n'a pas de suffixe alpha)
        s = re.sub(r"[a-zA-Z°%]+$", "", s)
        # Virgule decimale -> point
        s = s.replace(",", ".")
        return s

    cleaned = series.apply(clean)
    numeric = pd.to_numeric(cleaned, errors="coerce")

    n_non_null = int(series.notna().sum())
    if n_non_null == 0:
        return series, False

    ratio = numeric.notna().sum() / n_non_null
    # Ne convertir que si la grande majorite passe (evite de casser des colonnes texte)
    if ratio >= 0.8:
        return numeric, True
    return series, False


# -----------------------------------------------------------------------------
# Detection des problemes post-chargement
# -----------------------------------------------------------------------------

def max_consecutive_nans(series: pd.Series) -> int:
    """Retourne la longueur de la plus longue sequence de NaN consecutifs."""
    is_nan = series.isna().to_numpy()
    if not is_nan.any():
        return 0
    max_gap = 0
    current = 0
    for v in is_nan:
        if v:
            current += 1
            if current > max_gap:
                max_gap = current
        else:
            current = 0
    return max_gap


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne des colonnes MultiIndex en 'parent_enfant' (lisible)."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat = []
    for col in df.columns:
        parts = [str(p).strip() for p in col
                 if str(p) != "nan" and not str(p).startswith("Unnamed")]
        flat.append("_".join(parts) if parts else "unnamed")
    df = df.copy()
    df.columns = flat
    return df


# -----------------------------------------------------------------------------
# Chargement principal
# -----------------------------------------------------------------------------

def list_excel_sheets(file_obj: Path | io.BytesIO | Any) -> list[str]:
    """Liste les feuilles d'un fichier Excel (pour UI de selection)."""
    xl = pd.ExcelFile(file_obj)
    return xl.sheet_names


def load_file(
    file_obj: Path | io.BytesIO | Any,
    filename: str,
    sheet: str | None = None,
    custom_sentinels: list[str] | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Charge un fichier Excel ou CSV avec detection automatique et nettoyage complet.

    Parameters
    ----------
    file_obj : chemin, BytesIO, ou UploadedFile Streamlit
    filename : nom du fichier (pour detection d'extension)
    sheet    : nom de la feuille Excel (None = 1ere feuille, avec warning si multi)
    custom_sentinels : valeurs textuelles supplementaires a traiter comme NaN

    Returns
    -------
    (df, report) : DataFrame nettoye + rapport de tracabilite
    """
    report = LoadReport()
    name_lower = filename.lower()

    if name_lower.endswith((".xlsx", ".xls")):
        df_raw = _load_excel(file_obj, sheet, report)
    elif name_lower.endswith((".csv", ".txt", ".tsv")):
        df_raw = _load_csv(file_obj, report)
    else:
        raise ValueError(
            f"Extension non supportee : {filename}. "
            f"Attendu : .xlsx, .xls, .csv, .txt, .tsv."
        )

    if df_raw.empty:
        report.n_rows = 0
        report.n_cols = 0
        return df_raw, report

    # Nettoyage colonne par colonne : sentinelles -> dates -> numerique FR
    df = df_raw.copy()
    for col in df.columns:
        s = df[col]

        s_clean, n_sent = replace_sentinels(s, extra=custom_sentinels)
        if n_sent > 0:
            report.sentinels_replaced[str(col)] = n_sent
        s = s_clean

        s_date, is_date = parse_dates_flexible(s)
        if is_date:
            report.dates_parsed.append(str(col))
            df[col] = s_date
            continue  # une date ne doit pas ensuite etre re-convertie en numerique

        s_num, converted = normalize_french_numeric(s)
        if converted:
            report.french_decimals_converted.append(str(col))
            s = s_num

        df[col] = s

    # Warnings post-nettoyage
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            n_dup = int(df[col].duplicated().sum())
            if n_dup > 0:
                report.warnings.append(
                    f"Colonne '{col}' : {n_dup} timestamp(s) en double.")

        if pd.api.types.is_numeric_dtype(df[col]):
            gap = max_consecutive_nans(df[col])
            if gap >= GAP_ALERT_THRESHOLD:
                report.warnings.append(
                    f"Colonne '{col}' : trou capteur detecte "
                    f"({gap} valeurs manquantes consecutives).")

    report.n_rows = len(df)
    report.n_cols = len(df.columns)
    return df, report


# -----------------------------------------------------------------------------
# Loaders specifiques (internes)
# -----------------------------------------------------------------------------

def _load_excel(file_obj: Any, sheet: str | None,
                report: LoadReport) -> pd.DataFrame:
    """Chargement Excel : multi-feuilles + headers multi-lignes."""
    xl = pd.ExcelFile(file_obj)
    report.sheets_available = list(xl.sheet_names)

    chosen = sheet or xl.sheet_names[0]
    if chosen not in xl.sheet_names:
        raise ValueError(
            f"Feuille '{sheet}' absente. Disponibles : {xl.sheet_names}"
        )
    report.sheet = chosen

    if len(xl.sheet_names) > 1 and sheet is None:
        report.warnings.append(
            f"Fichier contient {len(xl.sheet_names)} feuilles "
            f"({xl.sheet_names}). Feuille '{chosen}' chargee par defaut."
        )

    # Lecture sans header pour detecter le nombre de lignes d'en-tete
    df_probe = pd.read_excel(xl, sheet_name=chosen, header=None, nrows=5)
    if df_probe.empty:
        return pd.DataFrame()

    n_header = detect_header_rows(df_probe)
    report.header_rows_detected = n_header

    if n_header == 2:
        df = pd.read_excel(xl, sheet_name=chosen, header=[0, 1])
        df = _flatten_multiindex_columns(df)
    else:
        df = pd.read_excel(xl, sheet_name=chosen, header=0)

    # Supprimer les lignes entierement vides (frequent en Excel industriel)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _load_csv(file_obj: Any, report: LoadReport) -> pd.DataFrame:
    """Chargement CSV : detection d'encodage et de separateur."""
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    else:
        raw = Path(file_obj).read_bytes()

    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    encoding = detect_encoding(raw)
    report.encoding = encoding

    try:
        text = raw.decode(encoding, errors="replace")
    except LookupError:
        # Encodage detecte mais non supporte par Python : fallback latin-1
        text = raw.decode("latin-1", errors="replace")
        report.encoding = "latin-1"
        report.warnings.append(
            f"Encodage detecte '{encoding}' non supporte, fallback latin-1.")

    separator = detect_separator(text)
    report.separator = separator

    # keep_default_na=False : on gere les sentinelles nous-memes (tracabilite audit)
    # skip_blank_lines=False : on preserve les trous capteur (lignes vides)
    df = pd.read_csv(
        io.StringIO(text),
        sep=separator,
        dtype=object,
        keep_default_na=False,
        na_values=[],
        skip_blank_lines=False,
    )
    return df
