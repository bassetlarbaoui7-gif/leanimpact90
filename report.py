"""
Export PDF du rapport d'analyse.
MVP : un tableau simple, pas de graphiques embarques pour l'instant.
Section "Analyse des pertes" ajoutee si losses_results fourni.
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}


class Report(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "LI90 - Rapport d'analyse de derives", **NL, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, datetime.now().strftime("%d/%m/%Y %H:%M"), **NL, align="C")
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _append_losses_section(pdf: Report, losses_results: dict) -> None:
    """Ajoute la section Analyse des pertes au PDF en place."""
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Analyse des pertes", **NL)
    pdf.ln(2)

    # PPM global
    ppm = losses_results.get("ppm")
    pdf.set_font("Helvetica", "", 10)
    if ppm is not None and math.isfinite(ppm):
        pdf.cell(0, 6, f"PPM global : {ppm:.0f}", **NL)
    else:
        pdf.cell(0, 6, "PPM global : non calcule (volume non renseigne)", **NL)
    pdf.ln(2)

    # Tableau des correlations
    corr = losses_results.get("correlations")
    if corr is not None and len(corr) > 0:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Correlations parametre / defaut", **NL)

        headers = ["Parametre", "N", "Test", "r", "p-value"]
        widths = [50, 15, 25, 25, 25]
        pdf.set_font("Helvetica", "B", 9)
        for h, w in zip(headers, widths, strict=True):
            pdf.cell(w, 7, h, border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for _, row in corr.iterrows():
            r_val = row["r"]
            p_val = row["p_value"]
            r_nan = isinstance(r_val, float) and math.isnan(r_val)
            p_nan = isinstance(p_val, float) and math.isnan(p_val)
            cells = [
                str(row["param"])[:28],
                str(int(row["n"])),
                str(row["test_retenu"]),
                "-" if r_val is None or r_nan else f"{r_val:.3f}",
                "-" if p_val is None or p_nan else f"{p_val:.4f}",
            ]
            for c, w in zip(cells, widths, strict=True):
                pdf.cell(w, 6, c, border=1, align="C")
            pdf.ln()

    # Tableau Cpk si fourni
    cpk = losses_results.get("cpk")
    if cpk is not None and len(cpk) > 0:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Capabilite Cpk", **NL)

        headers = ["Parametre", "LSL", "USL", "Cpk", "Verdict"]
        widths = [45, 22, 22, 22, 55]
        pdf.set_font("Helvetica", "B", 9)
        for h, w in zip(headers, widths, strict=True):
            pdf.cell(w, 7, h, border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for _, row in cpk.iterrows():
            cpk_val = row.get("cpk")
            if cpk_val is None:
                cpk_str = "-"
            elif isinstance(cpk_val, float) and math.isinf(cpk_val):
                cpk_str = "inf"
            elif isinstance(cpk_val, float) and math.isnan(cpk_val):
                cpk_str = "-"
            else:
                cpk_str = f"{cpk_val:.3f}"
            cells = [
                str(row["param"])[:25],
                f"{row['lsl']:.2f}",
                f"{row['usl']:.2f}",
                cpk_str,
                str(row["verdict"])[:30],
            ]
            for c, w in zip(cells, widths, strict=True):
                pdf.cell(w, 6, c, border=1, align="C")
            pdf.ln()


def _append_shifts_section(pdf: Report, shifts_results: dict) -> None:
    """Ajoute la section inter-shift au PDF."""
    if not shifts_results.get("detected"):
        return
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Analyse inter-shift (Kruskal-Wallis)", **NL)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    shift_col = shifts_results.get("shift_column", "?")
    n_shifts = shifts_results.get("n_shifts", 0)
    pdf.cell(
        0, 6,
        f"Colonne detectee : {shift_col}  -  {n_shifts} shift(s) compares",
        **NL,
    )
    pdf.ln(2)

    comp = shifts_results.get("comparison")
    if comp is None or len(comp) == 0:
        return

    headers = ["Parametre", "Test", "Statistique", "p-value", "Significatif"]
    widths = [45, 30, 30, 30, 30]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths, strict=True):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for _, row in comp.iterrows():
        p_val = row.get("p_value")
        p_nan = isinstance(p_val, float) and math.isnan(p_val)
        stat = row.get("statistic", row.get("H", None))
        stat_str = "-" if stat is None or (
            isinstance(stat, float) and math.isnan(stat)
        ) else f"{stat:.2f}"
        p_str = "-" if p_val is None or p_nan else f"{p_val:.4f}"
        signif = "OUI" if (p_val is not None and not p_nan and p_val < 0.05) \
            else "non"
        cells = [
            str(row.get("param", row.get("parameter", "")))[:25],
            str(row.get("test", "Kruskal-Wallis"))[:18],
            stat_str,
            p_str,
            signif,
        ]
        for c, w in zip(cells, widths, strict=True):
            pdf.cell(w, 6, c, border=1, align="C")
        pdf.ln()


def generate_pdf(results: pd.DataFrame, output_path: str,
                 source_file: str = "",
                 losses_results: dict | None = None,
                 shifts_results: dict | None = None) -> None:
    pdf = Report()
    pdf.add_page()

    # Metadonnees
    pdf.set_font("Helvetica", "", 10)
    if source_file:
        pdf.cell(0, 6, f"Fichier source : {source_file}", **NL)
    pdf.cell(0, 6, f"Parametres analyses : {len(results)}", **NL)
    pdf.ln(4)

    # Section Shewhart
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detection de derives (Shewhart 3-sigma)", **NL)
    pdf.ln(2)

    headers = ["Parametre", "N", "Moyenne", "Ecart-type",
               "LCL", "UCL", "Viol.", "Criticite %"]
    widths = [40, 15, 25, 25, 25, 25, 15, 22]

    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths, strict=True):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for _, row in results.iterrows():
        cells = [
            str(row["parameter"])[:22],
            str(int(row["n"])),
            f"{row['mean']:.3f}",
            f"{row['std']:.3f}",
            f"{row['lcl']:.3f}",
            f"{row['ucl']:.3f}",
            str(int(row["violations"])),
            f"{row['criticality']:.2f}",
        ]
        for c, w in zip(cells, widths, strict=True):
            pdf.cell(w, 6, c, border=1, align="C")
        pdf.ln()

    # Section pertes (optionnelle)
    if losses_results is not None:
        _append_losses_section(pdf, losses_results)

    # Section shifts (optionnelle)
    if shifts_results is not None:
        _append_shifts_section(pdf, shifts_results)

    pdf.output(output_path)
