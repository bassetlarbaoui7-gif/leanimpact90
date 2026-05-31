"""
core/cbr/ishikawa_visual.py - Generateur du diagramme Ishikawa (arete de poisson).

Produit un SVG parametrique moderne (dark + orange) qui montre :
  - La tete (le probleme) a droite
  - La colonne vertebrale orange
  - 5 branches 5M (3 en haut, 2 en bas)
  - Pour chaque branche : sa CAUSE RACINE + etoiles de confiance + etat

Objectif : vue MACRO qui reduit la charge mentale. Le detail des 5 Pourquoi
de chaque branche est affiche separement (expanders) dans la page F3.
"""
from __future__ import annotations

import html

# Palette (alignee sur ui_theme)
C_BG        = "#0a0a0f"
C_SPINE     = "#f97316"
C_BRANCH    = "#fb923c"
C_TEXT      = "#f4f4f5"
C_MUTED     = "#a1a1aa"
C_OK        = "#22c55e"
C_DANGER    = "#ef4444"
C_CARD      = "#15151c"
C_BORDER    = "#27272f"

# Etats d'une cause racine
ETAT_PROPOSE = "propose"
ETAT_VALIDE  = "valide"
ETAT_REFUSE  = "refuse"

ETAT_COLORS = {
    ETAT_PROPOSE: C_SPINE,
    ETAT_VALIDE:  C_OK,
    ETAT_REFUSE:  C_DANGER,
}


def _stars(confidence: float) -> str:
    """Retourne une chaine d'etoiles pleines/vides selon la confiance 0..1."""
    n_full = max(0, min(5, round(confidence * 5)))
    return "★" * n_full + "☆" * (5 - n_full)


def _esc(text: str, max_len: int = 42) -> str:
    """Echappe le texte pour SVG et tronque."""
    if text is None:
        text = ""
    t = str(text)
    if len(t) > max_len:
        t = t[: max_len - 1] + "…"
    return html.escape(t)


def _wrap_two_lines(text: str, max_per_line: int = 24) -> tuple[str, str]:
    """Coupe un texte en max 2 lignes pour tenir dans un cartouche."""
    if text is None:
        text = ""
    t = str(text).strip()
    if len(t) <= max_per_line:
        return _esc(t, max_per_line), ""
    # Coupe au dernier espace avant la limite
    cut = t.rfind(" ", 0, max_per_line)
    if cut == -1:
        cut = max_per_line
    line1 = t[:cut].strip()
    line2 = t[cut:].strip()
    return _esc(line1, max_per_line), _esc(line2, max_per_line)


# ---------------------------------------------------------------------------
# Layout des 5 branches sur la spine
# ---------------------------------------------------------------------------
# Geometrie : spine horizontale a y=YC. 3 branches en haut, 2 en bas.
WIDTH   = 1120
HEIGHT  = 660
YC      = 330              # y de la colonne vertebrale
X_START = 80               # debut spine (queue)
X_HEAD  = 880              # debut tete (probleme)

# (nom_affiche, x_base_sur_spine, direction)  direction = -1 haut / +1 bas
BRANCH_LAYOUT = [
    ("Methode",       250, -1),
    ("Matiere",       460, -1),
    ("Main-d'oeuvre", 670, -1),
    ("Materiel",      355, +1),
    ("Milieu",        575, +1),
]


def render_ishikawa_svg(
    tree: dict,
    probleme_titre: str,
    *,
    etats: dict[str, str] | None = None,
) -> str:
    """
    Genere le SVG du diagramme Ishikawa.

    Args:
        tree   : dict {branche_m: [noeuds]} (sortie de path_engine)
        probleme_titre : titre du probleme (affiche dans la tete)
        etats  : dict {branche_m: ETAT_*} pour colorer les causes racines

    Returns:
        chaine SVG complete (a injecter via st.markdown unsafe_allow_html).
    """
    etats = etats or {}
    parts: list[str] = []

    parts.append(
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%; height:auto; background:{C_BG}; '
        f'border-radius:16px; border:1px solid {C_BORDER};">'
    )

    # --- Defs : glow filter ---
    parts.append(
        '<defs>'
        '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="3" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        '</defs>'
    )

    # --- Colonne vertebrale ---
    parts.append(
        f'<line x1="{X_START}" y1="{YC}" x2="{X_HEAD}" y2="{YC}" '
        f'stroke="{C_SPINE}" stroke-width="4" filter="url(#glow)"/>'
    )
    # Fleche vers la tete
    parts.append(
        f'<polygon points="{X_HEAD},{YC-9} {X_HEAD},{YC+9} {X_HEAD+16},{YC}" '
        f'fill="{C_SPINE}"/>'
    )

    # --- Tete : le probleme ---
    head_w, head_h = 200, 96
    hx, hy = X_HEAD + 20, YC - head_h // 2
    p1, p2 = _wrap_two_lines(probleme_titre, 22)
    parts.append(
        f'<rect x="{hx}" y="{hy}" width="{head_w}" height="{head_h}" rx="14" '
        f'fill="{C_SPINE}" filter="url(#glow)"/>'
    )
    parts.append(
        f'<text x="{hx + head_w//2}" y="{YC - 16}" text-anchor="middle" '
        f'fill="white" font-size="12" font-weight="700" '
        f'font-family="Inter,sans-serif">PROBLEME</text>'
    )
    parts.append(
        f'<text x="{hx + head_w//2}" y="{YC + 4}" text-anchor="middle" '
        f'fill="white" font-size="13" font-weight="600" '
        f'font-family="Inter,sans-serif">{p1}</text>'
    )
    if p2:
        parts.append(
            f'<text x="{hx + head_w//2}" y="{YC + 22}" text-anchor="middle" '
            f'fill="white" font-size="13" font-weight="600" '
            f'font-family="Inter,sans-serif">{p2}</text>'
        )

    # --- Branches ---
    for nom, x_base, direction in BRANCH_LAYOUT:
        noeuds = tree.get(nom, [])
        # Cause racine = dernier noeud / celui marque est_cause_racine
        racine = None
        for n in noeuds:
            if n.get("est_cause_racine"):
                racine = n
        if racine is None and noeuds:
            racine = noeuds[-1]

        tip_x = x_base - 60
        tip_y = YC + direction * 220
        # Ligne diagonale spine -> tip
        parts.append(
            f'<line x1="{x_base}" y1="{YC}" x2="{tip_x}" y2="{tip_y}" '
            f'stroke="{C_BRANCH}" stroke-width="2.5"/>'
        )
        # Label de la branche (au bout)
        label_y = tip_y + (direction * 6) + (-6 if direction < 0 else 18)
        parts.append(
            f'<text x="{tip_x}" y="{label_y}" text-anchor="middle" '
            f'fill="{C_BRANCH}" font-size="15" font-weight="800" '
            f'font-family="Inter,sans-serif">{_esc(nom, 16)}</text>'
        )

        # Cartouche cause racine au milieu de la branche
        if racine is not None:
            etat = etats.get(nom, ETAT_PROPOSE)
            col  = ETAT_COLORS.get(etat, C_SPINE)
            conf = float(racine.get("confidence", 0.5))
            mid_x = (x_base + tip_x) // 2 - 70
            mid_y = (YC + tip_y) // 2 + (direction * 4)
            box_w, box_h = 200, 64
            bx = mid_x - box_w + 60
            by = mid_y - box_h // 2

            l1, l2 = _wrap_two_lines(racine.get("reponse", ""), 26)
            parts.append(
                f'<rect x="{bx}" y="{by}" width="{box_w}" height="{box_h}" rx="10" '
                f'fill="{C_CARD}" stroke="{col}" stroke-width="1.8"/>'
            )
            # Petit point colore d'etat
            parts.append(
                f'<circle cx="{bx + 14}" cy="{by + 16}" r="5" fill="{col}"/>'
            )
            parts.append(
                f'<text x="{bx + 28}" y="{by + 20}" fill="{col}" '
                f'font-size="10" font-weight="700" '
                f'font-family="Inter,sans-serif">CAUSE RACINE</text>'
            )
            parts.append(
                f'<text x="{bx + 14}" y="{by + 38}" fill="{C_TEXT}" '
                f'font-size="11.5" font-weight="600" '
                f'font-family="Inter,sans-serif">{l1}</text>'
            )
            if l2:
                parts.append(
                    f'<text x="{bx + 14}" y="{by + 53}" fill="{C_TEXT}" '
                    f'font-size="11.5" font-weight="600" '
                    f'font-family="Inter,sans-serif">{l2}</text>'
                )
            # Etoiles de confiance
            parts.append(
                f'<text x="{bx + box_w - 8}" y="{by + 20}" text-anchor="end" '
                f'fill="{col}" font-size="12" '
                f'font-family="sans-serif">{_stars(conf)}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)
