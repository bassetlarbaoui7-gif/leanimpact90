"""
F1 - Signaler / Importer / Saisir un probleme.

Trois portes d'entree dans le logiciel :
  1. Signalement operateur : l'operateur saisit ce qu'il voit (30 sec)
  2. Importer un fichier   : le responsable charge un Excel / CSV
  3. Saisie directe        : le responsable ecrit le probleme directement

A la fin de chaque mode, un bouton "Analyser la cause racine" amene
directement a F3 sans passer par les etapes intermediaires.
"""
from __future__ import annotations

import streamlit as st

from core import db
from data_loader import load_file, list_excel_sheets
from ui_theme import COLOR_TEXT_MUTED, COLOR_PRIMARY, COLOR_OK, section
from vue_b.synthese import render_synthese


SEVERITES = ["faible", "moyenne", "haute", "critique"]
TYPES_INCIDENT = [
    "Defaut produit", "Panne machine", "Derive parametre",
    "Probleme matiere", "Probleme methode", "Autre",
]


# ---------------------------------------------------------------------------
# KPI bar
# ---------------------------------------------------------------------------
def _kpi_bar() -> None:
    df_brut       = db.list_incidents(statut="brut")
    df_fiabilise  = db.list_incidents(statut="fiabilise")
    df_all        = db.list_incidents()
    n_brut, n_fia, n_all = len(df_brut), len(df_fiabilise), len(df_all)

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns: repeat(3, 1fr);
                    gap:12px; margin: 4px 0 18px 0;">
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">A traiter</div>
            <div style="font-size:24px; font-weight:700; color:{COLOR_PRIMARY};">{n_brut}</div>
          </div>
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Prets a analyser</div>
            <div style="font-size:24px; font-weight:700; color:{COLOR_OK};">{n_fia}</div>
          </div>
          <div style="padding:14px 18px; background:#111118;
                      border:1px solid #27272f; border-radius:12px;">
            <div style="font-size:12px; color:{COLOR_TEXT_MUTED};">Total</div>
            <div style="font-size:24px; font-weight:700;">{n_all}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# CTA - Bascule vers F3 Cause racine
# ---------------------------------------------------------------------------
def _set_nav_to_f3() -> None:
    """Callback : force la nav Vue B vers la page F3."""
    st.session_state["nav_b_page"] = "3. Cause racine IA"


def _render_cta_analyser() -> None:
    """Gros bouton qui amene a F3 (utilise en bas de chaque mode)."""
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.button(
            "Analyser la cause racine →",
            type="primary",
            use_container_width=True,
            key=f"f1_cta_{st.session_state.get('f1_mode', 'op')}",
            on_click=_set_nav_to_f3,
            help="Ouvre directement la page d'analyse de la vraie cause",
        )


# ---------------------------------------------------------------------------
# MODE 1 - Signalement operateur (form classique terrain)
# ---------------------------------------------------------------------------
def _mode_signalement(role: str) -> None:
    st.session_state["f1_mode"] = "op"
    with st.container(border=True):
        section("Signaler un probleme",
                "Ce que tu vois sur le terrain, en 30 secondes.")
        col1, col2 = st.columns(2)
        with col1:
            machine = st.text_input("Machine", placeholder="ex: M3 - Soudeuse",
                                     key="f1_op_machine")
            type_inc = st.selectbox("Type", TYPES_INCIDENT, key="f1_op_type")
            severite = st.select_slider(
                "Severite", options=SEVERITES, value="moyenne",
                key="f1_op_severite",
            )
        with col2:
            operateur = st.text_input(
                "Operateur (optionnel)",
                value=st.session_state.get("user_name", ""),
                key="f1_op_nom",
            )
            description = st.text_area(
                "Description",
                placeholder=("Ce que tu vois, ce que tu entends, depuis quand,"
                             " ce qui a change..."),
                height=120,
                key="f1_op_desc",
            )

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            submit = st.button(
                "Creer l'incident",
                type="primary",
                use_container_width=True,
                disabled=not machine.strip(),
                key="f1_op_submit",
            )
        with col_info:
            st.caption("L'incident sera transmis. Tu n'as pas besoin "
                       "d'en faire plus.")
        if submit:
            iid = db.create_incident(
                machine=machine,
                description=description,
                operateur_nom=operateur,
                type_incident=type_inc,
                severite=severite,
                cree_par_role=role,
            )
            st.success(f"Incident #{iid} cree.")
            st.rerun()


# ---------------------------------------------------------------------------
# MODE 2 - Importer un fichier (Excel / CSV)
# ---------------------------------------------------------------------------
def _mode_import(role: str) -> None:
    st.session_state["f1_mode"] = "import"
    with st.container(border=True):
        section("Importer un fichier de donnees",
                "Excel ou CSV. Les donnees seront analysees pour trouver "
                "la cause racine.")
        uploaded = st.file_uploader(
            "Glisse ton fichier ici",
            type=["xlsx", "xls", "csv"],
            key="f1_imp_file",
        )
        if uploaded is None:
            st.caption("Une fois charge, tu pourras passer directement a "
                       "l'analyse de la cause racine.")
            return

        # Charge le fichier
        selected_sheet = None
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = list_excel_sheets(uploaded)
                uploaded.seek(0)
                if len(sheets) > 1:
                    selected_sheet = st.selectbox(
                        "Feuille", options=sheets, index=0,
                        key="f1_imp_sheet",
                    )
            except Exception as e:
                st.warning(f"Impossible de lister les feuilles : {e}")

        try:
            df, _ = load_file(uploaded, uploaded.name, sheet=selected_sheet)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            return

        # Stock le df en session pour que F3 puisse l'utiliser
        st.session_state["df"] = df
        st.session_state["source_name"] = uploaded.name

        st.success(
            f"Fichier charge : {len(df)} lignes, {len(df.columns)} colonnes."
        )

        # Form rapide pour creer l'incident lie au fichier
        col1, col2 = st.columns(2)
        with col1:
            machine_imp = st.text_input(
                "Machine ou ligne concernee",
                placeholder="ex: Ligne 3 - Soudeuse M3",
                key="f1_imp_machine",
            )
            sev_imp = st.select_slider(
                "Severite", options=SEVERITES, value="moyenne",
                key="f1_imp_severite",
            )
        with col2:
            desc_imp = st.text_area(
                "Contexte (optionnel)",
                placeholder=("Que cherches-tu dans ces donnees ? "
                             "Quel probleme as-tu observe ?"),
                height=110,
                key="f1_imp_desc",
            )

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button(
                "Creer le probleme",
                type="primary",
                use_container_width=True,
                disabled=not machine_imp.strip(),
                key="f1_imp_submit",
            ):
                iid = db.create_incident(
                    machine=machine_imp,
                    description=(desc_imp or
                                 f"Donnees importees : {uploaded.name}"),
                    type_incident="Import de donnees",
                    severite=sev_imp,
                    cree_par_role=role,
                )
                # Le responsable a importe lui-meme : pas besoin de Tech N+1
                db.update_incident_statut(iid, "fiabilise")
                st.success(
                    f"Probleme #{iid} cree. Tu peux analyser la cause "
                    f"directement."
                )
                st.rerun()


# ---------------------------------------------------------------------------
# MODE 3 - Saisie directe responsable
# ---------------------------------------------------------------------------
def _mode_saisie_directe(role: str) -> None:
    st.session_state["f1_mode"] = "direct"
    with st.container(border=True):
        section("Decrire un probleme directement",
                "Tu connais deja le probleme ? Decris-le ici, et passe a "
                "l'analyse.")
        col1, col2 = st.columns(2)
        with col1:
            machine_d = st.text_input(
                "Machine ou processus",
                placeholder="ex: M3 - Soudeuse",
                key="f1_dir_machine",
            )
            type_d = st.selectbox(
                "Type de probleme", TYPES_INCIDENT, key="f1_dir_type",
            )
        with col2:
            sev_d = st.select_slider(
                "Severite",
                options=SEVERITES, value="haute",
                key="f1_dir_severite",
            )
            user_d = st.text_input(
                "Decrit par",
                value=st.session_state.get("user_name", ""),
                key="f1_dir_nom",
            )
        desc_d = st.text_area(
            "Description du probleme",
            placeholder=("Sois precis : quel parametre / quel produit / "
                         "depuis quand / impact constate..."),
            height=130,
            key="f1_dir_desc",
        )
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            if st.button(
                "Creer et passer a l'analyse",
                type="primary",
                use_container_width=True,
                disabled=not (machine_d.strip() and desc_d.strip()),
                key="f1_dir_submit",
            ):
                iid = db.create_incident(
                    machine=machine_d,
                    description=desc_d,
                    operateur_nom=user_d,
                    type_incident=type_d,
                    severite=sev_d,
                    cree_par_role=role,
                )
                db.update_incident_statut(iid, "fiabilise")
                st.success(
                    f"Probleme #{iid} cree. Tu peux analyser maintenant."
                )
                st.rerun()
        with col_info:
            st.caption("Le probleme passera directement en file d'analyse "
                       "(pas besoin de fiabilisation Tech N+1).")


# ---------------------------------------------------------------------------
# LISTE des incidents en cours
# ---------------------------------------------------------------------------
def _render_liste_incidents(role: str) -> None:
    with st.container(border=True):
        section("Problemes en cours",
                "A reprendre pour analyser ou fiabiliser.")
        df = db.list_incidents()
        if df.empty:
            st.info("Aucun probleme enregistre pour l'instant.")
            return
        # Filtre les statuts qui interessent F1
        df = df[df["statut"].isin(["brut", "fiabilise"])]
        if df.empty:
            st.info("Tous les problemes sont deja en cours d'analyse "
                    "ou clos.")
            return
        cols_show = ["id", "machine", "type_incident", "severite",
                     "statut", "operateur_nom", "description", "cree_le"]
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(
            df[cols_show], use_container_width=True, hide_index=True,
        )

        # Bouton Tech N+1 (fiabiliser) seulement si role concerne
        if role in ("technician", "ac_manager", "maintenance"):
            df_brut = df[df["statut"] == "brut"]
            if not df_brut.empty:
                col_sel, col_btn = st.columns([2, 1])
                with col_sel:
                    target_id = st.number_input(
                        "ID a fiabiliser (Tech N+1)",
                        min_value=1, step=1,
                        value=int(df_brut.iloc[0]["id"]),
                        key="f1_fiab_id",
                    )
                with col_btn:
                    st.markdown(
                        "<div style='height:28px'></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Marquer pret a analyser →",
                                 use_container_width=True,
                                 key="f1_fiab_btn"):
                        ok = db.update_incident_statut(
                            int(target_id), "fiabilise",
                        )
                        if ok:
                            st.success(
                                f"Probleme #{int(target_id)} pret a analyser."
                            )
                            st.rerun()
                        else:
                            st.error(f"Probleme #{int(target_id)} introuvable.")


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------
def render() -> None:
    role = st.session_state.get("role", "ac_manager")
    _kpi_bar()

    # --- Trois modes via onglets ----------------------------------------
    tab_op, tab_imp, tab_dir = st.tabs([
        "Signalement (operateur)",
        "Importer un fichier",
        "Saisie directe (responsable)",
    ])
    with tab_op:
        _mode_signalement(role)
    with tab_imp:
        _mode_import(role)
    with tab_dir:
        _mode_saisie_directe(role)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # --- Liste des problemes en cours -----------------------------------
    _render_liste_incidents(role)

    # --- CTA : Analyser la cause racine ---------------------------------
    _render_cta_analyser()

    # --- Gains & livrables de l'etape ------------------------------------
    render_synthese("f1")
