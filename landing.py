"""
LI90 - Landing page (refonte v0 / AlignPro edition).

- Hero Three.js (port vanilla du workflow-3d.tsx) : reseau neuronal +
  flux energetique + coeur IA + particules ambiantes.
- Deux boutons d'entree : Vue A (Stabiliser une ligne) / Vue B (Projet AC).
- Sections : workflow 6 etapes, KPI, CTA, footer.

Lance avec :  python -m streamlit run landing.py
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from ui_theme import (
    inject_theme,
    COLOR_BG, COLOR_CARD, COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_MUTED,
    COLOR_BORDER, COLOR_OK,
)


# ---------------------------------------------------------------------------
# Setup global
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LI90 - Le systeme nerveux de votre usine",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()


# ---------------------------------------------------------------------------
# Hero Three.js (port vanilla du composant workflow-3d.tsx)
# ---------------------------------------------------------------------------
HERO_THREEJS = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin:0; padding:0; background:#0a0a0f; overflow:hidden;
               font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               color:#f4f4f5; }
  #scene { position:absolute; top:0; left:0; width:100%; height:100%;
           opacity:0.75; }
  .overlay {
    position:absolute; top:0; left:0; right:0; bottom:0;
    background: linear-gradient(90deg, #0a0a0f 0%, rgba(10,10,15,0.6) 50%, transparent 100%);
    z-index: 1; pointer-events:none;
  }
  .content {
    position: absolute; z-index: 2;
    top: 50%; left: 6%; transform: translateY(-50%);
    max-width: 620px; pointer-events: none;
  }
  .tag {
    display:inline-flex; align-items:center; gap:8px;
    padding: 6px 14px; border-radius: 999px;
    background: rgba(249, 115, 22, 0.10);
    border: 1px solid rgba(249, 115, 22, 0.25);
    color: #f97316; font-size: 13px; font-weight: 500;
  }
  .tag .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #f97316;
    animation: pulseDot 2s infinite;
  }
  @keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(1.3); }
  }
  h1 {
    margin: 22px 0 0 0;
    font-size: clamp(2.4rem, 4.4vw, 3.6rem);
    line-height: 1.05;
    letter-spacing: -0.02em;
    font-weight: 800;
    color: #ffffff;
  }
  h1 .accent { color: #f97316; }
  .sub {
    margin-top: 22px;
    color: #a1a1aa;
    font-size: 1.05rem;
    line-height: 1.55;
    max-width: 560px;
  }
  .sub .bold { color: #ffffff; font-weight: 500; }
  .badges {
    margin-top: 38px;
    display: flex; gap: 28px;
    font-size: 13px; color: #71717a;
  }
  .badges .b { display:flex; align-items:center; gap:8px; }
  .badges .b svg { color: #22c55e; }
</style>
</head>
<body>
  <div id="scene"></div>
  <div class="overlay"></div>
  <div class="content">
    <div class="tag"><span class="dot"></span> Le systeme nerveux de votre usine</div>
    <h1>De l'incident a<br><span class="accent">l'excellence</span><br>operationnelle</h1>
    <p class="sub">
      LI90 guide votre equipe du signal terrain jusqu'a la solution deployee.
      <span class="bold">L'IA analyse. Vous validez. C'est tout.</span>
    </p>
    <div class="badges">
      <div class="b">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <span>Setup en 24h</span>
      </div>
      <div class="b">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <span>ROI visible en 30 jours</span>
      </div>
    </div>
  </div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
  const container = document.getElementById('scene');
  const W = () => container.clientWidth;
  const H = () => container.clientHeight;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, W()/H(), 0.1, 100);
  camera.position.set(0, 0, 6);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(W(), H());
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.3));
  const pl1 = new THREE.PointLight(0xf97316, 0.8); pl1.position.set(5,5,5);   scene.add(pl1);
  const pl2 = new THREE.PointLight(0xff6b35, 0.4); pl2.position.set(-5,-5,-5); scene.add(pl2);

  // ----- AI Core (icosahedron wireframe + glows) -----
  const coreGroup = new THREE.Group();
  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.25, 1),
    new THREE.MeshBasicMaterial({ color: 0xf97316, wireframe: true })
  );
  coreGroup.add(core);
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(0.40, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0xf97316, transparent: true, opacity: 0.15 })
  );
  coreGroup.add(glow);
  const outerGlow = new THREE.Mesh(
    new THREE.SphereGeometry(0.60, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0xf97316, transparent: true, opacity: 0.05 })
  );
  coreGroup.add(outerGlow);
  scene.add(coreGroup);

  // ----- Neural network (5 layers : Signal, Analyse, Validation, Faisabilite, Action) -----
  const networkGroup = new THREE.Group();
  const layersCfg = [3, 5, 6, 5, 3];
  const nodes = [];
  for (let L = 0; L < layersCfg.length; L++) {
    const x = (L - 2) * 1.2;
    const count = layersCfg[L];
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const radius = 0.8 + Math.random() * 0.3;
      const y = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const size = (L === 2) ? 0.12 : 0.08;
      const colorHex = (L === 2) ? 0xf97316 : 0xff8c42;

      const nodeMesh = new THREE.Mesh(
        new THREE.SphereGeometry(size, 16, 16),
        new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.9 })
      );
      nodeMesh.position.set(x, y, z);
      networkGroup.add(nodeMesh);

      const nodeGlow = new THREE.Mesh(
        new THREE.SphereGeometry(size * 2, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xf97316, transparent: true, opacity: 0.1 })
      );
      nodeGlow.position.set(x, y, z);
      networkGroup.add(nodeGlow);

      nodes.push({ pos: new THREE.Vector3(x, y, z), layer: L });
    }
  }
  // Connexions inter-couches
  for (let i = 0; i < nodes.length; i++) {
    for (let j = 0; j < nodes.length; j++) {
      if (nodes[j].layer === nodes[i].layer + 1) {
        const d = nodes[i].pos.distanceTo(nodes[j].pos);
        if (d < 2 && Math.random() > 0.3) {
          const geom = new THREE.BufferGeometry().setFromPoints([nodes[i].pos, nodes[j].pos]);
          const line = new THREE.Line(
            geom,
            new THREE.LineBasicMaterial({
              color: 0xf97316, transparent: true,
              opacity: 0.15 + Math.random() * 0.2,
            })
          );
          networkGroup.add(line);
        }
      }
    }
  }
  scene.add(networkGroup);

  // ----- Ambient particles (sphere autour de la scene) -----
  const AMB = 500;
  const ambPos = new Float32Array(AMB * 3);
  for (let i = 0; i < AMB; i++) {
    const r = 4 + Math.random() * 3;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    ambPos[i*3]   = r * Math.sin(ph) * Math.cos(th);
    ambPos[i*3+1] = r * Math.sin(ph) * Math.sin(th);
    ambPos[i*3+2] = r * Math.cos(ph);
  }
  const ambGeom = new THREE.BufferGeometry();
  ambGeom.setAttribute('position', new THREE.BufferAttribute(ambPos, 3));
  const ambient = new THREE.Points(ambGeom, new THREE.PointsMaterial({
    color: 0xf97316, size: 0.015, sizeAttenuation: true,
    transparent: true, opacity: 0.3, depthWrite: false,
  }));
  scene.add(ambient);

  // ----- Energy flow (particules orange en flux gauche -> droite) -----
  const FLOW = 800;
  const flowPos = new Float32Array(FLOW * 3);
  const flowVel = new Float32Array(FLOW * 3);
  for (let i = 0; i < FLOW; i++) {
    const t = Math.random();
    flowPos[i*3]   = (t - 0.5) * 6;
    const a = Math.random() * Math.PI * 2;
    const r = 0.5 + Math.random();
    flowPos[i*3+1] = Math.cos(a) * r;
    flowPos[i*3+2] = Math.sin(a) * r;
    flowVel[i*3]   = 0.5 + Math.random() * 0.5;
    flowVel[i*3+1] = (Math.random() - 0.5) * 0.1;
    flowVel[i*3+2] = (Math.random() - 0.5) * 0.1;
  }
  const flowGeom = new THREE.BufferGeometry();
  flowGeom.setAttribute('position', new THREE.BufferAttribute(flowPos, 3));
  const flow = new THREE.Points(flowGeom, new THREE.PointsMaterial({
    color: 0xf97316, size: 0.04, sizeAttenuation: true,
    transparent: true, opacity: 0.6, depthWrite: false,
    blending: THREE.AdditiveBlending,
  }));
  scene.add(flow);

  // ----- Animation loop -----
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const t  = clock.getElapsedTime();
    const dt = clock.getDelta();

    core.rotation.y = t * 0.5;
    core.rotation.x = t * 0.3;
    const s = 1 + Math.sin(t * 2) * 0.1;
    glow.scale.setScalar(s);

    networkGroup.rotation.y = Math.sin(t * 0.20) * 0.3;
    networkGroup.rotation.x = Math.sin(t * 0.15) * 0.1;

    ambient.rotation.y = t * 0.02;
    ambient.rotation.x = t * 0.01;

    const arr = flowGeom.attributes.position.array;
    for (let i = 0; i < FLOW; i++) {
      arr[i*3]   += flowVel[i*3] * dt * 2;
      arr[i*3+1] += Math.sin(t * 2 + i) * 0.002;
      arr[i*3+2] += Math.cos(t * 2 + i) * 0.002;
      if (arr[i*3] > 3) {
        arr[i*3] = -3;
        const a = Math.random() * Math.PI * 2;
        const r = 0.5 + Math.random();
        arr[i*3+1] = Math.cos(a) * r;
        arr[i*3+2] = Math.sin(a) * r;
      }
    }
    flowGeom.attributes.position.needsUpdate = true;

    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    camera.aspect = W() / H();
    camera.updateProjectionMatrix();
    renderer.setSize(W(), H());
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Sections en-dessous du hero (workflow 6 etapes, KPI, CTA, footer)
# ---------------------------------------------------------------------------
WORKFLOW_STEPS = [
    {"n": "1", "title": "Collecte",      "desc": "Saisie operateur",
     "detail": "Form + audio depuis le terrain, requete SQL auto"},
    {"n": "2", "title": "Cadrage",       "desc": "QQOQCP",
     "detail": "Le Resp. AC cadre le probleme avec precision"},
    {"n": "3", "title": "Cause racine",  "desc": "IA Ishikawa 5M",
     "detail": "Causes racines avec probabilites et niveau de preuve"},
    {"n": "4", "title": "Validation",    "desc": "0 reunion",
     "detail": "Resp. Prod + Tech N+1 valident dans le logiciel"},
    {"n": "5", "title": "Faisabilite",   "desc": "Gains par role",
     "detail": "Cout / temps / gain traduit pour chaque direction"},
    {"n": "6", "title": "Suivi",         "desc": "ROI reel",
     "detail": "Plan d'action priorise, statuts, ROI mesure"},
]

KPI_ITEMS = [
    {"metric": "3 min",  "label": "Cause racine identifiee",
     "desc": "L'IA analyse historique + capteurs + retours terrain"},
    {"metric": "0",      "label": "Reunions pour valider",
     "desc": "Chaque acteur valide dans son contexte, en parallele"},
    {"metric": "+23%",   "label": "OEE en moyenne",
     "desc": "Actions deployees avec micro-formations automatiques"},
]


def _render_workflow_section() -> None:
    """6 cartes workflow numerotees, style v0 AlignPro."""
    cards_html = ""
    for s in WORKFLOW_STEPS:
        cards_html += f"""
        <div class="wf-card">
          <div class="wf-num">{s['n']}</div>
          <div class="wf-title">{s['title']}</div>
          <div class="wf-desc">{s['desc']}</div>
          <div class="wf-detail">{s['detail']}</div>
        </div>
        """
    st.markdown(
        f"""
        <style>
          .wf-section {{
            padding: 80px 6% 60px 6%;
            background: #0d0d12;
          }}
          .wf-section .lbl {{
            color: {COLOR_PRIMARY}; font-size: 13px; font-weight: 500;
            text-align: center; display:block;
          }}
          .wf-section h2 {{
            color: #ffffff; font-size: 2.4rem; font-weight: 700;
            text-align: center; margin: 10px 0 6px 0;
            letter-spacing: -0.02em;
          }}
          .wf-section .sub {{
            color: {COLOR_TEXT_MUTED}; text-align: center;
            max-width: 580px; margin: 0 auto 50px auto;
          }}
          .wf-grid {{
            display: grid; gap: 16px;
            grid-template-columns: repeat(6, 1fr);
            max-width: 1200px; margin: 0 auto;
          }}
          @media (max-width: 1100px) {{
            .wf-grid {{ grid-template-columns: repeat(3, 1fr); }}
          }}
          @media (max-width: 640px) {{
            .wf-grid {{ grid-template-columns: repeat(2, 1fr); }}
          }}
          .wf-card {{
            position: relative;
            padding: 22px 16px 18px 16px;
            border-radius: 16px;
            background: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            transition: all 200ms;
          }}
          .wf-card:hover {{
            background: rgba(249, 115, 22, 0.08);
            border-color: rgba(249, 115, 22, 0.50);
            transform: translateY(-3px);
          }}
          .wf-num {{
            position: absolute; top: -12px; left: -12px;
            width: 28px; height: 28px; border-radius: 50%;
            background: {COLOR_PRIMARY};
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 12px; font-weight: 700;
            box-shadow: 0 0 18px rgba(249,115,22,0.45);
          }}
          .wf-title {{
            color: #ffffff; font-weight: 600; font-size: 15px;
            margin-bottom: 4px;
          }}
          .wf-desc {{
            color: {COLOR_PRIMARY}; font-size: 13px; font-weight: 500;
            margin-bottom: 8px;
          }}
          .wf-detail {{
            color: #71717a; font-size: 12px; line-height: 1.45;
          }}
        </style>
        <div class="wf-section">
          <span class="lbl">Le flux complet</span>
          <h2>Un workflow qui vous guide</h2>
          <p class="sub">Chaque etape est claire. Vous savez toujours ou vous en
          etes et quoi faire ensuite.</p>
          <div class="wf-grid">{cards_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_section() -> None:
    cards_html = ""
    for k in KPI_ITEMS:
        cards_html += f"""
        <div class="kpi-card">
          <div class="kpi-metric">{k['metric']}</div>
          <div class="kpi-label">{k['label']}</div>
          <div class="kpi-desc">{k['desc']}</div>
        </div>
        """
    st.markdown(
        f"""
        <style>
          .kpi-section {{
            padding: 80px 6%;
            background: {COLOR_BG};
          }}
          .kpi-grid {{
            display: grid; gap: 28px;
            grid-template-columns: repeat(3, 1fr);
            max-width: 1100px; margin: 0 auto;
          }}
          @media (max-width: 900px) {{
            .kpi-grid {{ grid-template-columns: 1fr; }}
          }}
          .kpi-card {{
            padding: 36px 28px;
            border-radius: 18px;
            background: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
          }}
          .kpi-metric {{
            font-size: 3rem; font-weight: 800; color: {COLOR_PRIMARY};
            line-height: 1; letter-spacing: -0.02em;
          }}
          .kpi-label {{
            color: #ffffff; font-size: 18px; font-weight: 600;
            margin: 12px 0 10px 0;
          }}
          .kpi-desc {{ color: #71717a; font-size: 14px; line-height: 1.55; }}
        </style>
        <div class="kpi-section">
          <div class="kpi-grid">{cards_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        f"""
        <style>
          .li90-footer {{
            padding: 24px 6%; border-top: 1px solid {COLOR_BORDER};
            background: {COLOR_BG};
            display: flex; align-items: center; justify-content: space-between;
            color: #71717a; font-size: 13px;
          }}
          .li90-footer .brand {{ display:flex; align-items:center; gap:10px;
                                 color: #f4f4f5; font-weight: 600; }}
          .li90-footer .brand .logo {{
            width: 28px; height: 28px; border-radius: 8px;
            background: linear-gradient(135deg, #f97316, #ea580c);
            display:flex; align-items:center; justify-content:center;
            font-weight:800; color:white; font-size:11px;
          }}
        </style>
        <div class="li90-footer">
          <div class="brand"><div class="logo">L90</div> LI90</div>
          <div>Le systeme nerveux industriel &middot; MVP v1 &middot; Pilote 3 mois sans engagement</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Workflow en 2 etapes : (1) Hero + choix Vue   (2) Choix Role + Entrer
# ---------------------------------------------------------------------------
ROLES = [
    {
        "id": "operator",
        "title": "Operateur",
        "icon": "●",
        "color": "#22c55e",
        "desc": "Signaler les incidents depuis le terrain",
        "tasks": ["Signaler en 30 sec", "Audio / photo", "Voir mes signalements"],
    },
    {
        "id": "technician",
        "title": "Technicien N+1",
        "icon": "■",
        "color": "#3b82f6",
        "desc": "Clarifier et enrichir les donnees",
        "tasks": ["Nettoyer les donnees", "Ajouter le contexte", "Valider les causes"],
    },
    {
        "id": "ac_manager",
        "title": "Responsable AC",
        "icon": "◆",
        "color": "#f97316",
        "desc": "Piloter l'amelioration continue",
        "tasks": ["Analyser les causes IA", "Valider les actions", "Suivre le deploiement"],
    },
    {
        "id": "production",
        "title": "Resp. Production",
        "icon": "▲",
        "color": "#a855f7",
        "desc": "Superviser et approuver",
        "tasks": ["Vue globale KPIs", "Approuver ROI", "Suivre les gains"],
    },
]

# State init
if "landing_step" not in st.session_state:
    st.session_state["landing_step"] = "hero"   # "hero" | "role"
if "view" not in st.session_state:
    st.session_state["view"] = "A"
if "role_selected" not in st.session_state:
    st.session_state["role_selected"] = None


def _go_to_role(view: str) -> None:
    st.session_state["view"] = view
    st.session_state["landing_step"] = "role"
    st.rerun()


def _back_to_hero() -> None:
    st.session_state["landing_step"] = "hero"
    st.session_state["role_selected"] = None
    st.rerun()


def _enter_app() -> None:
    st.session_state["entered_app"] = True
    st.session_state["role"] = st.session_state["role_selected"]
    st.switch_page("pages/app.py")


# ===========================================================================
# ETAPE 1 : HERO 3D + 2 boutons (Vue A / Vue B)
# ===========================================================================
if st.session_state["landing_step"] == "hero":
    components.html(HERO_THREEJS, height=620, scrolling=False)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col_l, col_a, col_b, col_r = st.columns([1, 2, 2, 1])
    with col_a:
        if st.button(
            "Stabiliser une ligne",
            type="secondary",
            use_container_width=True,
            help="Detection des derives SPC, Pareto, Ishikawa IA - ce que"
                 " vous avez deja construit.",
            key="btn_view_a",
        ):
            _go_to_role("A")
    with col_b:
        if st.button(
            "Projet d'amelioration continue",
            type="primary",
            use_container_width=True,
            help="Workflow complet : signalement operateur -> IA cause"
                 " racine -> validation -> deploiement.",
            key="btn_view_b",
        ):
            _go_to_role("B")

    st.markdown(
        f"""
        <div style='text-align:center; margin: 16px 0 0 0;
                   font-size:12px; color:{COLOR_TEXT_MUTED};'>
          Setup en 24h &middot; ROI visible en 30 jours &middot;
          <span style='color:{COLOR_OK};'>Pilote 3 mois sans engagement</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_workflow_section()
    _render_kpi_section()
    _render_footer()


# ===========================================================================
# ETAPE 2 : SELECTION DU ROLE (4 cartes)
# ===========================================================================
elif st.session_state["landing_step"] == "role":
    view_label = (
        "Stabiliser une ligne" if st.session_state["view"] == "A"
        else "Projet d'amelioration continue"
    )

    # CSS local pour les cartes de role
    st.markdown(
        f"""
        <style>
          .role-screen {{
            min-height: 75vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            padding: 24px 0 40px 0;
          }}
          .role-head {{ text-align: center; margin-bottom: 28px; }}
          .role-head .lbl {{
            color: {COLOR_PRIMARY}; font-size: 13px; font-weight: 500;
            display: inline-block; padding: 4px 10px;
            background: rgba(249,115,22,0.10);
            border: 1px solid rgba(249,115,22,0.25);
            border-radius: 999px;
          }}
          .role-head h1 {{
            color: #ffffff; font-size: 2.2rem; font-weight: 800;
            margin: 14px 0 6px 0; letter-spacing: -0.02em;
          }}
          .role-head p {{
            color: {COLOR_TEXT_MUTED}; font-size: 15px; margin: 0;
          }}
          .role-card {{
            border-radius: 18px; padding: 24px;
            background: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            transition: all 200ms;
            min-height: 230px;
            position: relative;
          }}
          .role-card.selected {{
            border-color: {COLOR_PRIMARY};
            background: rgba(249, 115, 22, 0.06);
            box-shadow: 0 0 0 1px {COLOR_PRIMARY} inset,
                        0 10px 36px rgba(249,115,22,0.15);
          }}
          .role-card .ico {{
            width: 52px; height: 52px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 22px; font-weight: 700;
            margin-bottom: 14px;
          }}
          .role-card .title {{
            color: #ffffff; font-size: 18px; font-weight: 700;
            letter-spacing: -0.01em; margin-bottom: 6px;
          }}
          .role-card .desc {{
            color: {COLOR_TEXT_MUTED}; font-size: 13px;
            line-height: 1.5; margin-bottom: 14px;
          }}
          .role-card .tasks {{ list-style: none; padding: 0; margin: 0; }}
          .role-card .tasks li {{
            color: #a1a1aa; font-size: 12.5px; line-height: 1.7;
            display: flex; align-items: center; gap: 8px;
          }}
          .role-card .tasks li::before {{
            content: ""; width: 6px; height: 6px; border-radius: 50%;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # En-tete
    st.markdown(
        f"""
        <div class="role-head">
          <div class="lbl">{view_label}</div>
          <h1>Bienvenue sur LI90</h1>
          <p>Selectionnez votre role pour personnaliser votre experience</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 cartes en colonnes
    cols = st.columns(4, gap="medium")
    for col, role in zip(cols, ROLES):
        with col:
            sel = (st.session_state.get("role_selected") == role["id"])
            sel_class = "selected" if sel else ""
            tasks_html = "".join(
                f'<li><span style="background:{role["color"]};'
                f'width:6px;height:6px;border-radius:50%;'
                f'display:inline-block;margin-right:8px;"></span>{t}</li>'
                for t in role["tasks"]
            )
            st.markdown(
                f"""
                <div class="role-card {sel_class}">
                  <div class="ico"
                       style="background:{role['color']}22; color:{role['color']};">
                    {role['icon']}
                  </div>
                  <div class="title">{role['title']}</div>
                  <div class="desc">{role['desc']}</div>
                  <ul class="tasks">{tasks_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Selectionne" if sel else f"Choisir {role['title']}",
                key=f"role_btn_{role['id']}",
                use_container_width=True,
                type="primary" if sel else "secondary",
            ):
                st.session_state["role_selected"] = role["id"]
                st.rerun()

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # Boutons Retour + Entrer
    col_back, col_spacer, col_enter = st.columns([1, 2, 2])
    with col_back:
        if st.button("← Retour", key="btn_back_hero", use_container_width=True):
            _back_to_hero()
    with col_enter:
        disabled = (st.session_state.get("role_selected") is None)
        if st.button(
            "Entrer dans le logiciel →",
            type="primary",
            disabled=disabled,
            use_container_width=True,
            key="btn_enter_app",
            help=("Choisis d'abord un role" if disabled else
                  "Demarre le logiciel avec ton role et ta vue"),
        ):
            _enter_app()

    _render_footer()
