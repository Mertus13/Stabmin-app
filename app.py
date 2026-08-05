"""
Dashboard de prédiction de la stabilité des pentes en carrière à ciel ouvert
Modèle : XGBoost Classifier
Auteur : Mertus — Rapport de stage UI2M
"""

import pickle
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")  # pas d'interface graphique nécessaire (évite tout blocage)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)

# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================
st.set_page_config(
    page_title="Stabilité des Pentes — Prédiction ML",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLE (CSS personnalisé)
# =========================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5a6c7d;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 15px 10px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #1e3a5f !important;
    }
    .stButton>button {
        background-color: #1e3a5f;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2c5282;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CHARGEMENT DU MODELE ET DES FEATURES (mis en cache)
# =========================================================
@st.cache_resource
def load_model_and_features():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model_and_features()

# Classe 0 = Stable | Classe 1 = Instable  (confirmé par l'utilisateur)
CLASS_LABELS = {0: "Stable", 1: "Instable"}
CLASS_COLORS = {0: "#2f9e44", 1: "#e03131"}

# Libellés lisibles pour le PDF (le "³" brut ne s'affiche pas avec les polices
# standard de reportlab -> on utilise la balise <super> à la place)
PDF_LABELS = {
    "Unit Weight (kN/m³)": "Poids unitaire (kN/m<super>3</super>)",
    "Cohesion (kPa)": "Cohésion (kPa)",
    "Internal Friction Angle (°)": "Angle de frottement interne (°)",
    "Slope Angle (°)": "Angle de la pente (°)",
    "Slope Height (m)": "Hauteur de la pente (m)",
    "Pore Water Pressure Ratio": "Ratio de pression interstitielle (ru)",
}

# =========================================================
# GENERATION DU RAPPORT PDF (graphiques rendus avec matplotlib
# — pas de dépendance à kaleido/Chrome, donc pas de risque de blocage)
# =========================================================
def _save_fig_to_buffer(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf_gauge(value, bar_color):
    """Jauge semi-circulaire (0-100) avec bandes rouge/jaune/vert."""
    fig, ax = plt.subplots(figsize=(6, 3.6), subplot_kw={"aspect": "equal"})
    bands = [(0, 40, "#ff8787"), (40, 70, "#ffd43b"), (70, 100, "#69db7c")]
    for start, end, color in bands:
        theta1, theta2 = 180 - (end / 100 * 180), 180 - (start / 100 * 180)
        ax.add_patch(mpatches.Wedge((0, 0), 1.0, theta1, theta2, width=0.3, facecolor=color, edgecolor="white"))
    needle_theta = np.radians(180 - (value / 100 * 180))
    ax.plot([0, 0.78 * np.cos(needle_theta)], [0, 0.78 * np.sin(needle_theta)], color=bar_color, linewidth=4)
    ax.add_patch(mpatches.Circle((0, 0), 0.05, facecolor=bar_color))
    ax.text(0, -0.35, f"{value:.1f}%", ha="center", va="center", fontsize=24, fontweight="bold", color="#1e3a5f")
    ax.text(0, -0.55, "Probabilité de stabilité", ha="center", va="center", fontsize=10, color="#5a6c7d")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.7, 1.1)
    ax.axis("off")
    return _save_fig_to_buffer(fig)


def build_pdf_bar_proba(proba):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    classes = ["Stable", "Instable"]
    values = [proba[0] * 100, proba[1] * 100]
    colors_ = ["#2f9e44", "#e03131"]
    bars = ax.bar(classes, values, color=colors_, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Probabilité (%)")
    ax.spines[["top", "right"]].set_visible(False)
    return _save_fig_to_buffer(fig)


def build_pdf_radar(labels, norm_values):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = norm_values + norm_values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 4.2), subplot_kw={"projection": "polar"})
    ax.plot(angles, values, color="#1e3a5f", linewidth=2)
    ax.fill(angles, values, color="#1e3a5f", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    return _save_fig_to_buffer(fig)


def build_pdf_importance(feature_names_list, importances):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(np.array(feature_names_list)[order], np.array(importances)[order], color="#3b6ea5")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    return _save_fig_to_buffer(fig)


def generate_pdf_report(input_df, reinforcement, label, confidence, proba, model, feature_names, numeric_cols, bounds):
    """Construit un rapport PDF complet (paramètres, résultats, graphiques)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=19,
        textColor=rl_colors.HexColor("#1e3a5f"), spaceAfter=4,
        alignment=0 
    )
    subtitle_style = ParagraphStyle(
        "SubtitleCustom", parent=styles["Normal"], fontSize=10,
        textColor=rl_colors.HexColor("#5a6c7d"), spaceAfter=18
    )
    section_style = ParagraphStyle(
        "SectionCustom", parent=styles["Heading2"], fontSize=13,
        textColor=rl_colors.HexColor("#1e3a5f"), spaceBefore=16, spaceAfter=8
    )
    result_color = rl_colors.HexColor(CLASS_COLORS[0] if label == "Stable" else CLASS_COLORS[1])
    result_style = ParagraphStyle(
        "ResultCustom", parent=styles["Heading1"], fontSize=16, textColor=result_color, spaceAfter=4
    )
    normal_style = styles["Normal"]

    story = []

    # ---- En-tête ----
    story.append(Paragraph("Rapport de Prédiction - Stabilité des Pentes", title_style))
    story.append(Paragraph(
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} - "
        f"Dashboard d'aide à la décision (UI2M)",
        subtitle_style
    ))

    # ---- Résultat principal ----
    story.append(Paragraph(f"Statut : {label}", result_style))
    story.append(Paragraph(f"Confiance du modèle : {confidence:.1f}%", normal_style))
    story.append(Spacer(1, 14))

    # ---- Table des paramètres saisis ----
    story.append(Paragraph("Paramètres géotechniques saisis", section_style))
    header_style = ParagraphStyle(
        "TableHeader", parent=normal_style, textColor=rl_colors.white, fontName="Helvetica-Bold"
    )
    data = [[Paragraph("Paramètre", header_style), Paragraph("Valeur", header_style)]]
    for col, display_label in PDF_LABELS.items():
        val = input_df[col].iloc[0]
        val_str = f"{val:.2f}" if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
        data.append([Paragraph(display_label, normal_style), val_str])
    data.append([Paragraph("Type de renforcement", normal_style), reinforcement])

    tbl = Table(data, colWidths=[10 * cm, 5.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#e6e9ef")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f8f9fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # ---- Table des probabilités ----
    story.append(Paragraph("Probabilités du modèle", section_style))
    proba_data = [
        ["Classe", "Probabilité"],
        ["Stable", f"{proba[0]*100:.1f}%"],
        ["Instable", f"{proba[1]*100:.1f}%"],
    ]
    proba_tbl = Table(proba_data, colWidths=[10 * cm, 5.5 * cm])
    proba_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#e6e9ef")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f8f9fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(proba_tbl)

    # ---- Graphiques (une page dédiée) ----
    story.append(PageBreak())
    story.append(Paragraph("Visualisations", section_style))

    result_color_hex = CLASS_COLORS[0] if label == "Stable" else CLASS_COLORS[1]

    # 1. Jauge de confiance
    gauge_buf = build_pdf_gauge(proba[0] * 100, result_color_hex)
    story.append(Paragraph("Indice de confiance", styles["Heading3"]))
    story.append(RLImage(gauge_buf, width=13 * cm, height=7.8 * cm))
    story.append(Spacer(1, 10))

    # 2. Probabilités par classe
    bar_buf = build_pdf_bar_proba(proba)
    story.append(Paragraph("Probabilités par classe", styles["Heading3"]))
    story.append(RLImage(bar_buf, width=13 * cm, height=7.8 * cm))
    story.append(Spacer(1, 10))

    # 3. Profil radar des paramètres saisis
    radar_labels = list(PDF_LABELS.values())
    radar_labels_plain = [l.replace("<super>3</super>", "3") for l in radar_labels]
    radar_vals = input_df[numeric_cols].iloc[0].values
    norm_vals = [
        (v - bounds[c][0]) / (bounds[c][1] - bounds[c][0])
        for c, v in zip(numeric_cols, radar_vals)
    ]
    radar_buf = build_pdf_radar(radar_labels_plain, norm_vals)
    story.append(Paragraph("Profil des paramètres saisis", styles["Heading3"]))
    story.append(RLImage(radar_buf, width=13 * cm, height=9 * cm))
    story.append(Spacer(1, 10))

    # 4. Importance des variables du modèle
    imp_buf = build_pdf_importance(feature_names, model.feature_importances_)
    story.append(Paragraph("Importance des variables", styles["Heading3"]))
    story.append(RLImage(imp_buf, width=13 * cm, height=9 * cm))
    story.append(Spacer(1, 10))

    # ---- Interprétation ----
    story.append(Spacer(1, 8))
    story.append(Paragraph("Interprétation", section_style))
    if label == "Stable":
        interp = (
            f"Le modèle prédit une pente <b>stable</b> avec {confidence:.1f}% de confiance. "
            "La cohésion et l'angle de frottement interne saisis semblent suffisants au regard "
            "de la géométrie et des conditions hydriques renseignées."
        )
    else:
        interp = (
            f"Le modèle prédit une pente <b>instable</b> avec {confidence:.1f}% de confiance. "
            "Il est recommandé d'envisager un renforcement (soil nailing, mur de soutènement, "
            "géosynthétiques) ou de revoir la géométrie de la pente (angle/hauteur) avant exploitation."
        )
    story.append(Paragraph(interp, normal_style))

    # ---- Pied de page ----
    story.append(Spacer(1, 26))
    story.append(Paragraph(
        "Rapport technique - Stabilité des pentes / Modèle prédictif XGBoost",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=rl_colors.HexColor("#8a94a3"))
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================================================
# EN-TETE
# =========================================================
st.markdown('<div class="main-header">⛰️ Prédiction de la Stabilité des Pentes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Plateforme d\'aide à la décision pour la sécurisation des fronts de taille et des pentes minières</div>',
    unsafe_allow_html=True
)
st.divider()

# =========================================================
# BARRE LATERALE — FORMULAIRE DE SAISIE
# =========================================================
st.sidebar.header("📋 Paramètres géotechniques")
st.sidebar.caption("Renseignez les caractéristiques de la pente à évaluer.")

with st.sidebar.form("input_form"):

    st.subheader("Propriétés du sol")
    unit_weight = st.slider("Poids unitaire — Unit Weight (kN/m³)", 15.0, 30.0, 20.0, 0.1)
    cohesion = st.slider("Cohésion — Cohesion (kPa)", 0.0, 100.0, 25.0, 0.5)
    friction_angle = st.slider("Angle de frottement interne (°)", 10.0, 45.0, 30.0, 0.5)

    st.subheader("Géométrie de la pente")
    slope_angle = st.slider("Angle de la pente — Slope Angle (°)", 10.0, 80.0, 35.0, 0.5)
    slope_height = st.slider("Hauteur de la pente — Slope Height (m)", 5.0, 150.0, 30.0, 1.0)

    st.subheader("Conditions hydriques")
    pore_pressure_ratio = st.slider("Ratio de pression interstitielle (ru)", 0.0, 0.8, 0.2, 0.01)

    st.subheader("Type de renforcement")
    reinforcement = st.selectbox(
        "Méthode de renforcement appliquée",
        ["Aucun", "Geosynthetics", "Retaining Wall", "Soil Nailing"]
    )

    submitted = st.form_submit_button("🔍 Lancer la prédiction")

# =========================================================
# CONSTRUCTION DU VECTEUR D'ENTREE (dans l'ordre exact de features.pkl)
# =========================================================
def build_input_row():
    row = {
        "Unit Weight (kN/m³)": unit_weight,
        "Cohesion (kPa)": cohesion,
        "Internal Friction Angle (°)": friction_angle,
        "Slope Angle (°)": slope_angle,
        "Slope Height (m)": slope_height,
        "Pore Water Pressure Ratio": pore_pressure_ratio,
        "Reinforcement Type_Geosynthetics": 1 if reinforcement == "Geosynthetics" else 0,
        "Reinforcement Type_Retaining Wall": 1 if reinforcement == "Retaining Wall" else 0,
        "Reinforcement Type_Soil Nailing": 1 if reinforcement == "Soil Nailing" else 0,
    }
    return pd.DataFrame([row])[feature_names]  # réordonne selon features.pkl

# =========================================================
# INITIALISATION DE L'ETAT (pour garder les résultats affichés)
# =========================================================
if "has_prediction" not in st.session_state:
    st.session_state.has_prediction = False

if submitted:
    st.session_state.has_prediction = True
    st.session_state.input_df = build_input_row()
    st.session_state.pred_class = int(model.predict(st.session_state.input_df)[0])
    st.session_state.pred_proba = model.predict_proba(st.session_state.input_df)[0]

# =========================================================
# ZONE PRINCIPALE — RESULTATS
# =========================================================
if not st.session_state.has_prediction:
    st.info("👈 Renseignez les paramètres dans le menu latéral puis cliquez sur **Lancer la prédiction** pour obtenir un résultat.")
else:
    pred_class = st.session_state.pred_class
    proba = st.session_state.pred_proba
    input_df = st.session_state.input_df

    label = CLASS_LABELS[pred_class]
    color = CLASS_COLORS[pred_class]
    confidence = proba[pred_class] * 100

    # ---- Bandeau de résultat ----
    st.markdown(
        f"""
        <div style="background-color:{color}15; border-left: 6px solid {color};
                    border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;">
            <span style="font-size:1.6rem; font-weight:700; color:{color};">
                {"✅" if pred_class == 0 else "⚠️"} Pente prédite : {label}
            </span><br>
            <span style="font-size:1rem; color:#333;">
                Confiance du modèle : <b>{confidence:.1f}%</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Métriques clés ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Statut prédit", label)
    col2.metric("Probabilité Stable", f"{proba[0]*100:.1f}%")
    col3.metric("Probabilité Instable", f"{proba[1]*100:.1f}%")
    col4.metric("Renforcement", reinforcement)

    st.divider()

    # ---- Graphiques ----
    g1, g2 = st.columns([1, 1])

    with g1:
        st.subheader("🎯 Indice de confiance")
        gauge_value = proba[0] * 100  # probabilité "Stable"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Probabilité de stabilité (%)", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1e3a5f"},
                "steps": [
                    {"range": [0, 40], "color": "#ff4141"},
                    {"range": [40, 70], "color": "#ffd92f"},
                    {"range": [70, 100], "color": "#29b43b"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85,
                    "value": gauge_value,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=50, b=30, l=40, r=40))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g2:
        st.subheader("📊 Probabilités par classe")
        proba_df = pd.DataFrame({
            "Classe": ["Stable", "Instable"],
            "Probabilité (%)": [proba[0]*100, proba[1]*100]
        })
        fig_bar = px.bar(
            proba_df, x="Classe", y="Probabilité (%)",
            color="Classe",
            color_discrete_map={"Stable": "#2f9e44", "Instable": "#e03131"},
            text="Probabilité (%)"
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(height=300, showlegend=False, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    g3, g4 = st.columns([1, 1])

    with g3:
        st.subheader("🧭 Profil des paramètres saisis")
        numeric_cols = feature_names[:6]
        radar_vals = input_df[numeric_cols].iloc[0].values
        # Normalisation simple (0-1) pour affichage radar, bornes indicatives
        bounds = {
            "Unit Weight (kN/m³)": (15, 30),
            "Cohesion (kPa)": (0, 100),
            "Internal Friction Angle (°)": (10, 45),
            "Slope Angle (°)": (10, 80),
            "Slope Height (m)": (5, 150),
            "Pore Water Pressure Ratio": (0, 0.8),
        }
        norm_vals = [
            (v - bounds[c][0]) / (bounds[c][1] - bounds[c][0])
            for c, v in zip(numeric_cols, radar_vals)
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=numeric_cols + [numeric_cols[0]],
            fill="toself",
            name="Valeurs saisies",
            line_color="#1e3a5f"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            showlegend=False, height=350, margin=dict(t=30, b=10, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with g4:
        st.subheader("⭐ Importance des variables (modèle)")
        imp_df = pd.DataFrame({
            "Variable": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Importance", y="Variable", orientation="h",
            color="Importance", color_continuous_scale="Blues"
        )
        fig_imp.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ---- Tableau récapitulatif des données saisies ----
    with st.expander("📄 Voir les données saisies (format tableau)"):
        display_df = input_df.T.reset_index()
        display_df.columns = ["Paramètre", "Valeur"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---- Recommandation textuelle ----
    st.subheader("💡 Interprétation")
    if pred_class == 0:
        st.success(
            f"Le modèle prédit une pente **stable** avec {confidence:.1f}% de confiance. "
            "La cohésion et l'angle de frottement interne saisis semblent suffisants au regard "
            "de la géométrie et des conditions hydriques renseignées."
        )
    else:
        st.error(
            f"Le modèle prédit une pente **instable** avec {confidence:.1f}% de confiance. "
            "Il est recommandé d'envisager un renforcement (soil nailing, mur de soutènement, "
            "géosynthétiques) ou de revoir la géométrie de la pente (angle/hauteur) avant exploitation."
        )

    st.divider()

    # ---- Téléchargement du rapport PDF ----
    st.subheader("📥 Export du rapport")
    numeric_cols_pdf = feature_names[:6]
    bounds_pdf = {
        "Unit Weight (kN/m³)": (15, 30),
        "Cohesion (kPa)": (0, 100),
        "Internal Friction Angle (°)": (10, 45),
        "Slope Angle (°)": (10, 80),
        "Slope Height (m)": (5, 150),
        "Pore Water Pressure Ratio": (0, 0.8),
    }
    try:
        with st.spinner("Préparation du rapport PDF..."):
            pdf_buffer = generate_pdf_report(
                input_df=input_df,
                reinforcement=reinforcement,
                label=label,
                confidence=confidence,
                proba=proba,
                model=model,
                feature_names=feature_names,
                numeric_cols=numeric_cols_pdf,
                bounds=bounds_pdf,
            )
        st.download_button(
            label="📄 Télécharger le rapport PDF complet",
            data=pdf_buffer,
            file_name=f"rapport_stabilite_pente_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Erreur lors de la génération du PDF : {e}")

# =========================================================
# PIED DE PAGE
# =========================================================
st.divider()
st.markdown("**Auteur :** Mertus YANOGO")
st.markdown("**Université :** Universal Institutes Mining Management")
