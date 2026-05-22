# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
Versión Definitiva — Glosario Humano Permanente, Interfaz de Alto Impacto y 100% en Español
"""

import re
import io
import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

DEFAULT_DATA_FILE = "Data/steel_carbon_data.xlsx"

PROPIEDADES_MECANICAS = [
    "UTS (MPa)",
    "YS (MPa)",
    "Hardness (HB)",
    "Elongation (%)"
]

COLUMNAS_MODELO = [
    "%C",
    "UTS (MPa)",
    "YS (MPa)",
    "Hardness (HB)",
    "Elongation (%)"
]

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(15,23,42,0)",   
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#e2e8f0", family="monospace"),
    title_font=dict(color="#38bdf8", size=18),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#334155", borderwidth=1),
)

st.set_page_config(
    page_title="SteelMatch AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

TRADUCCIONES_TRATAMIENTOS = {
    "Annealed": "Recocido",
    "Normalized": "Normalizado",
    "Hot Rolled": "Laminado en caliente",
    "Cold Drawn": "Estirado en frío",
    "Quenched": "Templado",
    "Tempered": "Revenido",
    "As Rolled": "Laminado en bruto",
    "Other": "Otros"
}
TRADUCCIONES_INVERSAS = {v: k for k, v in TRADUCCIONES_TRATAMIENTOS.items()}

# ==========================================================
# ESTILO VISUAL — CSS CUSTOM PREMIUM
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

    .stApp {
        background:
        radial-gradient(circle at top left, rgba(56,189,248,0.15), transparent 30%),
        radial-gradient(circle at bottom right, rgba(99,102,241,0.12), transparent 30%),
        #020617;
        background-attachment: fixed;
    }

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        color: #e2e8f0;
    }

    h1 { 
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 4s linear infinite;
        font-size: 3rem !important;
        letter-spacing: -1px;
    }

    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    h2 {
        font-family: 'Syne', sans-serif !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
        border-left: 3px solid #38bdf8;
        padding-left: 12px;
        margin-top: 1.5rem !important;
    }

    /* ── TARJETAS DEL GLOSARIO UNIVERSAL ── */
    .contenedor-glosario {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }

    .tarjeta-glosario {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease;
    }
    
    .tarjeta-glosario:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }

    .tarjeta-glosario strong {
        color: #38bdf8;
        font-family: 'Syne', sans-serif;
        font-size: 15px;
    }

    .tarjeta-glosario .desc {
        font-size: 12px;
        color: #94a3b8;
        margin: 4px 0;
    }

    .tarjeta-glosario .uso {
        font-size: 11px;
        color: #34d399;
        background: rgba(52, 211, 153, 0.1);
        padding: 4px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 5px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 700;
        font-family: 'Syne', sans-serif;
        box-shadow: 0 4px 20px rgba(56,189,248,0.3);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }

    .guia-didactica {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #6366f1;
        padding: 15px 20px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 20px;
    }

    .conclusion-enriquecida {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 2px solid #38bdf8;
        border-radius: 20px;
        padding: 35px;
        margin-top: 3rem;
    }

    p, span, label, div { color: #e2e8f0 !important; }
    .stSelectbox label { color: #f8fafc !important; }
    .stTabs [data-baseweb="tab"] { color: #cbd5e1 !important; }
    .stTabs [aria-selected="true"] { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# FUNCIONES DE PROCESAMIENTO
# ==========================================================

def agrupar_tratamiento(texto):
    texto = str(texto).lower()
    if "annealed" in texto: return "Annealed"
    elif "normalized" in texto: return "Normalized"
    elif "hot rolled" in texto or "hot-rolled" in texto: return "Hot Rolled"
    elif "cold drawn" in texto or "cold-drawn" in texto: return "Cold Drawn"
    elif "quenched" in texto: return "Quenched"
    elif "tempered" in texto: return "Tempered"
    elif "as rolled" in texto: return "As Rolled"
    else: return "Other"

def extraer_temperatura(condicion):
    condicion = str(condicion)
    coincidencia = re.search(r"(\d+(?:\.\d+)?)\s*°\s*[Cc]", condicion)
    if coincidencia:
        temp = float(coincidencia.group(1))
        if 100 <= temp <= 1300: return temp
        return np.nan
    return np.nan

def aplicar_layout_plotly(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(56,189,248,0.2)")
    fig.update_yaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(56,189,248,0.2)")
    return fig

def leer_archivo_datos(nombre_archivo, archivo_bytes=None):
    extension = nombre_archivo.lower().split(".")[-1]
    obtener_fuente = lambda: io.BytesIO(archivo_bytes) if archivo_bytes is not None else nombre_archivo
    if extension in ["xlsx", "xls"]:
        return pd.read_excel(obtener_fuente())
    elif extension == "csv":
        return pd.read_csv(obtener_fuente())
    st.stop()

@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo, archivo_bytes=None):
    aceros = leer_archivo_datos(nombre_archivo, archivo_bytes)
    aceros.columns = aceros.columns.astype(str).str.strip()
    aceros["C (Min)"] = pd.to_numeric(aceros["C (Min)"], errors="coerce")
    aceros["C (Max)"] = pd.to_numeric(aceros["C (Max)"], errors="coerce")
    aceros["%C"] = (aceros["C (Min)"] + aceros["C (Max)"]) / 2
    for col in PROPIEDADES_MECANICAS:
        aceros[col] = pd.to_numeric(aceros[col], errors="coerce")
    aceros["Condition_simple"] = aceros["Conditions"].apply(agrupar_treatment if 'agrupar_treatment' in globals() else agrupar_tratamiento)
    aceros["Temp_C"] = aceros["Conditions"].apply(extraer_temperatura)
    return aceros

# ==========================================================
# MENÚ LATERAL INTERACTIVO (CORREGIDO)
# ==========================================================

st.sidebar.subheader("📁 Cargar Datos")
archivo_subido = st.sidebar.file_uploader("Subir base de datos (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    aceros = cargar_y_preprocesar_datos(archivo_subido.name, archivo_subido.getvalue())
else:
    aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración del Acero")

modo_usuario = st.sidebar.radio("👨‍💻 Perfil de configuración:", ["Automático Guiado", "Manual Profesional"])

if modo_usuario == "Automático Guiado":
    bloquear_controles = True
    nivel_resistencia = st.sidebar.select_slider("Fuerza / Resistencia estructural:", options=["Bajo", "Medio", "Alto"], value="Medio")
    nivel_dureza = st.sidebar.select_slider("Resistencia a rayaduras / Dureza:", options=["Bajo", "Medio", "Alto"], value="Medio")
    nivel_flexibilidad = st.sidebar.select_slider("Ductilidad / Capacidad de doblado:", options=["Bajo", "Medio", "Alto"], value="Medio")

    mapa_uts = {"Bajo": 400, "Medio": 650, "Alto": 1000}
    mapa_ys = {"Bajo": 250, "Medio": 420, "Alto": 650}
    mapa_hb = {"Bajo": 120, "Medio": 190, "Alto": 320}
    mapa_elo = {"Bajo": 10, "Medio": 20, "Alto": 35}
    mapa_c = {"Bajo": 0.15, "Medio": 0.45, "Alto": 0.80}

    carbono_sug = (mapa_c[nivel_resistencia] + mapa_c[nivel_dureza] + (1.0 - mapa_c[nivel_flexibilidad])) / 3
    carbono_sug = np.clip(carbono_sug, 0.05, 1.2)
    uts_sug = mapa_uts[nivel_resistencia]
    ys_sug = mapa_ys[nivel_resistencia]
    dureza_sug = mapa_hb[nivel_dureza]
    elo_sug = mapa_elo[nivel_flexibilidad]
else:
    bloquear_controles = False
    carbono_sug, uts_sug, ys_sug, dureza_sug, elo_sug = 0.45, 600, 400, 180, 20

# Variables limpias y unificadas para evitar NameError
carbono = st.sidebar.slider("Porcentaje de Carbono (%)", 0.0, 1.5, float(carbono_sug), 0.01, disabled=bloquear_controles)
uts = st.sidebar.slider("Fuerza Máxima - UTS (MPa)", 200, 1500, int(uts_sug), disabled=bloquear_controles)
ys = st.sidebar.slider("Límite Elástico - YS (MPa)", 100, 1300, int(ys_sug), disabled=bloquear_controles)
dureza = st.sidebar.slider("Dureza del Metal (HB)", 50, 500, int(dureza_sug), disabled=bloquear_controles)
elongacion = st.sidebar.slider("Capacidad de Estiramiento (%)", 1, 50, int(elo_sug), disabled=bloquear_controles)

datos_modelo_limpio = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])
lista_originales = datos_modelo_limpio["Condition_simple"].dropna().unique().tolist()
lista_espanol = ["Todos"] + sorted([TRADUCCIONES_TRATAMIENTOS.get(t, t) for t in lista_originales])
tratamiento_elegido_espanol = st.sidebar.selectbox("Tratamiento en Horno Requerido", lista_espanol)

buscar = st.sidebar.button("🚀 Encontrar Acero Ideal")

# ==========================================================
# VISTA PRINCIPAL & GLOSARIO VISUAL PERMANENTE
# ==========================================================

st.title("SteelMatch AI")

# --- GLOSARIO UNIVERSAL PERMANENTE (Visible arriba de todas las pestañas) ---
st.markdown("""
<div class="contenedor-glosario">
    <div class="tarjeta-glosario">
        <strong>💪 UTS (Resistencia Máxima)</strong>
        <div class="desc">La fuerza límite de estiramiento que tolera el acero antes de romperse por completo.</div>
        <div class="uso">🔨 Excelente para: Ejes, engranajes pesados y piezas mecánicas de alto impacto.</div>
    </div>
    <div class="tarjeta-glosario">
        <strong>📐 YS (Límite Elástico)</strong>
        <div class="desc">El punto máximo donde el acero se estira pero regresa a su forma original sin quedar chueco.</div>
        <div class="uso">🏗️ Excelente para: Vigas, estructuras de edificios y puentes estructurales.</div>
    </div>
    <div class="tarjeta-glosario">
        <strong>💎 Dureza (HB)</strong>
        <div class="desc">La firmeza de la superficie. Evita que el metal se raye, desgaste o abolle con el uso continuo.</div>
        <div class="uso">🔪 Excelente para: Cuchillas, herramientas de corte y rieles de tren.</div>
    </div>
    <div class="tarjeta-glosario">
        <strong>🎗️ Elongación (% / Ductilidad)</strong>
        <div class="desc">Qué tanto se puede estirar y deformar el acero como chicle sin llegar a partirse.</div>
        <div class="uso">🌀 Excelente para: Alambres, láminas moldeables y piezas soldadas.</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio", "📊 Gráficas de Control", "🌡️ Comportamiento Térmico", "🧪 Estudio ANOVA", "🔍 Recomendador Inteligente"
])

# ── TAB 1 — INICIO ──
with tab_inicio:
    st.header("¡Bienvenido al Laboratorio Metalúrgico Digital!")
    st.markdown("""
    <div class="guia-didactica">
        <p>Ajustando los valores del menú izquierdo puedes simular aleaciones en tiempo real. 
        El sistema cruzará tus requerimientos con registros físicos de laboratorio para darte recomendaciones comerciales exactas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Muestras Analizadas", len(aceros))
    col2.metric("Grados SAE en Almacén", aceros["SAE Grade"].nunique() if "SAE Grade" in aceros.columns else "N/A")
    col3.metric("Tratamientos Distintos", aceros["Condition_simple"].nunique())

    st.subheader("📋 Muestras del Laboratorio (Vista Rápida)")
    st.dataframe(aceros.head(10), use_container_width=True)

# ── TAB 2 — EXPLORACIÓN ──
with tab_exploracion:
    st.header("📊 Impacto Químico del Carbono")
    prop_c = st.selectbox("Elige la propiedad mecánica para evaluar contra el nivel de Carbono", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    df_carb["Tratamiento"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    fig_c = px.scatter(df_carb, x="%C", y=prop_c, color="Tratamiento", hover_data=["SAE Grade"] if "SAE Grade" in df_carb.columns else [], trendline="ols")
    st.plotly_chart(aplicar_layout_plotly(fig_c), use_container_width=True)

# ── TAB 3 — TEMPERATURA ──
with tab_temp:
    st.header("🌡️ Control y Alteración por Temperatura en Horno")
    prop_t = st.selectbox("Propiedad a analizar frente al calor", PROPIEDADES_MECANICAS, key="sb_p_t")
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔵 Proceso de Recocido (Annealed)")
        df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
        if not df_ann.empty:
            fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Blues")
            st.plotly_chart(aplicar_layout_plotly(fig_a), use_container_width=True)
        else:
            st.caption("Faltan datos de temperatura de recocido.")
            
    with col_b:
        st.subheader("🟣 Proceso de Normalizado (Normalized)")
        df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
        if not df_norm.empty:
            fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Purples")
            st.plotly_chart(aplicar_layout_plotly(fig_n), use_container_width=True)
        else:
            st.caption("Faltan datos de temperatura de normalizado.")

# ── TAB 4 — ANOVA ──
with tab_anova:
    st.header("🧪 Análisis de Pesos Estadísticos (ANOVA)")
    prop_anova = st.selectbox("Propiedad Criterio", PROPIEDADES_MECANICAS, key="sb_p_a")
    df_anova = aceros.dropna(subset=["%C", "Condition_simple", prop_anova]).copy()
    if len(df_anova) > 10:
        df_anova = df_anova.rename(columns={"%C": "Carbono", prop_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        mod = ols("Propiedad ~ Carbono + C(Tratamiento)", data=df_anova).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Influencia_%"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        st.dataframe(tab_a, use_container_width=True)
    else:
        st.warning("Datos insuficientes para realizar el pesaje matemático.")

# ── TAB 5 — RECOMENDADOR ──
with tab_recomendador:
    st.header("🔍 Recomendador Inteligente por Similitud")
    if not datos_modelo_limpio.empty:
        scaler = MinMaxScaler()
        scaler.fit(datos_modelo_limpio[COLUMNAS_MODELO])

        if buscar:
            df_res = datos_modelo_limpio.copy()
            if tratamiento_elegido_espanol != "Todos":
                tag_or = TRADUCCIONES_INVERSAS.get(tratamiento_elegido_espanol, "Other")
                df_res = df_res[df_res["Condition_simple"] == tag_or]

            if df_res.empty:
                st.warning("No hay muestras con ese tratamiento en el almacén actual.")
            else:
                usr_in = np.array([[carbono, uts, ys, dureza, elongacion]])
                usr_norm = scaler.transform(usr_in)
                mat_norm = scaler.transform(df_res[COLUMNAS_MODELO])

                dist = euclidean_distances(usr_norm, mat_norm)
                idx = np.argmin(dist)
                match_mat = df_res.iloc[idx]
                pct_sim = 100 / (1 + dist[0][idx])

                st.success("🎯 ¡Material Encontrado!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Grado Comercial", f"SAE {match_mat['SAE Grade']}" if "SAE Grade" in match_mat else "Desconocido")
                c2.metric("Tratamiento de Fábrica", TRADUCCIONES_TRATAMIENTOS.get(match_mat['Condition_simple'], match_mat['Condition_simple']))
                c3.metric("Similitud Algorítmica", f"{pct_sim:.2f}%")
                
                st.subheader("📋 Datos Técnicos Completos")
                st.dataframe(match_mat.to_frame(name="Valor Real"), use_container_width=True)
