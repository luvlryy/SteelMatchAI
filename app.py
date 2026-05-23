# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI (Versión Rediseñada)
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

# ── SILENCIAR ADVERTENCIAS MATEMÁTICAS EN LA INTERFAZ ──
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================================
# CONFIGURACIÓN GENERAL Y PALETA DE COLORES
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
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(color="#cbd5e1", family="monospace"),
    title_font=dict(color="#60a5fa", size=18),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#475569", borderwidth=1),
)


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="SteelMatch AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# DICCIONARIOS DE TRADUCCIÓN (MAPPING)
# ==========================================================

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
# ESTILO VISUAL — PALETA ACERO Y CORRECCIONES CSS
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

    /* Fondo principal: Gama de grises pizarra y acero */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
    }

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        color: #e2e8f0;
    }

    /* Títulos refinados - aspecto metálico pulido */
    h1 { 
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #94a3b8, #f8fafc, #94a3b8);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 5s linear infinite;
        font-size: 3rem !important;
        letter-spacing: -1px;
    }

    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    h2 {
        font-family: 'Syne', sans-serif !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
        border-left: 3px solid #60a5fa;
        padding-left: 12px;
        margin-top: 2rem !important;
    }

    h3 {
        font-family: 'Syne', sans-serif !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    /* Botón principal */
    .stButton>button {
        background: linear-gradient(135deg, #475569, #334155);
        color: #f8fafc;
        border: 1px solid #64748b;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 700;
        font-family: 'Syne', sans-serif;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #64748b, #475569);
        border: 1px solid #94a3b8;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(148,163,184,0.2);
    }

    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid rgba(148,163,184,0.15);
    }

    /* Métricas */
    [data-testid="metric-container"] {
        background: rgba(148,163,184,0.05);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 10px;
        padding: 16px;
        backdrop-filter: blur(8px);
    }

    /* ── CORRECCIÓN DE SELECTORES (Texto visible) ── */
    div[data-baseweb="select"] > div {
        background-color: #f8fafc !important; 
        border: 1px solid #94a3b8 !important;
    }
    div[data-baseweb="select"] * {
        color: #0f172a !important; /* Texto negro oscuro para contraste */
        font-weight: 600;
    }

    /* Cajas y menús de contenido */
    .conclusion-enriquecida {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid #64748b;
        border-radius: 12px;
        padding: 30px;
        margin-top: 2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    }
    
    .conclusion-enriquecida h3 {
        color: #f8fafc !important;
        font-size: 22px !important;
        margin-top: 0px !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        padding-bottom: 12px;
    }
    
    .grid-conclusion {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .card-conclusion-item {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 20px;
    }
    
    .card-conclusion-item strong {
        color: #60a5fa;
        font-size: 16px;
    }
    
    .card-conclusion-item p {
        margin: 8px 0 0 0 !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        color: #cbd5e1 !important;
    }

    .guia-didactica {
        background: rgba(30, 41, 59, 0.4);
        border-left: 4px solid #60a5fa;
        padding: 15px 20px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 20px;
    }
    
    .guia-didactica p {
        margin: 0 !important;
        font-size: 14px !important;
        color: #cbd5e1 !important;
    }

    .leyenda-grafica {
        background: rgba(15, 23, 42, 0.4);
        border: 1px dashed rgba(148, 163, 184, 0.3);
        padding: 18px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    /* Glosario Lateral */
    .glosario-lateral {
        background: rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
    .item-glosario-lat { margin-bottom: 12px; }
    .item-glosario-lat strong { color: #60a5fa; font-size: 13px; display: block; }
    .item-glosario-lat p { font-size: 11px !important; color: #94a3b8 !important; margin: 2px 0 0 0 !important; }
    .item-glosario-lat .tag-uso { font-size: 10px !important; color: #cbd5e1 !important; margin-top: 2px; display: block; }

    p, span, label, div { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
    .stTabs [aria-selected="true"] { color: #f8fafc !important; }
    [data-testid="metric-container"] * { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# FUNCIONES DE PROCESAMIENTO METALÚRGICO
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
    fig.update_xaxes(gridcolor="rgba(71,85,105,0.4)", zerolinecolor="rgba(148,163,184,0.3)")
    fig.update_yaxes(gridcolor="rgba(71,85,105,0.4)", zerolinecolor="rgba(148,163,184,0.3)")
    return fig


@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo):
    # En un entorno real se asume que el archivo existe en la ruta dada.
    try:
        aceros = pd.read_excel(nombre_archivo)
    except:
        # Failsafe dummy data por si no encuentra el excel al correrlo
        st.error(f"Asegúrate de que el archivo {nombre_archivo} exista en tu ruta.")
        st.stop()
        
    aceros.columns = aceros.columns.astype(str).str.strip()
    
    if "SAE Grade" in aceros.columns:
        aceros["SAE Grade"] = aceros["SAE Grade"].astype(str).str.strip()
    else:
        st.error("Falta columna 'SAE Grade'.")
        st.stop()

    columnas_necesarias = ["Conditions", "C (Min)", "C (Max)", "UTS (MPa)", "YS (MPa)", "Hardness (HB)", "Elongation (%)"]
    for col in columnas_necesarias:
        if col not in aceros.columns:
            st.error(f"Falta columna: {col}")
            st.stop()

    aceros["Conditions"] = aceros["Conditions"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    aceros["C (Min)"] = pd.to_numeric(aceros["C (Min)"], errors="coerce")
    aceros["C (Max)"] = pd.to_numeric(aceros["C (Max)"], errors="coerce")
    aceros["%C"] = (aceros["C (Min)"] + aceros["C (Max)"]) / 2

    for col in PROPIEDADES_MECANICAS:
        aceros[col] = pd.to_numeric(aceros[col], errors="coerce")

    aceros["Condition_simple"] = aceros["Conditions"].apply(agrupar_tratamiento)
    aceros["Temp_C"] = aceros["Conditions"].apply(extraer_temperatura)
    return aceros


# ==========================================================
# MENÚ LATERAL: CONFIGURACIÓN Y GLOSARIO INTEGRADO
# ==========================================================

# Carga directa de la base de datos (Sin subida manual)
aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

st.sidebar.header("⚙️ Configuración del Acero")

modo_usuario = st.sidebar.radio("👨‍💻 Elige tu perfil de uso:", ["Guiado (Por Niveles)", "Profesional (Control Manual)"])

if modo_usuario == "Guiado (Por Niveles)":
    deshabilitar_controles = True
    
    nivel_resistencia = st.sidebar.select_slider("💪 Nivel de Resistencia (UTS):", options=["Bajo", "Medio", "Alto"], value="Medio")
    nivel_dureza = st.sidebar.select_slider("💎 Nivel de Dureza Superficial:", options=["Bajo", "Medio", "Alto"], value="Medio")
    nivel_flexibilidad = st.sidebar.select_slider("🎗️ Nivel de Ductilidad:", options=["Bajo", "Medio", "Alto"], value="Medio")

    mapa_uts = {"Bajo": 380, "Medio": 600, "Alto": 950}
    mapa_ys = {"Bajo": 210, "Medio": 380, "Alto": 580}
    mapa_hb = {"Bajo": 105, "Medio": 180, "Alto": 300}
    mapa_elo = {"Bajo": 11, "Medio": 21, "Alto": 35}
    mapa_c = {"Bajo": 0.12, "Medio": 0.45, "Alto": 0.85}

    uts_val = mapa_uts[nivel_resistencia]
    ys_val = mapa_ys[nivel_resistencia]
    dureza_val = mapa_hb[nivel_dureza]
    elongacion_val = mapa_elo[nivel_flexibilidad]
    
    carbono_val = (mapa_c[nivel_resistencia] + mapa_c[nivel_dureza] + (1.0 - mapa_c[nivel_flexibilidad])) / 3
    carbono_val = np.clip(carbono_val, 0.05, 1.2)

    if nivel_resistencia == "Alto" or nivel_dureza == "Alto":
        st.sidebar.warning("⚠️ **Efecto Físico:** Al priorizar dureza, se requiere más Carbono. El acero perderá ductilidad.")
    elif nivel_flexibilidad == "Alto":
        st.sidebar.success("✅ **Efecto Físico:** Excelente maquinabilidad y soldabilidad, pero cederá ante fuerzas extremas.")

else:
    deshabilitar_controles = False
    carbono_val, uts_val, ys_val, dureza_val, elongacion_val = 0.45, 600, 400, 180, 20

carbono = st.sidebar.slider("% de Carbono deseado", 0.0, 1.5, float(carbono_val), 0.01, disabled=deshabilitar_controles)
uts = st.sidebar.slider("Resistencia UTS (MPa)", 200, 1500, int(uts_val), disabled=deshabilitar_controles)
ys = st.sidebar.slider("Límite Elástico YS (MPa)", 100, 1300, int(ys_val), disabled=deshabilitar_controles)
dureza = st.sidebar.slider("Dureza Superficial (HB)", 50, 500, int(dureza_val), disabled=deshabilitar_controles)
elongacion = st.sidebar.slider("Elongación / Ductilidad (%)", 1, 50, int(elongacion_val), disabled=deshabilitar_controles)

datos_modelo_limpio = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])
lista_originales = datos_modelo_limpio["Condition_simple"].dropna().unique().tolist()
lista_espanol = ["Todos"] + sorted([TRADUCCIONES_TRATAMIENTOS.get(t, t) for t in lista_originales])
tratamiento_elegido_espanol = st.sidebar.selectbox("Tratamiento térmico preferido", lista_espanol)

buscar = st.sidebar.button("🚀 Buscar material óptimo")

st.sidebar.markdown("---")
with st.sidebar.expander("📚 Glosario Práctico", expanded=True):
    st.markdown("""
    <div class="glosario-lateral">
        <div class="item-glosario-lat">
            <strong>💪 UTS (Resistencia Máxima)</strong>
            <p>Fuerza límite antes de fractura.</p>
            <span class="tag-uso">🔨 Ejes y piñones de impacto.</span>
        </div>
        <div class="item-glosario-lat">
            <strong>📐 YS (Límite Elástico)</strong>
            <p>Tolerancia al peso antes de deformarse.</p>
            <span class="tag-uso">🏗️ Vigas y soportes estructurales.</span>
        </div>
        <div class="item-glosario-lat">
            <strong>💎 Dureza (HB)</strong>
            <p>Resistencia a rayaduras y desgaste.</p>
            <span class="tag-uso">🔪 Herramientas y matrices.</span>
        </div>
        <div class="item-glosario-lat">
            <strong>🎗️ Elongación (%)</strong>
            <p>Capacidad de estiramiento y flexión.</p>
            <span class="tag-uso">🌀 Piezas moldeables en frío.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# ESTRUCTURA DE PESTAÑAS
# ==========================================================

st.title("SteelMatch AI")

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio", "📊 Exploración", "🌡️ Temperatura", "🧪 Influencia Fáctica", "🔍 Recomendador"
])


# ── TAB 1 — INICIO ──
with tab_inicio:
    st.header("Análisis Práctico de Propiedades del Acero")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>👋 <strong>Bienvenido al entorno de evaluación.</strong> Esta herramienta está diseñada para ingenieros que buscan entender rápidamente cómo la química (Carbono) y los tratamientos térmicos modifican el comportamiento físico de los aceros comerciales.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros en Base de Datos", len(aceros))
    col2.metric("Grados SAE Analizados", aceros["SAE Grade"].nunique())
    col3.metric("Tipos de Tratamientos", aceros["Condition_simple"].nunique())

    # Bloque de conclusiones movido exclusivamente aquí
    st.markdown("""
    <div class="conclusion-enriquecida">
        <h3>🛠️ Principios Clave del Acero para Ingeniería Práctica</h3>
        <p style="font-size: 15px !important; color: #cbd5e1 !important; line-height: 1.7 !important; margin: 0;">
            El comportamiento mecánico de los aceros al carbono obedece a reglas de causa y efecto directas. Resumen operativo:
        </p>
        <div class="grid-conclusion">
            <div class="card-conclusion-item">
                <strong>📈 Más Carbono (%C):</strong>
                <p>Aumenta drásticamente la dureza y resistencia mecánica (UTS). Ideal para componentes sujetos a desgaste continuo y fricción.</p>
            </div>
            <div class="card-conclusion-item">
                <strong>🔵 Recocido (Annealed):</strong>
                <p>Relaja el material. Baja la dureza y dispara la ductilidad (elongación). Facilita trabajos de doblez, soldadura y maquinado.</p>
            </div>
            <div class="card-conclusion-item">
                <strong>⚡ Estirado en Frío (Cold Drawn):</strong>
                <p>Deforma el grano mecánicamente. Eleva el límite elástico (YS), permitiendo que la pieza soporte mayores cargas estructurales sin ceder.</p>
            </div>
            <div class="card-conclusion-item">
                <strong>🌡️ Efecto Térmico:</strong>
                <p>El calor del horno permite reorganizar la estructura cristalina, logrando un balance fino entre tenacidad (absorción de golpes) y rigidez.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2 — EXPLORACIÓN ──
with tab_exploracion:
    st.header("📊 Exploración: Química vs Mecánica")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>💡 Visualiza la regla de oro: cómo el porcentaje de Carbono impacta el rendimiento físico del material.</p>
    </div>
    """, unsafe_allow_html=True)

    prop_c = st.selectbox("Elige la propiedad (Eje Y) para cruzarla con el Carbono:", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    df_carb["Tratamiento"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    fig_c = px.scatter(df_carb, x="%C", y=prop_c, color="Tratamiento", hover_data=["SAE Grade"], trendline="ols")
    
    # theme=None corrige el error de 'undefined'
    st.plotly_chart(aplicar_layout_plotly(fig_c), use_container_width=True, theme=None)


# ── TAB 3 — TEMPERATURA ──
with tab_temp:
    st.header("🌡️ Influencia de los Tratamientos Térmicos")
    
    prop_t = st.selectbox("Propiedad mecánica a evaluar frente a cambios de temperatura:", PROPIEDADES_MECANICAS, key="sb_p_t")
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    st.subheader("🔵 Comportamiento: Recocido")
    df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
    if not df_ann.empty:
        fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Blues")
        st.plotly_chart(aplicar_layout_plotly(fig_a), use_container_width=True, theme=None)
    else:
        st.warning("Datos insuficientes para mapeo térmico de recocido.")

    st.subheader("🟣 Comportamiento: Normalizado")
    df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
    if not df_norm.empty:
        fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Greys")
        st.plotly_chart(aplicar_layout_plotly(fig_n), use_container_width=True, theme=None)
    else:
        st.warning("Datos insuficientes para mapeo térmico de normalizado.")


# ── TAB 4 — ANOVA (REDISEÑO VISUAL) ──
with tab_anova:
    st.header("🧪 Impacto Tangible: ¿Qué controla tu Acero?")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>📊 <strong>Visión Práctica:</strong> En lugar de números estadísticos complejos, esta gráfica te muestra rápidamente <strong>quién tiene más culpa</strong> de alterar la propiedad que elijas: ¿los químicos que añadiste (Carbono) o el proceso en el horno (Tratamiento)?</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_anova = st.selectbox("Elige la propiedad que deseas evaluar:", PROPIEDADES_MECANICAS, key="sb_p_a")
    
    df_anova = aceros.dropna(subset=["%C", "Condition_simple", prop_anova]).copy()
    if len(df_anova) > 10:
        df_anova = df_anova.rename(columns={"%C": "Carbono", prop_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        mod = ols("Propiedad ~ Carbono + C(Tratamiento)", data=df_anova).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Influencia_%"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        
        # Extracción de porcentajes
        pct_carbono = tab_a.loc["Carbono", "Influencia_%"]
        pct_tratamiento = tab_a.loc["C(Tratamiento)", "Influencia_%"]
        pct_otros = tab_a.loc["Residual", "Influencia_%"]
        
        st.subheader(f"Nivel de Control sobre: {prop_anova}")
        
        # Gráfica de anillo (Donut) muy visual y fácil de leer
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Química (% Carbono)", "Proceso (Tratamiento Térmico)", "Otros Factores (Variabilidad)"],
            values=[pct_carbono, pct_tratamiento, pct_otros],
            hole=.5,
            marker_colors=["#60a5fa", "#64748b", "#334155"],
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="monospace"),
            showlegend=False
        )
        
        # theme=None previene el error undefined
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)


# ── TAB 5 — RECOMENDADOR INTELIGENTE ──
with tab_recomendador:
    st.header("🔍 Buscador de Materiales Comerciales")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>🤖 El sistema cruza tus parámetros deseados del panel izquierdo con la base de datos real para encontrar el grado comercial de acero que mejor se adapta a tus necesidades operativas.</p>
    </div>
    """, unsafe_allow_html=True)

    if not datos_modelo_limpio.empty:
        scaler = MinMaxScaler()
        scaler.fit(datos_modelo_limpio[COLUMNAS_MODELO])

        if buscar:
            df_res = datos_modelo_limpio.copy()
            if tratamiento_elegido_espanol != "Todos":
                tag_or = TRADUCCIONES_INVERSAS.get(tratamiento_elegido_espanol, "Other")
                df_res = df_res[df_res["Condition_simple"] == tag_or]

            if df_res.empty:
                st.warning("No hay coincidencias para ese tratamiento térmico específico en el almacén actual.")
            else:
                usr_in = np.array([[carbono, uts, ys, dureza, elongacion]])
                usr_norm = scaler.transform(usr_in)
                mat_norm = scaler.transform(df_res[COLUMNAS_MODELO])

                dist = euclidean_distances(usr_norm, mat_norm)
                idx = np.argmin(dist)
                match_mat = df_res.iloc[idx]
                pct_sim = 100 / (1 + dist[0][idx])

                st.success("🎯 Búsqueda analítica completada")
                
                st.subheader("🏆 Especificación Recomendada")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Grado SAE", f"{match_mat['SAE Grade']}")
                    st.metric("Tratamiento", TRADUCCIONES_TRATAMIENTOS.get(match_mat['Condition_simple'], match_mat['Condition_simple']))
                with c2:
                    st.metric("Similitud", f"{pct_sim:.1f}%")
                    st.metric("Carbono", f"{match_mat['%C']:.3f} %")
                with c3:
                    st.metric("UTS", f"{match_mat['UTS (MPa)']} MPa")
                    st.metric("Dureza", f"{match_mat['Hardness (HB)']} HB")

                st.markdown("---")
                st.subheader("📋 Ficha Técnica")
                st.dataframe(match_mat.to_frame(name="Valor Técnico"), use_container_width=True)
