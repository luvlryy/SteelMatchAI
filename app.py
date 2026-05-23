# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI (Versión Futuristic Console)
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

# Diseño base para gráficas - Futuristic Console
# Textos blancos puros y cuadrícula neón sutil para visibilidad
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",   
    plot_bgcolor="rgba(15,23,42,0.4)", # Fondo de gráfica ligeramente visible
    font=dict(color="#ffffff", family="sans-serif", size=14),
    title_font=dict(color="#06b6d4", size=20, family="sans-serif"), # Cian brillante
    legend=dict(
        bgcolor="rgba(2, 6, 23, 0.9)", 
        bordercolor="#06b6d4", 
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#ffffff")
    ),
    margin=dict(t=60, b=40, l=40, r=40)
)


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="SteelMatch AI | Quantum Materials",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# DICCIONARIOS DE TRADUCCIÓN Y NOMBRES AMIGABLES
# ==========================================================

TRADUCCIONES_TRATAMIENTOS = {
    "Annealed": "Recocido (Enfriado Lento)",
    "Normalized": "Normalizado (Enfriado al Aire)",
    "Hot Rolled": "Laminado en Caliente",
    "Cold Drawn": "Estirado en Frío",
    "Quenched": "Templado (Enfriado Rápido)",
    "Tempered": "Revenido (Calentamiento Suave)",
    "As Rolled": "Laminado en Bruto",
    "Other": "Otros"
}

TRADUCCIONES_INVERSAS = {v: k for k, v in TRADUCCIONES_TRATAMIENTOS.items()}


# ==========================================================
# ESTILO VISUAL — TEMÁTICA FUTURISTA Y ALTA VISIBILIDAD
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');

    /* Fondo principal: Futuristic Grid */
    .stApp {
        background-color: #020617;
        background-image: 
            linear-gradient(rgba(14, 165, 233, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(14, 165, 233, 0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        background-position: -1px -1px;
        background-attachment: fixed;
    }

    /* TEXTO GENERAL BLANCO BRILLANTE PARA MÁXIMA LEGIBILIDAD */
    html, body, [class*="css"], p, span, label, li {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important; 
    }
    
    p {
        font-size: 15px !important;
        line-height: 1.6 !important;
    }

    /* Títulos agradables a la vista con degradado neón */
    h1 { 
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #34d399, #ffffff, #06b6d4);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 5s linear infinite;
        font-size: 2.8rem !important;
    }

    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    h2 {
        color: #06b6d4 !important; /* Cian Brillante neón */
        font-weight: 700 !important;
        border-left: 5px solid #06b6d4;
        padding-left: 15px;
        margin-top: 2.5rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    h3, h4 {
        color: #ffffff !important; 
        font-weight: 600 !important;
    }

    /* Selectores y campos de entrada: Futuristic Console look */
    /* Fondo oscuro transparente, borde cian neón, texto BLANCO puro */
    div[data-baseweb="select"] > div, .stNumberInput input, .stSlider > div {
        background-color: rgba(15, 23, 42, 0.9) !important; 
        border: 1px solid #06b6d4 !important; 
        border-radius: 8px;
        box-shadow: 0 0 5px rgba(6, 182, 212, 0.2);
    }
    
    /* Texto DENTRO del selector - BLANCO MÁXIMO */
    div[data-baseweb="select"] * {
        color: #ffffff !important; 
        font-weight: 600;
    }

    /* Estilo del botón principal - Neon Cyberpunk */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 700;
        font-family: 'Syne', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.6);
    }

    /* Cajas de introducción y guías (Mantenemos blanco puro adentro) */
    .intro-seccion {
        background: rgba(15, 23, 42, 0.6);
        border: 2px solid #06b6d4;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .intro-seccion h4 {
        margin-top: 0 !important;
        color: #34d399 !important; /* Verde menta brillante */
        margin-bottom: 10px !important;
    }
    .intro-seccion p {
        color: #ffffff !important; /* Texto blanco puro */
    }

    /* Caja de leyenda de gráficas */
    .leyenda-grafica {
        background: rgba(2, 6, 23, 0.8);
        border: 1px solid #34d399; /* Borde verde neón */
        border-left: 5px solid #34d399;
        padding: 15px 20px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 10px;
        margin-bottom: 25px;
        font-size: 15px;
        line-height: 1.6;
        color: #ffffff !important;
    }
    .leyenda-grafica strong {
        color: #34d399 !important;
    }

    /* Tarjetas de Conclusión en cuadrícula */
    .grid-conclusion {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    
    .card-conclusion-item {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-conclusion-item:hover {
        transform: translateY(-5px);
        background: rgba(15, 23, 42, 0.9);
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
    }
    
    .card-conclusion-item strong {
        color: #06b6d4 !important; /* Cian neón */
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-conclusion-item p {
        margin: 10px 0 0 0 !important;
        font-size: 15px !important;
        color: #ffffff !important; /* Blanco puro */
    }

    /* Caja de interpretación de ANOVA */
    .interpretacion-box {
        background: rgba(2, 6, 23, 0.8);
        border: 2px solid #34d399;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.2);
    }
    .interpretacion-box p { color: #ffffff !important; }
    .interpretacion-box h3 { color: #34d399 !important; }

    /* Ajustes específicos de las pestañas */
    .stTabs [data-baseweb="tab"] { 
        color: #ffffff !important; 
        font-weight: 600; 
        background-color: rgba(15, 23, 42, 0.3);
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] { 
        color: #06b6d4 !important; 
        font-weight: 800; 
        border-bottom-color: #06b6d4 !important; 
        background-color: rgba(15, 23, 42, 0.7) !important;
    }
    
    /* Métricas numéricas súper visibles */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #06b6d4 !important; font-size: 14px !important; text-transform: uppercase;}
    [data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 10px;
    }

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


def aplicar_layout_estetico(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    # Cuadrícula neón cian sutil
    fig.update_xaxes(gridcolor="rgba(6, 182, 212, 0.1)", zerolinecolor="rgba(6, 182, 212, 0.3)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(6, 182, 212, 0.1)", zerolinecolor="rgba(6, 182, 212, 0.3)", showgrid=True)
    # Puntos más grandes y brillantes con borde
    fig.update_traces(
        marker=dict(size=11, opacity=0.9, line=dict(width=1.5, color='#ffffff')),
        selector=dict(mode='markers')
    )
    return fig


@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo):
    try:
        aceros = pd.read_excel(nombre_archivo)
    except:
        st.error(f"Error: Asegúrate de que el archivo {nombre_archivo} exista en la carpeta correspondiente.")
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

aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

st.sidebar.header("⚙️ Configuración del Protocolo")

modo_usuario = st.sidebar.radio("👨‍💻 Modo de Operación:", ["Niveles Predefinidos (Guiado)", "Parámetros Manuales (Avanzado)"])

if modo_usuario == "Niveles Predefinidos (Guiado)":
    deshabilitar_controles = True
    
    nivel_resistencia = st.sidebar.select_slider("💪 Fuerza Estructural (UTS):", options=["Baja", "Media", "Alta"], value="Media")
    nivel_dureza = st.sidebar.select_slider("💎 Resistencia al Desgaste:", options=["Baja", "Media", "Alta"], value="Media")
    nivel_flexibilidad = st.sidebar.select_slider("🎗️ Ductilidad (Elongación):", options=["Baja", "Media", "Alta"], value="Media")

    mapa_uts = {"Baja": 380, "Media": 600, "Alta": 950}
    mapa_ys = {"Baja": 210, "Media": 380, "Alta": 580}
    mapa_hb = {"Baja": 105, "Media": 180, "Alta": 300}
    mapa_elo = {"Baja": 11, "Media": 21, "Alta": 35}
    mapa_c = {"Baja": 0.12, "Media": 0.45, "Alta": 0.85}

    uts_val = mapa_uts[nivel_resistencia]
    ys_val = mapa_ys[nivel_resistencia]
    dureza_val = mapa_hb[nivel_dureza]
    elongacion_val = mapa_elo[nivel_flexibilidad]
    
    carbono_val = (mapa_c[nivel_resistencia] + mapa_c[nivel_dureza] + (1.0 - mapa_c[nivel_flexibilidad])) / 3
    carbono_val = np.clip(carbono_val, 0.05, 1.2)

    if nivel_resistencia == "Alta" or nivel_dureza == "Alta":
        st.sidebar.warning("⚠️ **Alerta Técnica:** Niveles altos de dureza/fuerza requieren alta concentración de Carbono. La ductilidad del material disminuirá significativamente.")
    elif nivel_flexibilidad == "Alta":
        st.sidebar.success("✅ **Validación:** Excelente capacidad de moldeado y soldadura, pero menor soporte de cargas brutas.")

else:
    deshabilitar_controles = False
    carbono_val, uts_val, ys_val, dureza_val, elongacion_val = 0.45, 600, 400, 180, 20

st.sidebar.markdown("---")
carbono = st.sidebar.slider("Concentración de Carbono (%)", 0.0, 1.5, float(carbono_val), 0.01, disabled=deshabilitar_controles)
uts = st.sidebar.slider("Resistencia Máxima (MPa)", 200, 1500, int(uts_val), disabled=deshabilitar_controles)
ys = st.sidebar.slider("Límite Elástico - Punto de Cedencia (MPa)", 100, 1300, int(ys_val), disabled=deshabilitar_controles)
dureza = st.sidebar.slider("Dureza Superficial (HB)", 50, 500, int(dureza_val), disabled=deshabilitar_controles)
elongacion = st.sidebar.slider("Índice de Ductilidad - Elongación (%)", 1, 50, int(elongacion_val), disabled=deshabilitar_controles)

datos_modelo_limpio = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])
lista_originales = datos_modelo_limpio["Condition_simple"].dropna().unique().tolist()
lista_espanol = ["Todos"] + sorted([TRADUCCIONES_TRATAMIENTOS.get(t, t) for t in lista_originales])
tratamiento_elegido_espanol = st.sidebar.selectbox("Protocolo Térmico (Horno)", lista_espanol)

buscar = st.sidebar.button("🚀 Ejecutar Algoritmo de Búsqueda")

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Cripto-Glosario de Materiales", expanded=False):
    st.markdown("""
    <div style="font-size: 14px; color: #ffffff; line-height: 1.6;">
        <p><strong style="color:#06b6d4">💪 UTS (Fuerza Máxima):</strong> Cuánta fuerza bruta soporta antes de partirse en dos.</p>
        <p><strong style="color:#06b6d4">📐 YS (Límite Elástico):</strong> Cuánto peso aguanta antes de quedarse deformado para siempre.</p>
        <p><strong style="color:#06b6d4">💎 Dureza (HB):</strong> Qué tan difícil es rayarlo o abollarlo.</p>
        <p><strong style="color:#06b6d4">🎗️ Elongación (%):</strong> Qué tanto se estira como liga antes de romperse.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# ESTRUCTURA DE PESTAÑAS (Módulos)
# ==========================================================

st.title("SteelMatch AI // Quantum Materials Analysis")

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Módulo Central", "📊 Matriz Química", "🔥 Protocolo Térmico", "⚖️ Control de Variables", "🔍 Requisitor"
])


# ── TAB 1 — INICIO (Módulo Central) ──
with tab_inicio:
    st.markdown("""
    <div class="intro-seccion">
        <h4>👋 Bienvenido al Protocolo de Simulación de Aceros SAE.</h4>
        <p>Iniciando entorno interactivo para decodificar el comportamiento de aleaciones de acero. No se requieren conocimientos avanzados en metalurgia. Imagina que el acero es una fórmula de síntesis: si alteras los agentes químicos (como el Carbono) o modificas el protocolo de horneado (Tratamientos Térmicos), el output físico del material cambia drásticamente. ¡Usa los módulos superiores para explorar los datos!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Aceros SAE Analizados", len(aceros))
    col2.metric("Grados Comerciales identified", aceros["SAE Grade"].nunique())
    col3.metric("Protocolos Térmicos", aceros["Condition_simple"].nunique())

    st.markdown("""
    <div style="margin-top: 30px;">
        <h3>🛠️ Reporte Preliminar: Fundamentos Operativos</h3>
        <div class="grid-conclusion">
            <div class="card-conclusion-item">
                <strong>🧂 Agente Químico Principal (Carbono):</strong>
                <p>Al incrementar la concentración de carbono, el material incrementa su dureza y fuerza mecánica a nivel UTS. Ideal para componentes de desgaste y matrices de corte.</p>
            </div>
            <div class="card-conclusion-item">
                <strong>🛀 Protocolo de Relajación (Recocido):</strong>
                <p>Calentamiento seguido de un enfriamiento controlado y extremadamente lento. Reduce tensiones internas y dispara la ductilidad (elongación), facilitando el mecanizado y la deformación.</p>
            </div>
            <div class="card-conclusion-item">
                <strong>💪 Modificación por Tensión (En Frío):</strong>
                <p>Deformación mecánica a temperatura ambiente. Incrementa notablemente el límite elástico (YS), permitiendo que la pieza soporte cargas mayores sin deformarse permanentemente (vigas estructurales).</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2 — EXPLORACIÓN (Matriz Química) ──
with tab_exploracion:
    st.markdown("""
    <div class="intro-seccion">
        <h4>📊 Módulo de Análisis Químico vs Físico</h4>
        <p>Entorno para visualizar la causalidad directa: cómo el porcentaje de Carbono en la aleación impacta el rendimiento operativo final. Observa la regresión para determinar si la propiedad se potencia o degrada al aumentar la química.</p>
    </div>
    """, unsafe_allow_html=True)

    prop_c = st.selectbox("Selecciona la propiedad física a evaluar contra el Carbono:", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    df_carb["Protocolo Térmico"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        👀 <strong>Guía de Interpretación de Consola:</strong><br>
        • Cada unidad de datos (punto) es una muestra de acero SAE real.<br>
        • El eje X representa la concentración de Carbono. El eje Y representa el output de <strong>{prop_c}</strong>.<br>
        • Una pendiente ascendente indica correlación positiva (el carbono ayuda); descendente indica correlación negativa.
    </div>
    """, unsafe_allow_html=True)

    # Gráfica Neón usando paleta qualitative (Vibrante)
    fig_c = px.scatter(
        df_carb, x="%C", y=prop_c, 
        color="Protocolo Térmico", 
        hover_data=["SAE Grade"], 
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Alphabet # Paleta muy vibrante neón
    )
    st.plotly_chart(aplicar_layout_estetico(fig_c), use_container_width=True, theme=None)


# ── TAB 3 — TEMPERATURA (Protocolo Térmico) ──
with tab_temp:
    st.markdown("""
    <div class="intro-seccion">
        <h4>🔥 Módulo de Influencia Térmica // Grados Kelvin</h4>
        <p>Análisis del impacto de la temperatura del horno. El calor reorganiza la estructura molecular del metal para liberar estrés o cambiar sus fases cristalinas. ¡Observa cómo la temperatura dicta el balance final entre tenacidad y rigidez!</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_t = st.selectbox("Output físico a evaluar reaccionando al calor:", PROPIEDADES_MECANICAS, key="sb_p_t")
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        👀 <strong>Guía de Interpretación Térmica:</strong><br>
        • Eje X: Temperatura de proceso (°C). Eje Y: Output de <strong>{prop_t}</strong>.<br>
        • El color de la muestra indica concentración de carbono: Brillante = Alto, Oscuro = Bajo.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔵 Protocolo: Recocido // Enfriamiento Controlado Lento")
    df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
    if not df_ann.empty:
        # Escala neón Teal/verde para recocido
        fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Viridis")
        st.plotly_chart(aplicar_layout_estetico(fig_a), use_container_width=True, theme=None)
    else:
        st.warning("ERROR: Datos insuficientes registrados para la simulación térmica de recocido.")

    st.subheader("🟣 Protocolo: Normalizado // Enfriamiento al Aire libre")
    df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
    if not df_norm.empty:
        # Escala neón Púrpura para normalizado
        fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Plasma")
        st.plotly_chart(aplicar_layout_estetico(fig_n), use_container_width=True, theme=None)
    else:
        st.warning("ERROR: Datos insuficientes registrados para la simulación térmica de normalizado.")


# ── TAB 4 — ¿QUÉ INFLUYE MÁS? (Control de Variables // ANOVA Redesigned) ──
with tab_anova:
    st.markdown("""
    <div class="intro-seccion">
        <h4>⚖️ Algoritmo de Atribución de Variables // Juicio ANOVA</h4>
        <p>Resolución del dilema técnico: Cuando una propiedad física fluctúa... ¿Cuál es el factor determinante? El algoritmo ejecuta un análisis estadístico para atribuir qué porcentaje del impacto es culpa de la química (% Carbono) y cuánto es responsabilidad del protocolo del horno (Tratamiento). El output se presenta en una interfaz visual de control.</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_anova = st.selectbox("Selecciona la propiedad física para ejecutar el algoritmo de control:", PROPIEDADES_MECANICAS, key="sb_p_a")
    
    df_anova = aceros.dropna(subset=["%C", "Condition_simple", prop_anova]).copy()
    if len(df_anova) > 10:
        df_anova = df_anova.rename(columns={"%C": "Carbono", prop_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        mod = ols("Propiedad ~ Carbono + C(Tratamiento)", data=df_anova).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Influencia_%"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        
        pct_carbono = tab_a.loc["Carbono", "Influencia_%"]
        pct_tratamiento = tab_a.loc["C(Tratamiento)", "Influencia_%"]
        pct_otros = tab_a.loc["Residual", "Influencia_%"]
        
        st.subheader(f"Vector de Control de Atribución para: {prop_anova}")
        
        # Gráfica de anillo (Donut) con colores neón de alto contraste
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Factor 1: Química<br>(% Carbono)", "Factor 2: Protocolo<br>(Tratamiento Térmico)", "Factor 3: Residual<br>(Variabilidad/Misterios)"],
            values=[pct_carbono, pct_tratamiento, pct_otros],
            hole=.6,
            marker_colors=["#06b6d4", "#a78bfa", "#334155"], # Cian Neón, Violeta neón, Gris oscuro
            textinfo='label+percent',
            textfont_size=16,
            hoverinfo='label+percent'
        )])
        
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", family="Inter", size=14),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

        st.markdown("<div class='interpretacion-box'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Veredicto del Algoritmo de Control")
        if pct_carbono > pct_tratamiento:
            st.markdown(f"¡Resulta que **la Matriz Química es determinante**! Para modificar el/la **{prop_anova}** del metal, el porcentaje de Carbono es el factor más poderoso (representa el **{pct_carbono:.1f}%** de la atribución de control). No desgastes recursos alterando el protocolo del horno, ¡revisita la fórmula química!")
        else:
            st.markdown(f"¡Resulta que **el Protocolo Térmico es determinante**! Para modificar el/la **{prop_anova}**, la concentración química no es relevante. Lo que realmente altera el output físico (con un **{pct_tratamiento:.1f}%** de atribución) es si aplicas enfriamiento lento, rápido o deformación en frío. ¡Ajusta los parámetros de horneado!")
        st.markdown("</div>", unsafe_allow_html=True)


# ── TAB 5 — RECOMENDADOR INTELIGENTE (Requisitor SAE) ──
with tab_recomendador:
    st.markdown("""
    <div class="intro-seccion">
        <h4>🔍 Requisitor Mágico de Muestras SAE // Matching Artificial</h4>
        <p>Iniciando protocolo de matching. El sistema cruza tus requisitos operativos definidos en el menú de control izquierdo contra la base de datos de muestras reales para encontrar el grado comercial de acero exacto que cumple tus especificaciones técnicas.</p>
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
                st.warning("matching FAILED. No se encontraron muestras que coincidan con el protocolo térmico especificado. Reajusta los parámetros operativos.")
            else:
                usr_in = np.array([[carbono, uts, ys, dureza, elongacion]])
                usr_norm = scaler.transform(usr_in)
                mat_norm = scaler.transform(df_res[COLUMNAS_MODELO])

                dist = euclidean_distances(usr_norm, mat_norm)
                idx = np.argmin(dist)
                match_mat = df_res.iloc[idx]
                pct_sim = 100 / (1 + dist[0][idx])

                st.success("🎉 matching SUCCESSFUL. Grado SAE óptimo localizado.")
                
                st.subheader("🏆 Especificación SAE Recomendada:")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Grado Comercial Identificado:", f"SAE {match_mat['SAE Grade']}")
                    st.metric("Debe venir con Protocolo Térmico:", TRADUCCIONES_TRATAMIENTOS.get(match_mat['Condition_simple'], match_mat['Condition_simple']))
                with c2:
                    st.metric("Nivel de Compatibilidad:", f"{pct_sim:.1f}%")
                    st.metric("Agente Químico Real (%C):", f"{match_mat['%C']:.3f} %")
                with c3:
                    st.metric("Output UTS Real:", f"{match_mat['UTS (MPa)']} MPa")
                    st.metric("Output Dureza Real (HB):", f"{match_mat['Hardness (HB)']} HB")
