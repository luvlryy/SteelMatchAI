# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI (Versión Alto Contraste)
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

# Diseño base para gráficas - Letras blancas para máximo contraste
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",   
    plot_bgcolor="rgba(15,23,42,0.3)",
    font=dict(color="#ffffff", family="sans-serif", size=14),
    title_font=dict(color="#93c5fd", size=18, family="sans-serif"),
    legend=dict(
        bgcolor="rgba(15,23,42,0.9)", 
        bordercolor="#64748b", 
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
    page_title="SteelMatch AI",
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
# ESTILO VISUAL — ALTO CONTRASTE Y LEGIBILIDAD
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Fondo principal: Tonos oscuros suaves */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
    }

    /* TEXTO GENERAL BLANCO PARA LEER SIN ESFUERZO */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important; 
    }
    
    p, span, label, li {
        color: #ffffff !important;
    }

    /* Títulos agradables a la vista */
    h1 { 
        font-weight: 800 !important;
        background: linear-gradient(90deg, #e2e8f0, #ffffff, #e2e8f0);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 5s linear infinite;
        font-size: 2.5rem !important;
    }

    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    h2 {
        color: #93c5fd !important; /* Azul más brillante */
        font-weight: 700 !important;
        border-left: 4px solid #60a5fa;
        padding-left: 15px;
        margin-top: 2rem !important;
    }
    
    h3, h4 {
        color: #f8fafc !important; 
    }

    /* Selectores: El fondo es blanco, así que forzamos la letra a negro oscuro aquí */
    div[data-baseweb="select"] > div {
        background-color: #f8fafc !important; 
        border: 1px solid #94a3b8 !important;
        border-radius: 8px;
    }
    div[data-baseweb="select"] * {
        color: #020617 !important; /* Negro súper oscuro */
        font-weight: 700;
    }

    /* Cajas de introducción y guías (Textos en blanco puro) */
    .intro-seccion {
        background: rgba(96, 165, 250, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.4);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .intro-seccion h4 {
        margin-top: 0 !important;
        color: #bfdbfe !important; /* Azul hielo muy claro */
        margin-bottom: 10px !important;
    }
    .intro-seccion p {
        color: #ffffff !important;
        font-size: 15px !important;
    }

    .leyenda-grafica {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #34d399;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
        margin-bottom: 25px;
        font-size: 15px;
        line-height: 1.6;
        color: #ffffff !important;
    }

    /* Tarjetas de Conclusión */
    .grid-conclusion {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    
    .card-conclusion-item {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 20px;
        transition: transform 0.2s;
    }
    .card-conclusion-item:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.1);
    }
    
    .card-conclusion-item strong {
        color: #93c5fd !important;
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

    .interpretacion-box {
        background: linear-gradient(to right, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .interpretacion-box p, .interpretacion-box h3 {
        color: #ffffff !important;
    }

    /* Ajustes específicos de las pestañas */
    .stTabs [data-baseweb="tab"] { color: #cbd5e1 !important; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; font-weight: 800; border-bottom-color: #60a5fa !important; }
    
    /* Métricas numéricas súper visibles */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #bfdbfe !important; font-size: 14px !important; }

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
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)", showgrid=True)
    fig.update_traces(
        marker=dict(size=10, opacity=0.85, line=dict(width=1.5, color='rgba(255,255,255,0.8)')),
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

st.sidebar.header("⚙️ Diseña tu Acero")

modo_usuario = st.sidebar.radio("👨‍💻 ¿Cómo quieres buscar?", ["Quiero usar Niveles Simples", "Quiero Control Exacto (Experto)"])

if modo_usuario == "Quiero usar Niveles Simples":
    deshabilitar_controles = True
    
    nivel_resistencia = st.sidebar.select_slider("💪 Fuerza Máxima:", options=["Baja", "Media", "Alta"], value="Media")
    nivel_dureza = st.sidebar.select_slider("💎 Resistencia a rayones (Dureza):", options=["Baja", "Media", "Alta"], value="Media")
    nivel_flexibilidad = st.sidebar.select_slider("🎗️ Capacidad de estirarse:", options=["Baja", "Media", "Alta"], value="Media")

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
        st.sidebar.warning("⚠️ **Ten en cuenta:** Si quieres que sea muy duro y resistente, será difícil de doblar y podría quebrarse con un golpe seco.")
    elif nivel_flexibilidad == "Alta":
        st.sidebar.success("✅ **Ten en cuenta:** Será súper fácil de moldear y soldar, pero no soportará pesos extremos.")

else:
    deshabilitar_controles = False
    carbono_val, uts_val, ys_val, dureza_val, elongacion_val = 0.45, 600, 400, 180, 20

carbono = st.sidebar.slider("% de Carbono (Ingrediente principal)", 0.0, 1.5, float(carbono_val), 0.01, disabled=deshabilitar_controles)
uts = st.sidebar.slider("Fuerza para romperse - UTS (MPa)", 200, 1500, int(uts_val), disabled=deshabilitar_controles)
ys = st.sidebar.slider("Punto de doblez - YS (MPa)", 100, 1300, int(ys_val), disabled=deshabilitar_controles)
dureza = st.sidebar.slider("Dureza superficial (HB)", 50, 500, int(dureza_val), disabled=deshabilitar_controles)
elongacion = st.sidebar.slider("Flexibilidad - Elongación (%)", 1, 50, int(elongacion_val), disabled=deshabilitar_controles)

datos_modelo_limpio = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])
lista_originales = datos_modelo_limpio["Condition_simple"].dropna().unique().tolist()
lista_espanol = ["Todos"] + sorted([TRADUCCIONES_TRATAMIENTOS.get(t, t) for t in lista_originales])
tratamiento_elegido_espanol = st.sidebar.selectbox("¿Pasó por algún horno? (Tratamiento)", lista_espanol)

buscar = st.sidebar.button("🚀 Buscar el Acero Ideal")

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Minidiccionario para entender todo", expanded=True):
    st.markdown("""
    <div style="font-size: 14px; color: #ffffff; line-height: 1.6;">
        <p><strong>💪 UTS (Fuerza Máxima):</strong> Cuánta fuerza bruta soporta antes de partirse en dos.</p>
        <p><strong>📐 YS (Punto de Doblez):</strong> Cuánto peso aguanta antes de quedarse deformado (chueco) para siempre.</p>
        <p><strong>💎 Dureza (HB):</strong> Qué tan difícil es rayarlo o hacerle una abolladura.</p>
        <p><strong>🎗️ Elongación (%):</strong> Qué tanto se puede estirar como si fuera liga antes de romperse.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# ESTRUCTURA DE PESTAÑAS
# ==========================================================

st.title("Descubriendo el Acero")

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio", "📊 Jugar con la Química", "🔥 Jugar con el Horno", "⚖️ ¿Qué influye más?", "🔍 Buscador Mágico"
])


# ── TAB 1 — INICIO ──
with tab_inicio:
    st.markdown("""
    <div class="intro-seccion">
        <h4>👋 ¡Hola! Bienvenido a tu Simulador de Materiales</h4>
        <p>Aquí vas a encontrar un espacio interactivo para entender cómo funciona el acero. No necesitas ser experto. Solo imagina que el acero es una receta de cocina: si le pones más ingredientes (como el Carbono) o si lo horneas diferente (Tratamientos Térmicos), el resultado final cambia por completo. ¡Explora las pestañas de arriba para descubrirlo!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Aceros Analizados", len(aceros))
    col2.metric("Nombres Comerciales", aceros["SAE Grade"].nunique())
    col3.metric("Tipos de Horneado", aceros["Condition_simple"].nunique())

    st.markdown("""
    <div style="margin-top: 30px;">
        <h3>🛠️ El Resumen: Lo que debes saber del acero</h3>
        <div class="grid-conclusion">
            <div class="card-conclusion-item">
                <strong>🧂 El Ingrediente Estrella (Carbono):</strong>
                <p>Si le pones más carbono, el acero se vuelve durísimo y muy fuerte. Es perfecto para hacer cuchillos o piezas que raspan contra otras cosas todo el día.</p>
            </div>
            <div class="card-conclusion-item">
                <strong>🛀 Enfriarlo Lento (Recocido):</strong>
                <p>Si lo calientas y lo dejas enfriar suaaavemente, el metal se "relaja". Se vuelve como plastilina gruesa: súper fácil de doblar, cortar y soldar.</p>
            </div>
            <div class="card-conclusion-item">
                <strong>💪 Estirarlo a la Fuerza (En Frío):</strong>
                <p>Si lo aplastas y lo estiras sin calentarlo, el metal se vuelve "tenso". Esto hace que soporte muchísimo peso sin doblarse, ideal para las vigas de un edificio.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2 — EXPLORACIÓN ──
with tab_exploracion:
    st.markdown("""
    <div class="intro-seccion">
        <h4>📊 La Química del Acero</h4>
        <p>En esta sección vas a encontrar cómo reacciona el acero cada vez que le agregas más Carbono. Podrás ver con tus propios ojos si se hace más fuerte, más duro, o si pierde flexibilidad.</p>
    </div>
    """, unsafe_allow_html=True)

    prop_c = st.selectbox("¿Qué característica quieres comparar contra el Carbono?", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    df_carb["Tratamiento (Horneado)"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        👀 <strong>¿Cómo se lee esto?</strong><br>
        • Cada puntito de color es un trozo de acero real. <br>
        • El eje de abajo nos dice cuánto Carbono tiene. El eje de lado nos dice cuánta <strong>{prop_c}</strong> logró alcanzar.<br>
        • Si los puntitos van hacia arriba en diagonal, ¡el Carbono le ayuda! Si van hacia abajo, ¡el Carbono lo empeora!
    </div>
    """, unsafe_allow_html=True)

    fig_c = px.scatter(
        df_carb, x="%C", y=prop_c, 
        color="Tratamiento (Horneado)", 
        hover_data=["SAE Grade"], 
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(aplicar_layout_estetico(fig_c), use_container_width=True, theme=None)


# ── TAB 3 — TEMPERATURA ──
with tab_temp:
    st.markdown("""
    <div class="intro-seccion">
        <h4>🔥 El Efecto del Horno</h4>
        <p>Aquí vas a encontrar qué pasa exactamente cuando subimos la temperatura del fuego. Calentar el acero hace que sus átomos se acomoden mejor y liberen estrés. ¡Mira cómo cambia su comportamiento según los grados de calor!</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_t = st.selectbox("Elige la característica que quieres ver reaccionar al calor:", PROPIEDADES_MECANICAS, key="sb_p_t")
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        👀 <strong>¿Cómo se lee esto?</strong><br>
        • Abajo están los grados de temperatura del horno. A los lados, el nivel de <strong>{prop_t}</strong>.<br>
        • Los colores te dicen si tiene mucho carbono (colores vivos) o poquito (colores claros).
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔵 Cuando lo dejamos enfriar lentamente (Recocido)")
    df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
    if not df_ann.empty:
        fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Tealgrn")
        st.plotly_chart(aplicar_layout_estetico(fig_a), use_container_width=True, theme=None)
    else:
        st.warning("No hay suficientes datos registrados de temperatura para mostrarte esta gráfica.")

    st.subheader("🟣 Cuando lo sacamos a enfriar al aire (Normalizado)")
    df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
    if not df_norm.empty:
        fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Purp")
        st.plotly_chart(aplicar_layout_estetico(fig_n), use_container_width=True, theme=None)
    else:
        st.warning("No hay suficientes datos registrados de temperatura para mostrarte esta gráfica.")


# ── TAB 4 — ¿QUÉ INFLUYE MÁS? (NUEVO ANOVA) ──
with tab_anova:
    st.markdown("""
    <div class="intro-seccion">
        <h4>⚖️ ¿Quién es el verdadero culpable?</h4>
        <p>En esta sección vas a encontrar la respuesta a un gran misterio: Si una característica del metal cambia mucho... ¿Fue por culpa de los ingredientes (Carbono) o por culpa de cómo lo horneamos (Tratamiento
