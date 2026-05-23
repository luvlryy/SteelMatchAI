# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI (Versión Definitiva)
"""

import re
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
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="SteelMatch AI | Quantum Materials",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",   
    plot_bgcolor="rgba(15,23,42,0.4)", 
    font=dict(color="#ffffff", family="sans-serif", size=14),
    title_font=dict(color="#06b6d4", size=20, family="sans-serif"), 
    legend=dict(
        bgcolor="rgba(2, 6, 23, 0.9)", 
        bordercolor="#06b6d4", 
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
        font=dict(color="#ffffff")
    ),
    margin=dict(t=80, b=80, l=50, r=40) 
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

DESCRIPCIONES_TRATAMIENTOS = {
    "Annealed": "Calentamiento seguido de un enfriamiento lento y controlado dentro del horno. Alivia tensiones internas, incrementa la ductilidad y facilita los procesos de mecanizado.",
    "Normalized": "Calentamiento por encima de la temperatura crítica seguido de un enfriamiento al aire libre. Homogeneiza el tamaño de grano de la microestructura y refina las propiedades mecánicas.",
    "Hot Rolled": "Proceso de deformación mecánica realizado a altas temperaturas (por encima de la temperatura de recristalización). Reduce tensiones de fluencia estructural.",
    "Cold Drawn": "Estirado o deformación mecánica en frío a temperatura ambiente. Incrementa notoriamente el límite elástico (YS) y la dureza por acritud mecánica.",
    "Quenched": "Enfriamiento drástico y veloz en agua o aceite desde la fase austenítica. Transforma la estructura en martensita, maximizando la dureza superficial a costa de la fragilidad.",
    "Tempered": "Tratamiento térmico secundario aplicado posterior al temple. Reduce la fragilidad extrema y estabiliza las tensiones moleculares, restaurando la tenacidad general.",
    "As Rolled": "Estado comercial en bruto tras pasar por los rodillos de laminación directa, sin tratamientos térmicos de reacondicionamiento estructural posteriores.",
    "Other": "Tratamientos metalúrgicos de carácter compuesto o variaciones específicas registradas de forma atípica."
}


# ==========================================================
# ESTILO VISUAL — TEMÁTICA FUTURISTA Y ALTA VISIBILIDAD
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');

    .stApp, [data-testid="stSidebar"] {
        background-color: #020617 !important;
        background-image: 
            linear-gradient(rgba(14, 165, 233, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(14, 165, 233, 0.05) 1px, transparent 1px) !important;
        background-size: 50px 50px !important;
        background-position: -1px -1px !important;
        background-attachment: fixed !important;
    }
    
    [data-testid="stSidebar"] { border-right: 1px solid rgba(6, 182, 212, 0.3) !important; }

    html, body, [class*="css"], p, span, label, li {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important; 
    }
    
    p { font-size: 15px !important; line-height: 1.6 !important; }

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
        color: #06b6d4 !important; 
        font-weight: 700 !important;
        border-left: 5px solid #06b6d4;
        padding-left: 15px;
        margin-top: 2.5rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    h3, h4 { color: #ffffff !important; font-weight: 600 !important; }
    
    /* Personalización de subtítulos Markdown */
    .stMarkdown h3 { color: #06b6d4 !important; font-size: 1.5rem !important; margin-top: 20px !important;}

    div[data-baseweb="select"] > div, .stNumberInput input, .stSlider > div {
        background-color: rgba(15, 23, 42, 0.9) !important; 
        border: 1px solid #06b6d4 !important; 
        border-radius: 8px;
        box-shadow: 0 0 5px rgba(6, 182, 212, 0.2);
    }
    
    div[data-baseweb="select"] > div * { color: #ffffff !important; font-weight: 600; }

    div[data-baseweb="popover"] div[role="listbox"],
    div[data-baseweb="popover"] ul[role="listbox"] {
        background-color: #f8fafc !important; 
        border: 2px solid #06b6d4 !important;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.4) !important;
    }
    
    div[data-baseweb="popover"] li[role="option"] {
        background-color: transparent !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }

    div[data-baseweb="popover"] li[role="option"] * { color: #000000 !important; font-weight: 700 !important; }

    div[data-baseweb="popover"] li[role="option"]:hover,
    div[data-baseweb="popover"] li[role="option"][aria-selected="true"] { background-color: #06b6d4 !important; }
    
    div[data-baseweb="popover"] li[role="option"]:hover *,
    div[data-baseweb="popover"] li[role="option"][aria-selected="true"] * { color: #ffffff !important; }

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

    .intro-seccion {
        background: rgba(15, 23, 42, 0.6);
        border: 2px solid #06b6d4;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    
    .intro-seccion h2 { margin-top: 0 !important; border: none; padding-left: 0; color: #34d399 !important; font-size: 1.8rem !important; text-transform: none;}

    .leyenda-grafica {
        background: rgba(2, 6, 23, 0.8);
        border: 1px solid #34d399; 
        border-left: 5px solid #34d399;
        padding: 15px 20px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 10px;
        margin-bottom: 25px;
        font-size: 15px;
        line-height: 1.6;
        color: #ffffff !important;
    }
    .leyenda-grafica strong { color: #34d399 !important; }
    
    .conclusion-box {
        background: rgba(167, 139, 250, 0.1);
        border-left: 5px solid #a78bfa;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 30px;
        border-radius: 4px 8px 8px 4px;
    }

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
    fig.update_xaxes(gridcolor="rgba(6, 182, 212, 0.1)", zerolinecolor="rgba(6, 182, 212, 0.3)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(6, 182, 212, 0.1)", zerolinecolor="rgba(6, 182, 212, 0.3)", showgrid=True)
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
# MENÚ LATERAL: GLOSARIO DESTACADO Y CONFIGURACIÓN
# ==========================================================

aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

# Glosario ahora es altamente visible al tope de la barra lateral
st.sidebar.markdown("### 📖 Glosario Rápido")
st.sidebar.info("""
**💪 UTS (Fuerza Máxima):** Cuánta fuerza bruta soporta antes de partirse en dos.  
**📐 YS (Límite Elástico):** Cuánto peso aguanta antes de quedarse deformado para siempre.  
**💎 Dureza (HB):** Qué tan difícil es rayarlo o abollarlo.  
**🎗️ Elongación (%):** Qué tanto se estira como liga antes de romperse.
""")

st.sidebar.markdown("---")
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


# ==========================================================
# ESTRUCTURA DE PESTAÑAS (Módulos Integrados)
# ==========================================================

st.title("SteelMatch AI // Inteligencia Analítica de Aceros")

tab_inicio, tab_inventario, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Bienvenida", "📋 Inventario Base", "📊 Matriz Química", "🔥 Protocolo Térmico", "⚖️ ANOVA", "🔍 Requisitor"
])


# ── TAB 1 — INICIO (Módulo Central Mejorado) ──
with tab_inicio:
    st.markdown("""
    <div class="intro-seccion">
        <h2>👋 ¡Bienvenido a SteelMatch AI! Tu simulador metalúrgico de bolsillo.</h2>
        <p>Descubre el increíble mundo de los aceros al carbono sin necesidad de ser un experto en ciencia de materiales. Imagina que el acero es como la receta de un pastel: si cambias los ingredientes principales (como el % de Carbono) o cambias la forma de hornearlo (los Tratamientos Térmicos), el resultado final tendrá "superpoderes" físicos completamente diferentes. ¡Usa las pestañas superiores para explorar los datos reales!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 ¿Qué hicimos? (Objetivos y Metodología)")
    st.write("Desarrollamos una plataforma interactiva que actúa como un microscopio de datos. Tomamos un gran volumen de información real sobre distintas aleaciones comerciales de acero SAE y las procesamos usando lenguajes de programación. Limpiamos, agrupamos y modelamos matemáticamente la información para generar visualizaciones interactivas que revelan el comportamiento oculto del material.")
    
    st.markdown("### 💡 ¿Por qué lo hicimos? (Hipótesis y Razonamiento)")
    st.write("Porque comprender los materiales es vital para construir el mundo moderno (desde el chasis de un auto hasta la estructura de un puente). Nuestra hipótesis principal sostiene que **el porcentaje de Carbono (%C) y el protocolo de horneado (Tratamiento Térmico) son los factores determinantes** absolutos sobre la resistencia (UTS), límite elástico (YS), dureza (HB) y flexibilidad (elongación) del metal. Creamos este panel para probarlo con datos empíricos.")
    
    st.markdown("### 📊 ¿Qué encontramos? (Conclusiones Destacadas)")
    st.markdown("""
    * **El Carbono da fuerza bruta, pero roba flexibilidad:** Observamos una clara tendencia donde el aumento de carbono dispara exponencialmente la Dureza y el UTS, pero hace caer en picada la Elongación (haciendo el material más frágil y menos elástico).
    * **El horno transforma el metal:** Confirmamos que protocolos como el "Recocido" relajan las tensiones del metal maximizando su capacidad de estiramiento, mientras que métodos más agresivos como el "Templado" lo vuelven extremadamente duro para resistir el desgaste.
    * **El calor debilita a favor de la tenacidad:** Encontramos que aumentar la temperatura en tratamientos específicos tiende a reducir ligeramente la fuerza estructural, pero a cambio estabiliza el material para que no se quiebre de forma sorpresiva.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Aceros SAE Analizados", len(aceros))
    col2.metric("Grados Comerciales Identificados", aceros["SAE Grade"].nunique())
    col3.metric("Protocolos Térmicos", aceros["Condition_simple"].nunique())


# ── TAB 2 — INVENTARIO BASE (Sección 3 de la Guía) ──
with tab_inventario:
    st.markdown("""
    <div class="intro-seccion">
        <h4>📋 Exploración Inicial e Identificación de Variables</h4>
        <p>Aquí listamos exactamente qué contiene la base de datos que estamos analizando. Conoce la variedad de grados SAE comerciales disponibles y cómo clasificamos los tratamientos aplicados a las muestras.</p>
    </div>
    """, unsafe_allow_html=True)

    c_inv1, c_inv2 = st.columns(2)
    
    with c_inv1:
        st.subheader("Lista de Grados de Aceros Disponibles")
        df_sae_counts = aceros["SAE Grade"].value_counts().reset_index()
        df_sae_counts.columns = ["Grado comercial (SAE Grade)", "Número de registros"]
        st.dataframe(df_sae_counts, use_container_width=True, hide_index=True)
        
    with c_inv2:
        st.subheader("Clasificación de Tratamientos")
        df_treat_counts = aceros["Condition_simple"].value_counts().reset_index()
        df_treat_counts.columns = ["Tratamiento", "Número de registros"]
        df_treat_counts["Descripción breve (Fundamento Técnico)"] = df_treat_counts["Tratamiento"].map(DESCRIPCIONES_TRATAMIENTOS)
        st.dataframe(df_treat_counts, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="conclusion-box">
        <strong>💡 Conclusión del Inventario:</strong>
        <p style="margin-top: 5px;">Al observar las tablas, podemos concluir que la base de datos está dominada principalmente por tratamientos comerciales básicos como el "Laminado" (Hot/Cold Rolled) y procesos de estandarización como el "Normalizado". Esto significa que la mayor parte de nuestra data refleja las condiciones en las que el acero se vende comúnmente en la industria primaria, antes de recibir tratamientos térmicos especializados y costosos como el Templado (Quenched).</p>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 3 — EXPLORACIÓN (Matriz Química y Boxplots) ──
with tab_exploracion:
    st.markdown("""
    <div class="intro-seccion">
        <h4>📊 Análisis Químico vs Físico</h4>
        <p>Descubre el poder del ingrediente principal: El Carbono. En este módulo comprobaremos visualmente si más Carbono equivale a un mejor material o si existe un costo oculto en esa aleación.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔬 Curva de Tendencia Multivariable Integrada")
    df_multivar = aceros.dropna(subset=["%C"] + PROPIEDADES_MECANICAS).copy()
    df_multivar["Elongation (%) x 10"] = df_multivar["Elongation (%)"] * 10
    
    st.markdown("""
    <div class="leyenda-grafica">
        📈 <strong>Cómo leer esta gráfica:</strong><br>
        • El eje inferior (X) muestra el porcentaje de Carbono en la receta.<br>
        • Para poder graficar todo junto, el valor original de "Elongación" fue <strong>multiplicado por 10</strong>.<br>
        • Cada color representa una propiedad distinta peleando por el espacio.
    </div>
    """, unsafe_allow_html=True)
    
    fig_multi = go.Figure()
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["UTS (MPa)"], mode='markers', name='UTS (MPa)', marker=dict(color='#34d399')))
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["YS (MPa)"], mode='markers', name='YS (MPa)', marker=dict(color='#06b6d4')))
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["Hardness (HB)"], mode='markers', name='Hardness (HB)', marker=dict(color='#a78bfa')))
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["Elongation (%) x 10"], mode='markers', name='Elongation (%) x 10', marker=dict(color='#f43f5e')))
    
    fig_multi.update_layout(title="Comportamiento Mecánico Simultáneo vs % de Carbono")
    fig_multi.update_xaxes(title_text="Concentración de Carbono (%C)")
    fig_multi.update_yaxes(title_text="Magnitud de la Propiedad")
    st.plotly_chart(aplicar_layout_estetico(fig_multi), use_container_width=True, theme=None)

    st.markdown("""
    <div class="conclusion-box">
        <strong>💡 Conclusión Multivariable:</strong>
        <p style="margin-top: 5px;">Esta gráfica demuestra el "trade-off" (intercambio) fundamental de la metalurgia: A medida que avanzamos hacia la derecha (más carbono), las nubes de puntos de Resistencia (UTS) y Límite Elástico (YS) suben radicalmente. Sin embargo, la nube roja (Elongación) va hacia abajo. <strong>Conclusión:</strong> El carbono te da un metal que soporta mucho más peso antes de deformarse, pero sacrifica por completo su capacidad de estirarse (se vuelve frágil como el vidrio frente a impactos repentinos).</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("🎯 Análisis Individual: Propiedad vs Carbono por Tratamiento")
    prop_c = st.selectbox("Selecciona la propiedad física a aislar:", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    df_carb["Protocolo Térmico"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        👀 <strong>Guía para tontos - Interpretación de Puntos y Líneas:</strong><br>
        • <strong>Los Puntos (Círculos):</strong> Cada circulito es una pieza real de acero. Su color indica qué tratamiento se le dio.<br>
        • <strong>Las Líneas Rectas:</strong> Son las líneas de tendencia matemática. Si la línea va "cuesta arriba", significa que mientras más carbono eches a la mezcla, el número de <strong>{prop_c}</strong> va a subir. Si la línea va "cuesta abajo", significa que el carbono arruina o disminuye esa propiedad.
    </div>
    """, unsafe_allow_html=True)

    fig_c = px.scatter(
        df_carb, x="%C", y=prop_c, 
        color="Protocolo Térmico", 
        hover_data=["SAE Grade"], 
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Alphabet 
    )
    st.plotly_chart(aplicar_layout_estetico(fig_c), use_container_width=True, theme=None)

    st.markdown("---")
    st.subheader(f"📦 Distribución y Dispersión Estadística de {prop_c} por Tratamiento")
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        📦 <strong>Guía para entender el Gráfico de Cajas (Boxplot):</strong><br>
        • <strong>La Caja de color:</strong> Representa dónde se agrupa el 50% central de todos los aceros probados. La línea que la atraviesa es la media (el valor más común).<br>
        • <strong>Los "Bigotes" (Líneas T):</strong> Muestran hasta dónde llegan los valores normales, tanto los más altos como los más bajos.<br>
        • <strong>Cajas largas vs cortas:</strong> Una caja muy larga significa que ese tratamiento da resultados muy locos y variables. Una caja cortita significa que el tratamiento es súper preciso y consistente.
    </div>
    """, unsafe_allow_html=True)

    fig_box = px.box(
        df_carb, x="Protocolo Térmico", y=prop_c,
        color="Protocolo Térmico",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        points="all"
    )
    # OCULTAMOS LA LEYENDA DEL BOXPLOT PARA QUE NO CHOQUE CON EL EJE X
    fig_box.update_layout(title=f"Boxplot: ¿Qué tratamiento garantiza mejor {prop_c}?", showlegend=False, xaxis_title="Categoría de Protocolo Térmico")
    st.plotly_chart(aplicar_layout_estetico(fig_box), use_container_width=True, theme=None)

    st.markdown(f"""
    <div class="conclusion-box">
        <strong>💡 Conclusión del Análisis Individual:</strong>
        <p style="margin-top: 5px;">Al analizar los puntos de dispersión y las cajas, queda en evidencia que el tratamiento térmico "mueve" las gráficas enteras hacia arriba o hacia abajo, sin importar tanto el carbono. Por ejemplo, los procesos trabajados "en frío" o con temple tienden a estar siempre en la parte más alta si hablamos de Dureza y Resistencia, demostrando que la forma física en que forzamos o enfriamos el metal altera su microestructura cristalina para hacerlo más rígido que su estado natural.</p>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 4 — TEMPERATURA (Protocolo Térmico) ──
with tab_temp:
    st.markdown("""
    <div class="intro-seccion">
        <h4>🔥 Módulo de Influencia Térmica // Impacto de los Grados Celsius</h4>
        <p>Aquí analizamos exclusivamente los tratamientos que reportan una temperatura exacta en el horno (Recocido y Normalizado). ¿Qué le pasa al metal cuando le subes el fuego al máximo?</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_t = st.selectbox("Output físico a evaluar reaccionando al calor:", PROPIEDADES_MECANICAS, key="sb_p_t")
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        🌡️ <strong>Guía de Interpretación Térmica:</strong><br>
        • <strong>El eje inferior (X):</strong> Representa los grados Centígrados del horno.<br>
        • <strong>El color del punto:</strong> En esta gráfica los colores no son tratamientos, ¡son el nivel de Carbono! Amarillo/Brillante significa que tiene mucho carbono, Oscuro/Morado significa bajo carbono.
    </div>
    """, unsafe_allow_html=True)

    c_temp1, c_temp2 = st.columns(2)
    
    with c_temp1:
        st.subheader("🔵 Recocido (Enfriado Lento)")
        df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
        if not df_ann.empty:
            fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Viridis", title=f"Recocido: Temp vs {prop_t}")
            st.plotly_chart(aplicar_layout_estetico(fig_a), use_container_width=True, theme=None)
        else:
            st.warning("Sin datos para recocido.")

    with c_temp2:
        st.subheader("🟣 Normalizado (Enfriado al Aire)")
        df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
        if not df_norm.empty:
            fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Plasma", title=f"Normalizado: Temp vs {prop_t}")
            st.plotly_chart(aplicar_layout_estetico(fig_n), use_container_width=True, theme=None)
        else:
            st.warning("Sin datos para normalizado.")

    st.markdown("""
    <div class="conclusion-box">
        <strong>💡 Conclusión Térmica:</strong>
        <p style="margin-top: 5px;">Al analizar ambos gráficos térmicos, notamos que a medida que la temperatura de austenitización (calentamiento en horno) aumenta, hay una leve tendencia a que caigan la dureza y resistencia, y aumente la homogeneidad. En el gráfico de Normalizado solemos ver "columnas" o líneas verticales de puntos. Esto ocurre porque la industria utiliza temperaturas estandarizadas redondas (como 870°C o 900°C) para procesar tandas masivas de acero, lo que agrupa múltiples muestras químicas diferentes bajo una sola temperatura de receta comercial.</p>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 5 — ¿QUÉ INFLUYE MÁS? (Control de Variables // ANOVA) ──
with tab_anova:
    st.markdown("""
    <div class="intro-seccion">
        <h4>⚖️ Algoritmo de Atribución de Variables // Juicio ANOVA</h4>
        <p>Resolución de la gran duda: Cuando una propiedad física cambia... ¿De quién es la culpa? El Análisis de Varianza (ANOVA) es un juez estadístico que evalúa qué porcentaje del cambio se debe al químico (Carbono) y qué porcentaje se debe a cómo lo horneamos (Tratamiento).</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_anova = st.selectbox("Selecciona la propiedad a enjuiciar:", PROPIEDADES_MECANICAS, key="sb_p_a")
    
    df_anova = aceros.dropna(subset=["%C", "Condition_simple", prop_anova]).copy()
    if len(df_anova) > 10:
        df_anova = df_anova.rename(columns={"%C": "Carbono", prop_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        mod = ols("Propiedad ~ Carbono + C(Tratamiento)", data=df_anova).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Influencia_%"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        
        pct_carbono = tab_a.loc["Carbono", "Influencia_%"]
        pct_tratamiento = tab_a.loc["C(Tratamiento)", "Influencia_%"]
        pct_otros = tab_a.loc["Residual", "Influencia_%"]
        
        st.subheader(f"Vector de Culpabilidad para: {prop_anova}")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Factor 1: Química<br>(% Carbono)", "Factor 2: Protocolo<br>(Tratamiento Térmico)", "Factor 3: Residual<br>(Otras variables)"],
            values=[pct_carbono, pct_tratamiento, pct_otros],
            hole=.6,
            marker_colors=["#06b6d4", "#a78bfa", "#334155"], 
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

        st.markdown("<div class='conclusion-box' style='border-left-color: #34d399; background: rgba(52, 211, 153, 0.1);'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Veredicto Final del Juez Estadístico")
        if pct_carbono > pct_tratamiento:
            st.markdown(f"¡El veredicto es contundente! Para modificar **{prop_anova}**, **la receta química es la dueña del balón**. El porcentaje de Carbono tiene un impacto dominante (representa el **{pct_carbono:.1f}%** de la influencia total). Si quieres cambiar este atributo en la vida real, gastarás menos dinero cambiando la fórmula de carbono desde el principio, que tratando de arreglarlo metiéndolo al horno con tratamientos térmicos.")
        else:
            st.markdown(f"¡Sorpresa en el jurado! Para modificar **{prop_anova}**, **el horno es quien manda**. La cantidad de carbono inicial pasa a segundo plano. Lo que realmente altera el material (con un **{pct_tratamiento:.1f}%** de impacto total) es el estrés térmico o mecánico que apliques: si lo dejas enfriar lentamente o lo sumerges bruscamente en agua fría. ¡Ajusta los parámetros de procesamiento en fábrica!")
        st.markdown("</div>", unsafe_allow_html=True)


# ── TAB 6 — RECOMENDADOR INTELIGENTE (Requisitor SAE) ──
with tab_recomendador:
    st.markdown("""
    <div class="intro-seccion">
        <h4>🔍 Requisitor Mágico de Muestras SAE // Matching Artificial</h4>
        <p>Iniciando protocolo de emparejamiento. Mueve las barras laterales de la izquierda definiendo tu "acero soñado" (cuánto peso quieres que aguante, cuánta dureza necesitas, etc.). La Inteligencia Artificial cruzará tus requerimientos contra toda la base de datos real para encontrar el grado comercial exacto que deberías comprar.</p>
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
                st.warning("❌ Matching FAILED. No se encontraron muestras que coincidan con el protocolo térmico especificado. Reajusta los parámetros de búsqueda en la barra izquierda.")
            else:
                usr_in = np.array([[carbono, uts, ys, dureza, elongacion]])
                usr_norm = scaler.transform(usr_in)
                mat_norm = scaler.transform(df_res[COLUMNAS_MODELO])

                dist = euclidean_distances(usr_norm, mat_norm)
                idx = np.argmin(dist)
                match_mat = df_res.iloc[idx]
                pct_sim = 100 / (1 + dist[0][idx])

                st.success("🎉 ¡Matching SUCCESSFUL! Hemos localizado tu acero comercial ideal.")
                
                st.subheader("🏆 Especificación SAE Recomendada para tu Proyecto:")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Grado Comercial Identificado:", f"SAE {match_mat['SAE Grade']}")
                    st.metric("Debe venir con Tratamiento:", TRADUCCIONES_TRATAMIENTOS.get(match_mat['Condition_simple'], match_mat['Condition_simple']))
                with c2:
                    st.metric("Nivel de Compatibilidad (Match):", f"{pct_sim:.1f}%")
                    st.metric("Agente Químico Real (%C):", f"{match_mat['%C']:.3f} %")
                with c3:
                    st.metric("Soporte UTS Real:", f"{match_mat['UTS (MPa)']} MPa")
                    st.metric("Dureza Real (HB):", f"{match_mat['Hardness (HB)']} HB")
