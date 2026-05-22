# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
Versión final consolidada con todas las mejoras de usabilidad:
  - Conclusión general reubicada en la parte superior para máxima visibilidad.
  - Estructura de conclusión enriquecida con viñetas metalúrgicas explícitas.
  - Recomendador intuitivo con perfiles de uso (Presets) en lenguaje cotidiano.
  - Traducción completa de tratamientos térmicos en la interfaz del usuario.
  - Reglas clave de cierre explicadas con lógica de causa y efecto.
  - Tema oscuro consistente en todas las gráficas Plotly.
  - Extracción de temperatura más robusta (evita capturar SAE Grades).
  - ANOVA con aviso de R² bajo y estadísticas descriptivas en expanders.
  - Estética mejorada con CSS personalizado.

Cómo ejecutar:
1. pip install streamlit pandas numpy scikit-learn plotly statsmodels openpyxl xlrd
2. Colocar este archivo junto al CSV/Excel del dataset o subirlo desde la app
3. streamlit run app.py
"""

# ==========================================================
# IMPORTACIONES
# ==========================================================

import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

import statsmodels.api as sm
from statsmodels.formula.api import ols


# ==========================================================
# CONFIGURACIÓN GENERAL Y DICCIONARIOS
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

# Diccionario global para traducir las condiciones del dataset a la interfaz en español
TRADUCCION_TRATAMIENTOS = {
    "Todos": "Todos los tratamientos",
    "Annealed": "Recocido (Annealed)",
    "Normalized": "Normalizado (Normalized)",
    "Hot Rolled": "Laminado en caliente (Hot Rolled)",
    "Cold Drawn": "Estirado en frío (Cold Drawn)",
    "Quenched": "Templado (Quenched)",
    "Tempered": "Revenido (Tempered)",
    "As Rolled": "Estado de laminación (As Rolled)",
    "Other": "Otros procesos"
}

# Paleta oscura global para todas las gráficas Plotly
PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(15,23,42,0)",   # fondo transparente
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#e2e8f0", family="monospace"),
    title_font=dict(color="#38bdf8", size=18),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#334155", borderwidth=1),
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
# ESTILO VISUAL — CSS PERSONALIZADO
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght=300;400;600;700&family=Syne:wght=400;600;800&display=swap');

    /* Fondo general */
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

    /* Títulos */
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
        margin-top: 2rem !important;
    }

    h3 {
        font-family: 'Syne', sans-serif !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }

    /* Botones */
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
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(56,189,248,0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(56,189,248,0.5);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }

    /* Métricas */
    [data-testid="metric-container"] {
        background: rgba(56,189,248,0.06);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 14px;
        padding: 16px;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        border-color: rgba(56,189,248,0.5);
        background: rgba(56,189,248,0.1);
    }

    /* Cards de hallazgos */
    .hallazgo-card {
        background: rgba(56,189,248,0.06);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 10px 0;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
        animation: fadeInUp 0.5s ease forwards;
    }

    .hallazgo-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56,189,248,0.5);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Badges */
    .badge-dominante {
        display: inline-block;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 14px;
        padding: 6px 16px;
        border-radius: 50px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(56,189,248,0.4);
    }

    .badge-secundario {
        display: inline-block;
        background: rgba(100,116,139,0.3);
        color: #94a3b8;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 13px;
        padding: 5px 14px;
        border-radius: 50px;
        border: 1px solid rgba(100,116,139,0.3);
    }

    /* Separador */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
        margin: 2rem 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,23,42,0.8);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        color: #64748b;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
    }

    /* Contenedor de conclusiones y alertas */
    .conclusion-box {
        background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(99,102,241,0.08));
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        animation: fadeInUp 0.6s ease forwards;
    }

    .conclusion-box ul {
        margin-top: 10px;
        margin-bottom: 0px;
        padding-left: 20px;
    }

    .conclusion-box li {
        margin-bottom: 8px;
    }

    .prop-badge {
        display: inline-block;
        background: rgba(56,189,248,0.15);
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 13px;
        padding: 4px 14px;
        border-radius: 8px;
        border: 1px solid rgba(56,189,248,0.3);
        margin-bottom: 10px;
    }

    /* Correcciones globales de visibilidad de texto */
    p, span, label, div {
        color: #e2e8f0 !important;
    }

    .stSelectbox label {
        color: #f8fafc !important;
    }

    input, textarea {
        color: #ffffff !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1 !important;
    }

    .stTabs [aria-selected="true"] {
        color: white !important;
    }

    [data-testid="stDataFrame"] {
        color: white !important;
    }

    [data-testid="metric-container"] * {
        color: #f8fafc !important;
    }

    .streamlit-expanderHeader {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    small {
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# FUNCIONES DE LIMPIEZA Y PREPROCESAMIENTO
# ==========================================================

def agrupar_tratamiento(texto):
    texto = str(texto).lower()
    if "annealed" in texto:
        return "Annealed"
    elif "normalized" in texto:
        return "Normalized"
    elif "hot rolled" in texto or "hot-rolled" in texto:
        return "Hot Rolled"
    elif "cold drawn" in texto or "cold-drawn" in texto:
        return "Cold Drawn"
    elif "quenched" in texto:
        return "Quenched"
    elif "tempered" in texto:
        return "Tempered"
    elif "as rolled" in texto:
        return "As Rolled"
    else:
        return "Other"


def extraer_temperatura(condicion):
    condicion = str(condicion)
    coincidencia = re.search(r"(\d+(?:\.\d+)?)\s*°\s*[Cc]", condicion)
    if coincidencia:
        temp = float(coincidencia.group(1))
        if 100 <= temp <= 1300:
            return temp
        return np.nan

    if re.search(r"(?:annealed|normalized)\s+at\s+(\d{3,4}(?:\.\d+)?)", condicion, flags=re.IGNORECASE):
        coincidencia_respaldo = re.search(
            r"(?:annealed|normalized)\s+at\s+(\d{3,4}(?:\.\d+)?)",
            condicion, flags=re.IGNORECASE
        )
        if coincidencia_respaldo:
            temp = float(coincidencia_respaldo.group(1))
            if 100 <= temp <= 1300:
                return temp
    return np.nan


def aplicar_layout_plotly(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(56,189,248,0.2)")
    fig.update_yaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(56,189,248,0.2)")
    return fig


def leer_archivo_datos(nombre_archivo, archivo_bytes=None):
    extension = nombre_archivo.lower().split(".")[-1]
    def obtener_fuente():
        if archivo_bytes is not None:
            return io.BytesIO(archivo_bytes)
        return nombre_archivo

    if extension == "csv":
        for encoding in ["latin1", "utf-8", "utf-8-sig"]:
            try:
                return pd.read_csv(obtener_fuente(), sep=",", encoding=encoding)
            except Exception:
                continue
        st.error("No se pudo leer el CSV. Revisa que el archivo no esté dañado.")
        st.stop()
    elif extension in ["xlsx", "xls"]:
        try:
            return pd.read_excel(obtener_fuente())
        except Exception as e:
            st.error(f"No se pudo leer el archivo de Excel: {e}")
            st.stop()
    else:
        st.error("Formato no compatible. Usa un archivo .csv, .xlsx o .xls.")
        st.stop()


@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo, archivo_bytes=None):
    try:
        aceros = leer_archivo_datos(nombre_archivo, archivo_bytes)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{nombre_archivo}'.")
        st.stop()

    aceros.columns = aceros.columns.astype(str).str.strip()
    aceros["SAE Grade"] = aceros["SAE Grade"].astype(str).str.strip()

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
# CARGA DE DATOS (SIDEBAR)
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Archivo de datos")

archivo_subido = st.sidebar.file_uploader("Sube el dataset en CSV o Excel", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    aceros = cargar_y_preprocesar_datos(archivo_subido.name, archivo_subido.getvalue())
    st.sidebar.success(f"Archivo cargado: {archivo_subido.name}")
else:
    aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)
    st.sidebar.info(f"Usando archivo local: {DEFAULT_DATA_FILE}.")


# ==========================================================
# TÍTULO PRINCIPAL Y DESCRIPCIÓN
# ==========================================================

st.title("⚙️ SteelMatch AI")
st.subheader("Análisis de propiedades mecánicas en aceros al carbono")

st.write("""
Esta web analiza cómo el porcentaje de carbono, el tratamiento térmico o mecánico
y la temperatura del tratamiento influyen en las propiedades mecánicas de aceros al carbono.
""")


# ==========================================================
# MEJORA: SECCIÓN DE CONCLUSIÓN GENERAL (UBICACIÓN SUPERIOR)
# ==========================================================

st.markdown("""
<div class="conclusion-box">
    <p style="line-height:1.8; font-size:15px; margin-bottom: 5px;">
        <strong style="color:#38bdf8; font-size:17px;">💡 Conclusión General:</strong> 
        El comportamiento mecánico de los aceros al carbono depende de un balance crítico entre su composición química (%C) y sus procesos térmicos o mecánicos.
    </p>
    <ul style="line-height:1.7; font-size:14px;">
        <li><strong style="color:#38bdf8;">Efecto del Carbono (%C):</strong> A mayor contenido de carbono &rarr; <strong>Aumenta</strong> la resistencia mecánica (UTS, YS) y la dureza (HB), pero <strong>disminuye</strong> notablemente la elongación (ductilidad). El carbono actúa bloqueando los planos de deslizamiento microestructurales.</li>
        <li><strong style="color:#818cf8;">Efecto de los Tratamientos:</strong>
            <ul>
                <li><strong style="color:#e2e8f0;">Annealed (Recocido):</strong> <strong>Disminuye</strong> la resistencia y la dureza para <strong>maximizar</strong> la ductilidad, facilitando procesos posteriores de maquinado o formado.</li>
                <li><strong style="color:#e2e8f0;">Normalized (Normalizado):</strong> Refina el tamaño de grano, <strong>aumentando</strong> la resistencia y dureza de manera uniforme en comparación con el recocido.</li>
                <li><strong style="color:#e2e8f0;">Quenched (Templado):</strong> <strong>Maximiza</strong> la dureza al transformar la estructura en martensita, requiriendo un revenido posterior para reducir la fragilidad extrema.</li>
                <li><strong style="color:#e2e8f0;">Hot Rolled / Cold Drawn:</strong> Los procesos mecánicos moldean el material; el trabajo en frío (Cold Drawn) incrementa drásticamente la resistencia por endurecimiento por deformación.</li>
            </ul>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# PESTAÑAS PRINCIPALES
# ==========================================================

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio",
    "📊 Exploración",
    "🌡️ Temperatura",
    "🧪 ANOVA",
    "🔍 Recomendador"
])


# ==========================================================
# TAB 1 — INICIO
# ==========================================================

with tab_inicio:
    st.header("Objetivo del proyecto")

    st.write("""
    El objetivo general es explorar cómo el contenido de carbono (%C) y los tratamientos
    térmicos o mecánicos afectan propiedades como resistencia última, límite elástico,
    dureza y elongación.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros cargados", len(aceros))
    with col2:
        st.metric("Tipos SAE distintos", aceros["SAE Grade"].nunique())
    with col3:
        st.metric("Tratamientos agrupados", aceros["Condition_simple"].nunique())

    st.subheader("Vista previa del dataset limpio")
    st.dataframe(aceros.head(20), use_container_width=True)


# ==========================================================
# TAB 2 — EXPLORACIÓN (CON NOMBRES TRADUCIDOS)
# ==========================================================

with tab_exploracion:
    st.header("Exploración inicial de aceros y tratamientos")

    st.subheader("Tabla de aceros")
    tabla_aceros = aceros["SAE Grade"].value_counts().reset_index()
    tabla_aceros.columns = ["SAE Grade", "Cantidad de datos"]
    st.dataframe(tabla_aceros, use_container_width=True)

    st.subheader("Tabla de tratamientos")
    tabla_tratamientos = aceros["Condition_simple"].value_counts().reset_index()
    tabla_tratamientos.columns = ["Tratamiento Original", "Cantidad de datos"]
    
    # MEJORA: Añadir traducción clara al español de cada tratamiento
    tabla_tratamientos["Nombre en Español"] = tabla_tratamientos["Tratamiento Original"].map(TRADUCCION_TRATAMIENTOS)

    descripciones = {
        "Annealed": "Recocido: suaviza el acero, reduce esfuerzos internos y mejora ductilidad.",
        "Normalized": "Normalizado: calentamiento y enfriamiento al aire para refinar microestructura.",
        "Hot Rolled": "Laminado en caliente: deformación a alta temperatura para dar forma.",
        "Cold Drawn": "Estirado en frío: aumenta resistencia y dureza por deformación plástica.",
        "Quenched": "Temple: enfriamiento rápido para maximizar dureza.",
        "Tempered": "Revenido: posterior al temple, reduce fragilidad y ajusta dureza.",
        "As Rolled": "Estado laminado sin tratamiento térmico adicional.",
        "Other": "Otras condiciones no clasificadas."
    }
    tabla_tratamientos["Descripción técnica"] = tabla_tratamientos["Tratamiento Original"].map(descripciones)
    
    # Reordenar columnas para legibilidad
    tabla_tratamientos = tabla_tratamientos[["Tratamiento Original", "Nombre en Español", "Cantidad de datos", "Descripción técnica"]]
    st.dataframe(tabla_tratamientos, use_container_width=True)

    st.subheader("Propiedades mecánicas vs %C")
    propiedad_carbono = st.selectbox("Selecciona una propiedad para graficar contra %C", PROPIEDADES_MECANICAS, key="propiedad_carbono")

    datos_carbono = aceros.dropna(subset=["%C", propiedad_carbono, "Condition_simple"])

    fig_carbono = px.scatter(
        datos_carbono, x="%C", y=propiedad_carbono, color="Condition_simple",
        hover_data=["SAE Grade", "Conditions"], trendline="ols",
        title=f"{propiedad_carbono} vs %C por tratamiento",
        labels={"%C": "Porcentaje de carbono promedio", propiedad_carbono: propiedad_carbono, "Condition_simple": "Tratamiento"}
    )
    fig_carbono = aplicar_layout_plotly(fig_carbono)
    st.plotly_chart(fig_carbono, use_container_width=True)


# ==========================================================
# TAB 3 — TEMPERATURA
# ==========================================================

with tab_temp:
    st.header("Parte A — Pregunta 3: Influencia de la temperatura")

    propiedad_temp = st.selectbox("Selecciona la propiedad mecánica para analizar contra temperatura", PROPIEDADES_MECANICAS, key="propiedad_temp")

    st.markdown(f'<span class="prop-badge">Analizando: {propiedad_temp}</span>', unsafe_allow_html=True)
    datos_temp = aceros.dropna(subset=["Temp_C", propiedad_temp, "Condition_simple"])

    # --- Annealed ---
    st.subheader("🔵 Recocido (Annealed) — Propiedad mecánica vs temperatura")
    annealed = datos_temp[datos_temp["Condition_simple"] == "Annealed"]

    if not annealed.empty:
        fig_annealed = px.scatter(
            annealed, x="Temp_C", y=propiedad_temp, color="%C",
            hover_data=["SAE Grade", "%C", "Conditions"], trendline="ols",
            title=f"{propiedad_temp} vs Temperatura — Annealed",
            labels={"Temp_C": "Temperatura del tratamiento (°C)", propiedad_temp: propiedad_temp, "%C": "% Carbono"},
            color_continuous_scale="Blues"
        )
        fig_annealed = aplicar_layout_plotly(fig_annealed)
        st.plotly_chart(fig_annealed, use_container_width=True)

        with st.expander("📋 Estadísticas descriptivas — Annealed"):
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            s = annealed[propiedad_temp].dropna()
            col_e1.metric("Media", f"{s.mean():.1f}")
            col_e2.metric("Mín", f"{s.min():.1f}")
            col_e3.metric("Máx", f"{s.max():.1f}")
            col_e4.metric("Registros", len(s))
            st.dataframe(annealed[["SAE Grade", "Conditions", "Temp_C", "%C", propiedad_temp]], use_container_width=True)
    else:
        st.warning("No hay datos de Annealed con temperatura válida.")

    st.markdown("---")

    # --- Normalized ---
    st.subheader("🟣 Normalizado (Normalized) — Propiedad mecánica vs temperatura")
    normalized = datos_temp[datos_temp["Condition_simple"] == "Normalized"]

    if not normalized.empty:
        fig_normalized = px.scatter(
            normalized, x="Temp_C", y=propiedad_temp, color="%C",
            hover_data=["SAE Grade", "%C", "Conditions"], trendline="ols",
            title=f"{propiedad_temp} vs Temperatura — Normalized",
            labels={"Temp_C": "Temperatura del tratamiento (°C)", propiedad_temp: propiedad_temp, "%C": "% Carbono"},
            color_continuous_scale="Purples"
        )
        fig_normalized = aplicar_layout_plotly(fig_normalized)
        st.plotly_chart(fig_normalized, use_container_width=True)

        st.subheader("📌 ¿Por qué aparecen puntos alineados verticalmente?")
        puntos_verticales = normalized.groupby("Temp_C").agg(
            registros=("SAE Grade", "count"), aceros_distintos=("SAE Grade", "nunique"),
            carbono_min=("%C", "min"), carbono_max=("%C", "max"),
            propiedad_min=(propiedad_temp, "min"), propiedad_max=(propiedad_temp, "max")
        ).reset_index().sort_values("registros", ascending=False)
        st.dataframe(puntos_verticales, use_container_width=True)

        with st.expander("📋 Estadísticas descriptivas — Normalized"):
            col_n1, col_n2, col_n3, col_n4 = st.columns(4)
            sn = normalized[propiedad_temp].dropna()
            col_n1.metric("Media", f"{sn.mean():.1f}")
            col_n2.metric("Mín", f"{sn.min():.1f}")
            col_n3.metric("Máx", f"{sn.max():.1f}")
            col_n4.metric("Registros", len(sn))
            st.dataframe(normalized[["SAE Grade", "Conditions", "Temp_C", "%C", propiedad_temp]], use_container_width=True)


# ==========================================================
# TAB 4 — ANOVA
# ==========================================================

with tab_anova:
    st.header("Parte B — Pregunta 4: ANOVA")

    propiedad_anova = st.selectbox("Selecciona la propiedad mecánica para el ANOVA", PROPIEDADES_MECANICAS, key="propiedad_anova")
    st.markdown(f'<span class="prop-badge">Analizando: {propiedad_anova}</span>', unsafe_allow_html=True)

    datos_anova = aceros.dropna(subset=["%C", "Condition_simple", propiedad_anova]).copy()

    if len(datos_anova) > 10 and datos_anova["Condition_simple"].nunique() > 1:
        datos_anova = datos_anova.rename(columns={"%C": "Carbono", propiedad_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        modelo = ols("Propiedad ~ Carbono + C(Tratamiento)", data=datos_anova).fit()
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        tabla_anova["Influencia_%"] = (tabla_anova["sum_sq"] / tabla_anova["sum_sq"].sum()) * 100
        tabla_anova_visual = tabla_anova.reset_index().rename(columns={"index": "Factor"})

        r2 = modelo.rsquared
        if r2 < 0.4:
            st.warning(f"⚠️ El R² del modelo es **{r2:.2f}**, lo que indica una parte limitada explicada por estos factores únicos.")

        st.subheader("Tabla ANOVA")
        st.dataframe(tabla_anova_visual, use_container_width=True)

        tabla_factores = tabla_anova_visual[~tabla_anova_visual["Factor"].str.lower().str.contains("residual")].copy()

        fig_anova = px.bar(
            tabla_factores, x="Factor", y="Influencia_%", title=f"Influencia de cada factor — {propiedad_anova}",
            labels={"Factor": "Factor analizado", "Influencia_%": "Variación explicada (%)"},
            text="Influencia_%", color="Factor", color_discrete_sequence=["#38bdf8", "#818cf8"]
        )
        fig_anova.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_anova = aplicar_layout_plotly(fig_anova)
        st.plotly_chart(fig_anova, use_container_width=True)

        factor_mayor = tabla_factores.sort_values("Influencia_%", ascending=False).iloc[0]
        factor_menor = tabla_factores.sort_values("Influencia_%", ascending=False).iloc[-1]

        st.subheader("🏆 Hallazgos principales")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown(f'<div class="hallazgo-card"><h4 style="color:#38bdf8; margin:0 0 8px 0;">Factor dominante</h4><span class="badge-dominante">{factor_mayor["Factor"]}</span><p style="margin:12px 0 0 0; color:#94a3b8; font-size:14px;">Explica el <strong style="color:#e2e8f0;">{factor_mayor["Influencia_%"]:.1f}%</strong></p></div>', unsafe_allow_html=True)
        with col_h2:
            st.markdown(f'<div class="hallazgo-card"><h4 style="color:#818cf8; margin:0 0 8px 0;">Factor secundario</h4><span class="badge-secundario">{factor_menor["Factor"]}</span><p style="margin:12px 0 0 0; color:#94a3b8; font-size:14px;">Explica el <strong style="color:#e2e8f0;">{factor_menor["Influencia_%"]:.1f}%</strong></p></div>', unsafe_allow_html=True)


# ==========================================================
# TAB 5 — RECOMENDADOR MEJORADO (PRESETS + IDIOMA ESPAÑOL)
# ==========================================================

with tab_recomendador:
    st.header("🔍 Recomendador Inteligente SteelMatch AI")

    st.write("""
    **¿No sabes qué valores técnicos asignar?** Selecciona tu objetivo de uso práctico o, si eres un experto,
    elige la configuración manual para ajustar los sliders de la barra lateral a tu gusto.
    """)

    datos_modelo = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])
    if datos_modelo.empty:
        st.warning("No hay datos suficientes para usar el recomendador.")
        st.stop()

    scaler = MinMaxScaler()
    scaler.fit(datos_modelo[COLUMNAS_MODELO])

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Configuración del Acero")

    # MEJORA: Presets inteligentes explicados con lenguaje común y corriente
    perfil_uso = st.sidebar.selectbox(
        "💡 ¿Para qué vas a usar el acero?",
        [
            "Que se pueda doblar, moldear o soldar fácilmente (Alta Ductilidad)",
            "Construcción general o soporte de estructuras (Equilibrado)",
            "Para herramientas, resortes o alta resistencia al desgaste (Alta Dureza)",
            "Personalizado (Ingresar valores manualmente)"
        ]
    )

    # Lógica que asigna valores óptimos y bloquea sliders de forma automática
    if "Ductilidad" in perfil_uso:
        def_carbono, def_uts, def_ys, def_dureza, def_elongacion = 0.15, 400, 250, 120, 32
        deshabilitar_controles = True
    elif "Equilibrado" in perfil_uso:
        def_carbono, def_uts, def_ys, def_dureza, def_elongacion = 0.40, 620, 420, 185, 18
    elif "Alta Dureza" in perfil_uso:
        def_carbono, def_uts, def_ys, def_dureza, def_elongacion = 0.85, 980, 620, 290, 7
        deshabilitar_controles = True
    else:  # Manual
        def_carbono, def_uts, def_ys, def_dureza, def_elongacion = 0.40, 600, 400, 180, 20
        deshabilitar_controles = False

    if deshabilitar_controles:
        st.sidebar.caption("🎯 Los sliders han sido calibrados automáticamente para cumplir con tu objetivo.")

    # Creación de los Sliders interactivos
    carbono = st.sidebar.slider("% de Carbono Deseado", 0.0, 1.5, def_carbono, 0.01, disabled=deshabilitar_controles)
    uts = st.sidebar.slider("Resistencia Máxima UTS (MPa)", 200, 1500, def_uts, disabled=deshabilitar_controles)
    ys = st.sidebar.slider("Límite Elástico YS (MPa)", 100, 1300, def_ys, disabled=deshabilitar_controles)
    dureza = st.sidebar.slider("Dureza del Material (HB)", 50, 500, def_dureza, disabled=deshabilitar_controles)
    elongacion = st.sidebar.slider("Capacidad de Doblado / Elongación (%)", 1, 50, def_elongacion, disabled=deshabilitar_controles)

    # MEJORA: Lista de tratamientos traducida al español para el menú desplegable
    lista_interna = ["Todos"] + sorted(datos_modelo["Condition_simple"].dropna().unique().tolist())
    opciones_traducidas = [TRADUCCION_TRATAMIENTOS.get(t, t) for t in lista_interna]

    tratamiento_visual = st.sidebar.selectbox("Filtrar por Tratamiento Térmico", opciones_traducidas)
    
    # Recuperamos la clave original en inglés para que la búsqueda matemática funcione perfectamente
    tratamiento = lista_interna[opciones_traducidas.index(tratamiento_visual)]
    
    buscar = st.sidebar.button("Buscar acero ideal")

    if not deshabilitar_controles:
        st.info("ℹ️ Tienes activado el modo personalizado. Puedes ajustar los parámetros libremente a mano.")
    else:
        st.success(f"🎯 Perfil técnico optimizado para: **{perfil_uso}**")

    if buscar or deshabilitar_controles:
        datos_busqueda = datos_modelo.copy()
        if tratamiento != "Todos":
            datos_busqueda = datos_busqueda[datos_busqueda["Condition_simple"] == tratamiento]

        entrada_usuario = np.array([[carbono, uts, ys, dureza, elongacion]])
        entrada_usuario_normalizada = scaler.transform(entrada_usuario)
        X_busqueda = scaler.transform(datos_busqueda[COLUMNAS_MODELO])

        distancias = euclidean_distances(entrada_usuario_normalizada, X_busqueda)
        indice_mejor = np.argmin(distancias)
        mejor_material = datos_busqueda.iloc[indice_mejor]
        similitud = 100 / (1 + distancias[0][indice_mejor])

        st.subheader("⚙️ Resultado del emparejamiento")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Grado SAE Recomendado", mejor_material["SAE Grade"])
            st.metric("Tratamiento de Fábrica", TRADUCCION_TRATAMIENTOS.get(mejor_material["Condition_simple"], mejor_material["Condition_simple"]))
        with col2:
            st.metric("Porcentaje de Similitud", f"{similitud:.2f}%")
            st.metric("Contenido de Carbono Real", f"{mejor_material['%C']:.3f}%")
        with col3:
            st.metric("Resistencia Mecánica UTS", f"{mejor_material['UTS (MPa)']} MPa")
            st.metric("Flexibilidad (Elongación)", f"{mejor_material['Elongation (%)']}%")

        st.subheader("📋 Propiedades mecánicas completas del acero hallado")
        st.dataframe(mejor_material.to_frame(name="Valor"), use_container_width=True)


# ==========================================================
# MEJORA: CIERRE - REGLAS CLAVE DE CAUSA Y EFECTO
# ==========================================================

st.markdown("---")

st.markdown("""
<div class="conclusion-box" style="background: linear-gradient(135deg, rgba(14,165,233,0.1), rgba(99,102,241,0.1)); border-left: 5px solid #0ea5e9;">
    <h3 style="color:#38bdf8; margin-top:0;">📝 Resumen de Reglas Clave (Causa y Efecto)</h3>
    <ul style="list-style-type: none; padding-left: 0; line-height: 1.8;">
        <li style="margin-bottom: 12px;">
            🔺 <strong>A mayor porcentaje de Carbono (%C):</strong> Aumenta directamente la <strong>resistencia y la dureza</strong> del acero, permitiendo soportar más carga física, pero disminuye su ductilidad (se vuelve más quebradizo y difícil de doblar).
        </li>
        <li style="margin-bottom: 12px;">
            🔵 <strong>Tratamientos de Recocido (Annealed):</strong> Funcionan de manera inversa; reducen la dureza extrema del material y <strong>elevan la Elongación (%)</strong>, permitiendo que el acero sea <strong>altamente deformable y fácil de doblar sin fracturarse</strong>.
        </li>
        <li style="margin-bottom: 12px;">
            ⚙️ <strong>En conclusión:</strong> La química del material (%C) establece la resistencia base del acero, mientras que los tratamientos térmicos y su temperatura actúan como un control final para adaptar la flexibilidad o rigidez exacta que requiere una aplicación de ingeniería.
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)
