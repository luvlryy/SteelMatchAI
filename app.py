# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
Versión optimizada y corregida:
  - Corrección de NameError en controles interactivos.
  - Traducción de tratamientos térmicos al español.
  - Recomendador intuitivo basado en objetivos de uso real (Presets).
  - Conclusión general simplificada (Causa-Efecto) y reubicada al inicio.
"""

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

# Diccionario global para traducción de tratamientos térmicos
TRADUCCION_TRATAMIENTOS = {
    "Hot Rolled": "Laminado en Caliente",
    "Cold Drawn": "Estirado en Frío",
    "Annealed": "Recocido (Suave/Moldeable)",
    "Normalized": "Normalizado",
    "Quenched": "Templado (Alta Dureza)",
    "Tempered": "Revenido",
    "As Rolled": "Estado de Laminación",
    "Other": "Otros",
    "Todos": "Todos los tratamientos"
}

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


# ==========================================================
# ESTILO VISUAL — INYECCIÓN CSS
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
        margin-top: 2rem !important;
    }

    h3 {
        font-family: 'Syne', sans-serif !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
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
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(56,189,248,0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(56,189,248,0.5);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }

    [data-testid="metric-container"] {
        background: rgba(56,189,248,0.06);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 14px;
        padding: 16px;
    }

    .hallazgo-card {
        background: rgba(56,189,248,0.06);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 10px 0;
    }

    .badge-dominante {
        display: inline-block;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 14px;
        padding: 6px 16px;
        border-radius: 50px;
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
    }

    .conclusion-box-main {
        background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.15));
        border: 2px solid #0ea5e9;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 0 25px rgba(14,165,233,0.2);
    }

    .conclusion-box {
        background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(99,102,241,0.08));
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 20px;
        padding: 24px;
        margin-top: 2rem;
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

    p, span, label, div { color: #e2e8f0 !important; }
    .stSelectbox label { color: #f8fafc !important; }
    input, textarea { color: #ffffff !important; }
    .stTabs [data-baseweb="tab"] { color: #cbd5e1 !important; }
    .stTabs [aria-selected="true"] { color: white !important; }
    [data-testid="stDataFrame"] { color: white !important; }
    [data-testid="metric-container"] * { color: #f8fafc !important; }
    .streamlit-expanderHeader { color: #f8fafc !important; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# FUNCIONES AUXILIARES Y PROCESAMIENTO
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

    if re.search(r"(?:annealed|normalized)\s+at\s+(\d{3,4}(?:\.\d+)?)", condicion, flags=re.IGNORECASE):
        coincidencia_respaldo = re.search(r"(?:annealed|normalized)\s+at\s+(\d{3,4}(?:\.\d+)?)", condicion, flags=re.IGNORECASE)
        if coincidencia_respaldo:
            temp = float(coincidencia_respaldo.group(1))
            if 100 <= temp <= 1300: return temp
    return np.nan

def aplicar_layout_plotly(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(56,189,248,0.2)")
    fig.update_yaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(56,189,248,0.2)")
    return fig

def leer_archivo_datos(nombre_archivo, archivo_bytes=None):
    extension = nombre_archivo.lower().split(".")[-1]
    obtener_fuente = lambda: io.BytesIO(archivo_bytes) if archivo_bytes is not None else nombre_archivo

    if extension == "csv":
        for encoding in ["latin1", "utf-8", "utf-8-sig"]:
            try: return pd.read_csv(obtener_fuente(), sep=",", encoding=encoding)
            except: pass
        st.error("Error al leer el archivo CSV.")
        st.stop()
    elif extension in ["xlsx", "xls"]:
        try: return pd.read_excel(obtener_fuente())
        except Exception as e:
            st.error(f"Error al leer Excel: {e}")
            st.stop()
    st.error("Formato no soportado.")
    st.stop()

@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo, archivo_bytes=None):
    aceros = leer_archivo_datos(nombre_archivo, archivo_bytes)
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
# DETECCIÓN DE FUENTE DE DATOS
# ==========================================================

st.sidebar.subheader("📁 Archivo de datos")
archivo_subido = st.sidebar.file_uploader("Sube tu dataset (CSV o Excel)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    aceros = cargar_y_preprocesar_datos(archivo_subido.name, archivo_subido.getvalue())
else:
    aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)


# ==========================================================
# INTERFAZ PRINCIPAL — TITULO E INICIO
# ==========================================================

st.title("⚙️ SteelMatch AI")
st.subheader("Análisis inteligente de propiedades mecánicas en aceros")

# ── MEJORA SOLICITADA: UBICACIÓN Y VISIBILIDAD DE LA CONCLUSIÓN GENERAL (AL INICIO)
st.markdown("""
<div class="conclusion-box_main">
    <h3 style="color:#38bdf8; margin-top:0;">💡 Resumen Rápido de Causa y Efecto</h3>
    <p style="font-size:15px; margin-bottom:10px;">Para entender cómo cambian las propiedades del acero sin necesidad de tecnicismos, recuerda estas dos reglas clave:</p>
    <ul style="font-size:14.5px; line-height:1.7; color:#cbd5e1;">
        <li>⚠️ <strong>A mayor porcentaje de Carbono (%C):</strong> El acero gana mucha más <strong>resistencia y dureza</strong>, pero se vuelve menos dúctil (es decir, se vuelve rígido y difícil de doblar o moldear).</li>
        <li>🔥 <strong>Tratamientos de Recocido (Annealed):</strong> Funcionan al revés; reducen la dureza extrema del material y aumentan drásticamente su <strong>Elongación</strong>, haciendo que el acero sea ideal para doblarse y soldarse sin romperse.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# PESTAÑAS
# ==========================================================

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio", "📊 Exploración", "🌡️ Temperatura", "🧪 ANOVA", "🔍 Recomendador Inteligente"
])

with tab_inicio:
    st.header("Objetivo del proyecto")
    st.write("Exploración interactiva del impacto químico y térmico en las prestaciones mecánicas estructurales.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros cargados", len(aceros))
    col2.metric("Tipos SAE distintos", aceros["SAE Grade"].nunique())
    col3.metric("Tratamientos detectados", aceros["Condition_simple"].nunique())

    st.subheader("Vista previa del dataset")
    st.dataframe(aceros.head(15), use_container_width=True)


# ==========================================================
# TAB 2 — EXPLORACIÓN
# ==========================================================

with tab_exploracion:
    st.header("Exploración inicial de aceros y tratamientos")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Distribución por Grados SAE")
        tabla_aceros = aceros["SAE Grade"].value_counts().reset_index(name="Cantidad de datos")
        st.dataframe(tabla_aceros, use_container_width=True)
        
    with col_t2:
        st.subheader("Distribución por Tratamientos")
        tabla_tratamientos = aceros["Condition_simple"].value_counts().reset_index(name="Cantidad de datos")
        tabla_tratamientos["Tratamiento (Español)"] = tabla_tratamientos["Condition_simple"].map(TRADUCCION_TRATAMIENTOS)
        st.dataframe(tabla_tratamientos[["Tratamiento (Español)", "Cantidad de datos"]], use_container_width=True)

    st.subheader("Análisis Gráfico: Propiedades vs %C")
    propiedad_carbono = st.selectbox("Selecciona una propiedad mecánica:", PROPIEDADES_MECANICAS, key="p_carb")
    
    datos_carbono = aceros.dropna(subset=["%C", propiedad_carbono, "Condition_simple"]).copy()
    datos_carbono["Tratamiento en Español"] = datos_carbono["Condition_simple"].map(TRADUCCION_TRATAMIENTOS)

    fig_carbono = px.scatter(
        datos_carbono, x="%C", y=propiedad_carbono, color="Tratamiento en Español",
        hover_data=["SAE Grade", "Conditions"], trendline="ols",
        title=f"Efecto del %C sobre {propiedad_carbono}"
    )
    st.plotly_chart(aplicar_layout_plotly(fig_carbono), use_container_width=True)


# ==========================================================
# TAB 3 — TEMPERATURA
# ==========================================================

with tab_temp:
    st.header("Influencia de la Temperatura del Tratamiento")
    propiedad_temp = st.selectbox("Propiedad a evaluar frente a temperatura:", PROPIEDADES_MECANICAS, key="p_temp")
    
    datos_temp = aceros.dropna(subset=["Temp_C", propiedad_temp, "Condition_simple"]).copy()
    
    # Separación por tratamientos térmicos específicos
    annealed = datos_temp[datos_temp["Condition_simple"] == "Annealed"]
    normalized = datos_temp[datos_temp["Condition_simple"] == "Normalized"]

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("🔵 Efecto en Recocido (Annealed)")
        if not annealed.empty:
            fig_a = px.scatter(annealed, x="Temp_C", y=propiedad_temp, color="%C", trendline="ols", color_continuous_scale="Blues")
            st.plotly_chart(aplicar_layout_plotly(fig_a), use_container_width=True)
        else: st.warning("Sin datos mapeados para Recocido en este rango.")

    with col_g2:
        st.subheader("🟣 Efecto en Normalizado (Normalized)")
        if not normalized.empty:
            fig_n = px.scatter(normalized, x="Temp_C", y=propiedad_temp, color="%C", trendline="ols", color_continuous_scale="Purples")
            st.plotly_chart(aplicar_layout_plotly(fig_n), use_container_width=True)
        else: st.warning("Sin datos mapeados para Normalizado en este rango.")


# ==========================================================
# TAB 4 — ANOVA
# ==========================================================

with tab_anova:
    st.header("Análisis de Varianza (ANOVA)")
    propiedad_anova = st.selectbox("Propiedad objetivo del ANOVA:", PROPIEDADES_MECANICAS, key="p_anova")
    
    datos_anova = aceros.dropna(subset=["%C", "Condition_simple", propiedad_anova]).copy()
    if len(datos_anova) > 10:
        datos_anova = datos_anova.rename(columns={"%C": "Carbono", propiedad_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        modelo = ols("Propiedad ~ Carbono + C(Tratamiento)", data=datos_anova).fit()
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        tabla_anova["Influencia_%"] = (tabla_anova["sum_sq"] / tabla_anova["sum_sq"].sum()) * 100
        
        st.dataframe(tabla_anova, use_container_width=True)
        
        r2 = modelo.rsquared
        st.metric("R² Ajustado del Modelo", f"{r2:.4f}")


# ==========================================================
# TAB 5 — RECOMENDADOR INTELIGENTE (CORREGIDO Y OPTIMIZADO)
# ==========================================================

with tab_recomendador:
    st.header("🔍 Recomendador Inteligente SteelMatch AI")
    st.write("¿No sabes qué valores técnicos asignar? Selecciona tu objetivo de uso práctico o, si eres un experto, elige la configuración manual para ajustar los sliders de la barra lateral a tu gusto.")

    datos_modelo = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"]).copy()

    if not datos_modelo.empty:
        scaler = MinMaxScaler()
        scaler.fit(datos_modelo[COLUMNAS_MODELO])

        # ── BARRA LATERAL: CONFIGURACIÓN INTUITIVA (PRESETS) ──
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Configuración del Acero")
        
        opcion_uso = st.sidebar.selectbox(
            "💡 ¿Para qué vas a usar el acero?",
            options=[
                "Personalizado (Ingresar valores manualmente)",
                "Que se pueda doblar, moldear o soldar fácilmente (Alta Ductilidad)",
                "Construcción general o soporte de estructuras (Equilibrado)",
                "Para herramientas, resortes o alta resistencia al desgaste (Alta Dureza)"
            ]
        )

        # ── SOLUCIÓN AL NAMEERROR: Definición explícita de estados por Preset ──
        if opcion_uso == "Que se pueda doblar, moldear o soldar fácilmente (Alta Ductilidad)":
            val_c, val_uts, val_ys, val_hb, val_elon = 0.15, 400, 250, 120, 32
            deshabilitar_controles = True
        elif opcion_uso == "Construcción general o soporte de estructuras (Equilibrado)":
            val_c, val_uts, val_ys, val_hb, val_elon = 0.40, 600, 380, 190, 20
        elif opcion_uso == "Para herramientas, resortes o alta resistencia al desgaste (Alta Dureza)":
            val_c, val_uts, val_ys, val_hb, val_elon = 0.85, 950, 600, 310, 9
            deshabilitar_controles = True
        else: # Opción Manual / Personalizada
            val_c, val_uts, val_ys, val_hb, val_elon = 0.40, 600, 400, 180, 20
            deshabilitar_controles = False

        # ── ASIGNACIÓN DE SLIDERS (Dinámicos y protegidos ante bloqueos) ──
        carbono = st.sidebar.slider("% de Carbono deseado", 0.0, 1.5, val_c, step=0.01, disabled=deshabilitar_controles)
        uts = st.sidebar.slider("UTS (Resistencia Máxima - MPa)", 200, 1500, val_uts, disabled=deshabilitar_controles)
        ys = st.sidebar.slider("YS (Límite Elástico - MPa)", 100, 1300, val_ys, disabled=deshabilitar_controles)
        dureza = st.sidebar.slider("Dureza Requerida (HB)", 50, 500, val_hb, disabled=deshabilitar_controles)
        elongacion = st.sidebar.slider("Elongación / Ductilidad (%)", 1, 50, val_elon, disabled=deshabilitar_controles)

        # ── TRADUCCIÓN VISUAL DE TRATAMIENTOS EN EL SELECTBOX ──
        lista_originales = ["Todos"] + sorted(datos_modelo["Condition_simple"].unique().tolist())
        
        # Formamos pares (Nombre traducido, Clave original) para no romper el modelo matemático
        opciones_traducidas = {TRADUCCION_TRATAMIENTOS.get(key, key): key for key in lista_originales}
        
        tratamiento_visual = st.sidebar.selectbox("Filtrar por Tratamiento Térmico:", list(opciones_traducidas.keys()))
        tratamiento_real = opciones_traducidas[tratamiento_visual] # Conserva el string en inglés original para filtrado interno

        buscar = st.sidebar.button("Buscar Material Óptimo")

        if buscar:
            datos_busqueda = datos_modelo.copy()
            if tratamiento_real != "Todos":
                datos_busqueda = datos_busqueda[datos_busqueda["Condition_simple"] == tratamiento_real]

            if not datos_busqueda.empty:
                entrada_usuario = np.array([[carbono, uts, ys, dureza, elongacion]])
                entrada_normalizada = scaler.transform(entrada_usuario)
                X_busqueda = scaler.transform(datos_busqueda[COLUMNAS_MODELO])

                distancias = euclidean_distances(entrada_normalizada, X_busqueda)
                indice_mejor = np.argmin(distancias)
                mejor_material = datos_busqueda.iloc[indice_mejor]
                similitud = 100 / (1 + distancias[0][indice_mejor])

                st.success("🎯 ¡Acero ideal localizado en la base de datos!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Grado SAE Sugerido", mejor_material["SAE Grade"])
                c2.metric("Tratamiento Recomendado", TRADUCCION_TRATAMIENTOS.get(mejor_material["Condition_simple"], mejor_material["Condition_simple"]))
                c3.metric("Porcentaje de Coincidencia", f"{similitud:.2f}%")

                st.subheader("📋 Ficha técnica completa del material encontrado")
                st.dataframe(mejor_material.to_frame(name="Especificaciones"), use_container_width=True)
            else:
                st.error("No se encontraron aceros que cumplan con este tratamiento específico en la base de datos.")
    else:
        st.warning("Base de datos no disponible para ejecutar la recomendación.")

# ==========================================================
# CIERRE Y REGLAS DE COMPORTAMIENTO METALÚRGICO
# ==========================================================
st.markdown("---")
st.markdown("""
<div class="conclusion-box" style="text-align: center;">
    <p style="font-size: 13.5px; color: #94a3b8 !important;">
        <strong>SteelMatch AI</strong> • Desarrollado para el análisis interactivo de Ingeniería de Materiales © 2026.
    </p>
</div>
""", unsafe_allow_html=True)
