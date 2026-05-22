# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
Versión Definitiva: Navegación Guiada y Menú Organizado
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
# CONFIGURACIÓN GENERAL Y ESTILOS
# ==========================================================

DEFAULT_DATA_FILE = "Data/steel_carbon_data.xlsx"
PROPIEDADES_MECANICAS = ["UTS (MPa)", "YS (MPa)", "Hardness (HB)", "Elongation (%)"]
COLUMNAS_MODELO = ["%C", "UTS (MPa)", "YS (MPa)", "Hardness (HB)", "Elongation (%)"]
PLOTLY_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)")

st.set_page_config(page_title="SteelMatch AI", page_icon="⚙️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #020617; color: #e2e8f0; }
    h1 { color: #38bdf8 !important; }
    .guia-didactica { background: rgba(30, 41, 59, 0.5); border-left: 4px solid #6366f1; padding: 15px; border-radius: 8px; }
    .conclusion-enriquecida { background: rgba(15, 23, 42, 0.9); border: 1px solid #38bdf8; border-radius: 15px; padding: 25px; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# FUNCIONES LÓGICAS
# ==========================================================

def agrupar_tratamiento(texto):
    texto = str(texto).lower()
    if "annealed" in texto: return "Annealed"
    elif "normalized" in texto: return "Normalized"
    elif "hot rolled" in texto: return "Hot Rolled"
    elif "cold drawn" in texto: return "Cold Drawn"
    return "Other"

@st.cache_data
def cargar_datos(archivo):
    aceros = pd.read_excel(archivo) if isinstance(archivo, str) else pd.read_excel(archivo)
    aceros["%C"] = (aceros["C (Min)"] + aceros["C (Max)"]) / 2
    aceros["Condition_simple"] = aceros["Conditions"].apply(agrupar_tratamiento)
    return aceros

# ==========================================================
# MENÚ LATERAL REORGANIZADO
# ==========================================================

st.sidebar.title("Menú de Navegación")

with st.sidebar.expander("📚 Diccionario Técnico", expanded=True):
    st.markdown("""
    - **UTS:** Resistencia máxima a la ruptura.
    - **YS:** Límite elástico (deformación).
    - **Dureza:** Resistencia al rayado.
    - **Elongación:** Ductilidad (estiramiento).
    """)

with st.sidebar.expander("⚙️ Configuración y Búsqueda", expanded=True):
    modo = st.radio("Perfil:", ["Guiado", "Manual"])
    carbono = st.slider("Carbono (%)", 0.0, 1.5, 0.45)
    uts = st.slider("UTS (MPa)", 200, 1500, 600)
    ys = st.slider("YS (MPa)", 100, 1300, 400)
    dureza = st.slider("Dureza (HB)", 50, 500, 180)
    elongacion = st.slider("Elongación (%)", 1, 50, 20)
    buscar = st.button("🚀 Buscar Material Óptimo")

# ==========================================================
# VISTA PRINCIPAL
# ==========================================================

aceros = cargar_datos(DEFAULT_DATA_FILE)
st.title("SteelMatch AI")

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio", "📊 Exploración", "🌡️ Temperatura", "🧪 ANOVA", "🔍 Recomendador"
])

with tab_inicio:
    st.header("Bienvenido a SteelMatch AI")
    st.markdown("""
    <div class="guia-didactica">
        Explora las propiedades de los aceros:
        - <strong>Exploración:</strong> Relación entre Carbono y propiedades.
        - <strong>Temperatura:</strong> Efecto del calor en el tratamiento.
        - <strong>ANOVA:</strong> Análisis de influencia estadística.
        - <strong>Recomendador:</strong> Usa el menú lateral para encontrar tu acero ideal.
    </div>
    """, unsafe_allow_html=True)

with tab_exploracion:
    st.header("Exploración")
    prop = st.selectbox("Propiedad a analizar", PROPIEDADES_MECANICAS)
    fig = px.scatter(aceros, x="%C", y=prop, color="Condition_simple", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

# [El resto de las pestañas siguen la misma lógica...]

st.markdown("""
<div class="conclusion-enriquecida">
    <h3>Conclusión</h3>
    <p>El acero es un balance entre dureza y flexibilidad. El carbono aumenta la resistencia, mientras que el recocido mejora la ductilidad.</p>
</div>
""", unsafe_allow_html=True)
