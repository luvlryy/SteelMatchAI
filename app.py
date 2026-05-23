# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI (Versión Final Académica)
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

# ── CONFIGURACIÓN Y ESTILOS ──
warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="SteelMatch AI | Quantum Materials", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #020617 !important; }
    h1 { color: #06b6d4 !important; font-family: 'Syne', sans-serif; }
    .card-item { background: rgba(15, 23, 42, 0.7); border: 1px solid #06b6d4; padding: 15px; border-radius: 10px; margin: 10px; }
</style>
""", unsafe_allow_html=True)

# ── CARGA DE DATOS ──
@st.cache_data
def cargar_datos():
    df = pd.read_excel("Data/steel_carbon_data.xlsx")
    df.columns = df.columns.str.strip()
    df["%C"] = (pd.to_numeric(df["C (Min)"], errors="coerce") + pd.to_numeric(df["C (Max)"], errors="coerce")) / 2
    df["Condition_simple"] = df["Conditions"].astype(str).apply(lambda x: "Annealed" if "Annealed" in x else ("Normalized" if "Normalized" in x else "Other"))
    return df

try:
    aceros = cargar_datos()
except:
    st.error("Error: No se encontró el archivo en 'Data/steel_carbon_data.xlsx'. Verifica la ruta.")
    st.stop()

# ── INTERFAZ (PESTAÑAS) ──
tab_inv, tab_inicio, tab_analisis, tab_rec = st.tabs([
    "📋 Inventario y Glosario", "🏠 Módulo Central", "📊 Análisis de Propiedades", "🔍 Requisitor"
])

with tab_inv:
    st.subheader("Resumen Académico de Aceros")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Conteo por Grado SAE")
        st.table(aceros["SAE Grade"].value_counts().reset_index().rename(columns={"index": "Grado SAE", "SAE Grade": "Conteo"}))
    with col2:
        st.write("### Tratamientos Térmicos Identificados")
        st.table(aceros["Condition_simple"].value_counts().reset_index().rename(columns={"index": "Tratamiento", "Condition_simple": "Frecuencia"}))
    
    st.markdown("---")
    st.subheader("Glosario Técnico")
    st.write("- **Annealed (Recocido):** Tratamiento para ablandar el acero y mejorar su ductilidad.")
    st.write("- **Normalized (Normalizado):** Tratamiento para refinar el grano y uniformar propiedades.")

with tab_analisis:
    st.subheader("Análisis de Tendencias (Composición vs Propiedades)")
    prop = st.selectbox("Selecciona propiedad a graficar:", ["UTS (MPa)", "YS (MPa)", "Hardness (HB)", "Elongation (%)"])
    fig = px.scatter(aceros, x="%C", y=prop, color="Condition_simple", title=f"Influencia del Carbono en {prop}")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab_inicio:
    st.title("SteelMatch AI | Quantum Materials")
    st.write("Bienvenida al sistema experto para la selección de aceros al carbono.")
    # (Aquí va tu código original de lógica de motor de búsqueda...)

with tab_rec:
    st.subheader("Buscador de Especificaciones")
    # (Aquí va tu lógica de buscador original...)
