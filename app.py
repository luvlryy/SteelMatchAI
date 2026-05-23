# -*- coding: utf-8 -*-
"""
SteelMatch AI — Versión Ingeniería Práctica y Visual
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

# ── CONFIGURACIÓN INICIAL ──
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_DATA_FILE = "Data/steel_carbon_data.xlsx"

PROPIEDADES_MECANICAS = {
    "UTS (MPa)": "💪 Resistencia (Fuerza máxima)",
    "YS (MPa)": "📐 Límite Elástico (Soporte de carga)",
    "Hardness (HB)": "💎 Dureza (Resistencia a rayaduras)",
    "Elongation (%)": "🎗️ Elongación (Capacidad de doblado)"
}

# Inverso para procesos internos
INV_PROPIEDADES = {v: k for k, v in PROPIEDADES_MECANICAS.items()}

COLUMNAS_MODELO = ["%C", "UTS (MPa)", "YS (MPa)", "Hardness (HB)", "Elongation (%)"]

st.set_page_config(page_title="SteelMatch AI", page_icon="⚙️", layout="wide")

# ── TRADUCCIONES ──
TRATS_ES = {
    "Annealed": "Recocido (Ablandado)",
    "Normalized": "Normalizado (Equilibrado)",
    "Hot Rolled": "Laminado en caliente",
    "Cold Drawn": "Estirado en frío (Reforzado)",
    "Quenched": "Templado (Endurecido)",
    "Tempered": "Revenido (Tenaz)",
    "As Rolled": "Laminado en bruto",
    "Other": "Otros procesos"
}
INV_TRATS = {v: k for k, v in TRATS_ES.items()}

# ── DISEÑO VISUAL (CSS) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Syne:wght@800&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
    }

    /* Estilo Metálico para Títulos */
    h1 { 
        font-family: 'Syne', sans-serif !important;
        background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        text-align: center;
        padding-bottom: 20px;
    }

    h2 {
        font-family: 'Inter', sans-serif !important;
        color: #f8fafc !important;
        border-bottom: 2px solid #334155;
        padding-bottom: 10px;
    }

    /* Selectores con texto negro para legibilidad */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Tarjetas de información */
    .intro-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
    }

    .interpretacion-card {
        background: #0f172a;
        border-left: 5px solid #60a5fa;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin-top: 15px;
    }

    .stat-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── FUNCIONES ──
@st.cache_data
def cargar_datos(ruta):
    df = pd.read_excel(ruta)
    df.columns = df.columns.astype(str).str.strip()
    df["%C"] = (pd.to_numeric(df["C (Min)"], errors="coerce") + pd.to_numeric(df["C (Max)"], errors="coerce")) / 2
    for col in INV_PROPIEDADES.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    def limpiar_trat(t):
        t = str(t).lower()
        for k in TRATS_ES.keys():
            if k.lower() in t: return k
        return "Other"
    
    df["Condition_simple"] = df["Conditions"].apply(limpiar_trat)
    
    def ext_temp(c):
        match = re.search(r"(\d+)\s*°\s*[Cc]", str(c))
        return float(match.group(1)) if match else np.nan
    
    df["Temp_C"] = df["Conditions"].apply(ext_temp)
    return df

def graficar_bonito(fig):
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
        font=dict(family="Inter", size=14),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ── LÓGICA DE DATOS ──
aceros = cargar_datos(DEFAULT_DATA_FILE)

# ── BARRA LATERAL ──
st.sidebar.header("🛠️ Panel de Control")
modo = st.sidebar.radio("¿Cómo quieres buscar?", ["Fácil (Niveles)", "Avanzado (Números)"])

if modo == "Fácil (Niveles)":
    r_res = st.sidebar.select_slider("Resistencia:", ["Baja", "Media", "Alta"], "Media")
    r_dur = st.sidebar.select_slider("Dureza:", ["Baja", "Media", "Alta"], "Media")
    r_duc = st.sidebar.select_slider("Doblado:", ["Baja", "Media", "Alta"], "Media")
    
    mapa = {"Baja": 0.2, "Media": 0.5, "Alta": 0.8}
    carb_s, uts_s, dureza_s, elo_s = mapa[r_res]*1.2, mapa[r_res]*800, mapa[r_dur]*300, mapa[r_duc]*30
    ys_s = uts_s * 0.7
else:
    carb_s, uts_s, ys_s, dureza_s, elo_s = 0.45, 600, 400, 180, 20

c_val = st.sidebar.slider("% Carbono", 0.0, 1.5, float(carb_s))
u_val = st.sidebar.slider("Resistencia (MPa)", 200, 1500, int(uts_s))
y_val = st.sidebar.slider("Límite Elástico (MPa)", 100, 1300, int(ys_s))
d_val = st.sidebar.slider("Dureza (HB)", 50, 500, int(dureza_s))
e_val = st.sidebar.slider("Doblado (%)", 1, 50, int(elo_s))

trat_sel = st.sidebar.selectbox("Tratamiento deseado", ["Todos"] + list(TRATS_ES.values()))
btn_buscar = st.sidebar.button("🔍 Encontrar mi acero ideal")

# ── CUERPO PRINCIPAL ──
st.title("SteelMatch AI")

tabs = st.tabs(["🏠 Inicio", "📊 Explorador Visual", "🌡️ El Horno", "🧬 ¿Quién manda?", "🔍 Buscador"])

# -- TAB 1: INICIO --
with tabs[0]:
    st.markdown("""
    <div class="intro-box">
        <h2>👋 ¡Hola! Bienvenido al mundo del acero</h2>
        <p>Esta herramienta es tu asistente para entender por qué algunos metales son duros como rocas y otros se doblan como ligas. 
        No necesitas ser un científico; solo ajusta los controles y nosotros te mostramos el resultado.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Aceros Analizados", len(aceros))
    col2.metric("Grados SAE", aceros["SAE Grade"].nunique())
    col3.metric("Tratamientos", aceros["Condition_simple"].nunique())

    st.markdown("""
    <div class="interpretacion-card">
        <h3>💡 Lo que debes saber hoy:</h3>
        <ul>
            <li><b>Carbono:</b> Es el "músculo". Más carbono = Más fuerza, pero es más difícil de doblar.</li>
            <li><b>El Horno:</b> Es el "entrenamiento". Calentar y enfriar el metal cambia su personalidad.</li>
            <li><b>Dureza:</b> Qué tanto aguanta que lo rayen o golpeen.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -- TAB 2: EXPLORADOR --
with tabs[1]:
    st.markdown("""
    <div class="intro-box">
        <h3>📊 ¿Qué estamos viendo?</h3>
        <p>Aquí comparamos la <b>química</b> (Carbono) contra el <b>rendimiento</b>. 
        Usa el selector de abajo para ver cómo cambia el acero a medida que le pones más "músculo" (Carbono).</p>
    </div>
    """, unsafe_allow_html=True)
    
    opcion = st.selectbox("¿Qué característica quieres ver?", list(PROPIEDADES_MECANICAS.values()))
    col_y = INV_PROPIEDADES[opcion]
    
    df_plot = aceros.dropna(subset=["%C", col_y]).copy()
    df_plot["Tratamiento"] = df_plot["Condition_simple"].map(TRATS_ES)
    
    fig = px.scatter(df_plot, x="%C", y=col_y, color="Tratamiento", trendline="ols", 
                     title=f"Impacto del Carbono en {opcion}",
                     labels={"%C": "% de Carbono", col_y: opcion})
    st.plotly_chart(graficar_bonito(fig), use_container_width=True, theme=None)
    
    st.markdown(f"""
    <div class="interpretacion-card">
        <h4>🧐 Cómo leer esto:</h4>
        <p>Si los puntos van <b>hacia arriba</b>: Significa que mientras más Carbono tiene, más alta es la {opcion}.<br>
        Si la línea va <b>hacia abajo</b>: Significa que el Carbono debilita esta característica.</p>
    </div>
    """, unsafe_allow_html=True)

# -- TAB 3: TEMPERATURA --
with tabs[2]:
    st.markdown("""
    <div class="intro-box">
        <h3>🌡️ El efecto del calor</h3>
        <p>Aquí vemos cómo cambia el metal según los grados de calor en el horno. 
        El calor "relaja" los átomos y permite que el metal sea más fácil de trabajar.</p>
    </div>
    """, unsafe_allow_html=True)
    
    op_t = st.selectbox("Elige característica:", list(PROPIEDADES_MECANICAS.values()), key="temp_sel")
    col_t = INV_PROPIEDADES[op_t]
    
    df_t = aceros.dropna(subset=["Temp_C", col_t])
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔵 Recocido (Ablandar)")
        sub_ann = df_t[df_t["Condition_simple"] == "Annealed"]
        if not sub_ann.empty:
            f1 = px.scatter(sub_ann, x="Temp_C", y=col_t, color="%C", trendline="ols", color_continuous_scale="Blues")
            st.plotly_chart(graficar_bonito(f1), use_container_width=True, theme=None)
        else: st.warning("Sin datos de temperatura aquí.")
    
    with c2:
        st.subheader("⚪ Normalizado (Equilibrar)")
        sub_norm = df_t[df_t["Condition_simple"] == "Normalized"]
        if not sub_norm.empty:
            f2 = px.scatter(sub_norm, x="Temp_C", y=col_t, color="%C", trendline="ols", color_continuous_scale="Greys")
            st.plotly_chart(graficar_bonito(f2), use_container_width=True, theme=None)
        else: st.warning("Sin datos de temperatura aquí.")

    st.markdown("""
    <div class="interpretacion-card">
        <h4>🧐 Interpretación rápida:</h4>
        <p>Normalmente, a <b>mayor temperatura</b>, el metal se vuelve más "suave" (menos resistencia, pero más fácil de doblar). 
        Es como la mantequilla: fría es dura, caliente es maleable.</p>
    </div>
    """, unsafe_allow_html=True)

# -- TAB 4: ANOVA (QUIEN MANDA) --
with tabs[3]:
    st.markdown("""
    <div class="intro-box">
        <h3>🧬 ¿Quién manda en el cambio?</h3>
        <p>¿Es culpa de la química o es culpa del horno? Este gráfico te dice qué porcentaje de responsabilidad tiene cada uno 
        en la característica que elijas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    op_m = st.selectbox("Elige qué analizar:", list(PROPIEDADES_MECANICAS.values()), key="manda_sel")
    col_m = INV_PROPIEDADES[op_m]
    
    df_m = aceros.dropna(subset=["%C", "Condition_simple", col_m]).copy()
    if len(df_m) > 10:
        df_m = df_m.rename(columns={"%C": "C", col_m: "P", "Condition_simple": "T"})
        mod = ols("P ~ C + C(T)", data=df_m).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Inf"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        
        c_p, t_p = tab_a.loc["C", "Inf"], tab_a.loc["C(T)", "Inf"]
        
        fig_pie = go.Figure(data=[go.Pie(labels=["🧪 Química (Carbono)", "🔥 Proceso (Horno)", "🌀 Otros"], 
                                       values=[c_p, t_p, 100-c_p-t_p], hole=.6,
                                       marker_colors=["#60a5fa", "#94a3b8", "#334155"])])
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)
        
        manda = "la Química" if c_p > t_p else "el Horno"
        st.markdown(f"""
        <div class="interpretacion-card">
            <h4>📢 El Veredicto:</h4>
            <p>En este caso, <b>{manda}</b> es quien tiene el control principal ({max(c_p, t_p):.1f}% de importancia). 
            Si quieres cambiar la {op_m}, enfócate primero en ajustar eso.</p>
        </div>
        """, unsafe_allow_html=True)

# -- TAB 5: RECOMENDADOR --
with tabs[4]:
    st.markdown("""
    <div class="intro-box">
        <h3>🔍 El Buscador Inteligente</h3>
        <p>Aquí te decimos qué acero comercial (SAE) comprar según lo que necesitas. 
        <b>¡No olvides presionar el botón de la izquierda!</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if btn_buscar:
        scaler = MinMaxScaler()
        datos_m = aceros.dropna(subset=COLUMNAS_MODELO)
        scaler.fit(datos_m[COLUMNAS_MODELO])
        
        df_f = datos_m.copy()
        if trat_sel != "Todos":
            df_f = df_f[df_f["Condition_simple"] == INV_TRATS[trat_sel]]
        
        if not df_f.empty:
            usr = scaler.transform([[c_val, u_val, y_val, d_val, e_val]])
            db_norm = scaler.transform(df_f[COLUMNAS_MODELO])
            dist = euclidean_distances(usr, db_norm)
            match = df_f.iloc[np.argmin(dist)]
            
            st.balloons()
            st.success(f"¡Encontramos tu acero ideal: SAE {match['SAE Grade']}!")
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Grado Comercial", f"SAE {match['SAE Grade']}")
            with c2: st.metric("Carbono Real", f"{match['%C']:.3f} %")
            with c3: st.metric("Resistencia Real", f"{match['UTS (MPa)']} MPa")
            
            st.dataframe(match.to_frame(name="Especificación"), use_container_width=True)
        else:
            st.warning("No hay aceros con ese tratamiento, intenta con 'Todos'.")
