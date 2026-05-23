# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI (Versión Mejorada y Documentada)
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
# CONFIGURACIÓN GENERAL Y PALETA DE COLORES
# ==========================================================

DEFAULT_DATA_FILE = "Data/steel_carbon_data.xlsx" # Asegúrate de que esta ruta sea correcta

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
    # LEYENDA MOVIDA ABAJO PARA EVITAR OVERLAPPING CON LOS TÍTULOS
    legend=dict(
        bgcolor="rgba(2, 6, 23, 0.9)", 
        bordercolor="#06b6d4", 
        borderwidth=1,
        orientation="h",
        yanchor="top",
        y=-0.2, # Ubicación inferior
        xanchor="center",
        x=0.5,
        font=dict(color="#ffffff")
    ),
    margin=dict(t=30, b=80, l=40, r=40) # Mayor margen inferior para acomodar la leyenda
)


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="SteelMatch AI | Análisis de Aceros",
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

DESCRIPCIONES_TRATAMIENTOS = {
    "Annealed": "Calentamiento seguido de un enfriamiento lento y controlado dentro del horno. Alivia tensiones internas, incrementa la ductilidad y facilita los procesos de mecanizado.",
    "Normalized": "Calentamiento por encima de la temperatura crítica seguido de un enfriamiento al aire libre. Homogeneiza el tamaño de grano de la microestructura y refina las propiedades mecánicas.",
    "Hot Rolled": "Proceso de deformación mecánica realizado a altas temperaturas (por encima de la temperatura de recristalización). Reduce tensiones de fluencia estructural.",
    "Cold Drawn": "Estirado o deformación mecánica en frío a temperatura ambiente. Incrementa notoriamente el límite elástico (YS) y la dureza por acritud mecánica.",
    "Quenched": "Enfriamiento drástico y veloz en agua o aceite desde la fase austenítica. Transforma la estructura en martensita, maximizando la dureza superficial a costa de la fragilidad.",
    "Tempered": "Tratamiento térmico secundario aplicado posterior al temple. Reduce la fragilidad extrema y estabiliza las tensiones moleculares, restaurando la tenacidad general.",
    "As Rolled": "Estado comercial en bruto tras pasar por los rodillos de laminación directa, sin tratamientos térmicos de reacondicionamiento estructural posteriores.",
    "Other": "Tratamientos metalúrgicos de carácter compuesto o variaciones específicas."
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
    
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(6, 182, 212, 0.3) !important;
    }

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

    div[data-baseweb="select"] > div, .stNumberInput input, .stSlider > div {
        background-color: rgba(15, 23, 42, 0.9) !important; 
        border: 1px solid #06b6d4 !important; 
        border-radius: 8px;
    }
    
    div[data-baseweb="select"] > div * { color: #ffffff !important; font-weight: 600; }

    div[data-baseweb="popover"] div[role="listbox"],
    div[data-baseweb="popover"] ul[role="listbox"] {
        background-color: #f8fafc !important; 
        border: 2px solid #06b6d4 !important;
        border-radius: 8px;
    }
    
    div[data-baseweb="popover"] li[role="option"] {
        background-color: transparent !important;
    }

    div[data-baseweb="popover"] li[role="option"] * {
        color: #000000 !important; 
        font-weight: 700 !important;
    }

    div[data-baseweb="popover"] li[role="option"]:hover,
    div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
        background-color: #06b6d4 !important;
    }
    
    div[data-baseweb="popover"] li[role="option"]:hover *,
    div[data-baseweb="popover"] li[role="option"][aria-selected="true"] * {
        color: #ffffff !important; 
    }

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
    }

    .intro-seccion {
        background: rgba(15, 23, 42, 0.6);
        border: 2px solid #06b6d4;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .intro-seccion h4 { color: #34d399 !important; margin-top: 0 !important; }

    .leyenda-grafica {
        background: rgba(2, 6, 23, 0.8);
        border: 1px solid #06b6d4; 
        border-left: 5px solid #06b6d4;
        padding: 15px 20px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 15px;
        color: #ffffff !important;
    }
    
    .conclusion-box {
        background: rgba(2, 6, 23, 0.8);
        border: 1px solid #34d399; 
        border-left: 5px solid #34d399;
        padding: 15px 20px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 15px;
        color: #ffffff !important;
    }

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
    }
    
    .card-conclusion-item strong {
        color: #06b6d4 !important; 
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-conclusion-item p { margin: 10px 0 0 0 !important; }

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
    return fig

@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo):
    try:
        aceros = pd.read_excel(nombre_archivo)
    except:
        st.error(f"Error: Asegúrate de que el archivo {nombre_archivo} exista en la carpeta Data.")
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

st.sidebar.header("⚙️ Menú de Navegación Rápida")

st.sidebar.markdown("---")
with st.sidebar.expander("📖 GLOSARIO TÉCNICO", expanded=True):
    st.markdown("""
    <div style="font-size: 14px; color: #ffffff; line-height: 1.6;">
        <p><strong style="color:#06b6d4">💪 UTS (Fuerza Máxima):</strong> Cuánta fuerza bruta soporta antes de partirse en dos.</p>
        <p><strong style="color:#06b6d4">📐 YS (Límite Elástico):</strong> Cuánto peso aguanta antes de quedarse deformado permanentemente.</p>
        <p><strong style="color:#06b6d4">💎 Dureza (HB):</strong> Qué tan difícil es rayarlo o abollarlo en la superficie.</p>
        <p><strong style="color:#06b6d4">🎗️ Elongación (%):</strong> Qué tanto se estira como liga antes de romperse (ductilidad).</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# ESTRUCTURA DE PESTAÑAS (Módulos Integrados)
# ==========================================================

st.title("SteelMatch AI 🤖 // Análisis de Aceros al Carbono")

tab_inicio, tab_inventario, tab_exploracion, tab_temp, tab_anova = st.tabs([
    "🏠 Bienvenida y Reporte", "📋 Inventario de Datos", "📊 Gráficas y Tendencias", "🔥 Protocolo Térmico", "⚖️ ANOVA"
])


# ── TAB 1 — INICIO (Página de Bienvenida Mejorada) ──
with tab_inicio:
    st.markdown("""
    <div class="intro-seccion">
        <h2>🚀 ¡Bienvenido al Simulador SteelMatch AI!</h2>
        <p>Estás dentro de una plataforma interactiva diseñada para decodificar los secretos metalúrgicos del acero al carbono. No necesitas ser un ingeniero experto para entenderlo. Aquí transformamos números complejos en visualizaciones claras que revelan cómo la química y el fuego moldean el metal que construye nuestro mundo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 ¿Qué hicimos? (Objetivos y Metodología)")
    st.markdown("""
    <div class="leyenda-grafica">
        <ul>
            <li><strong>Recopilación:</strong> Extrajimos y limpiamos un extenso dataset de aceros al carbono (SAE) que incluye registros de propiedades como resistencia, dureza y ductilidad.</li>
            <li><strong>Preprocesamiento:</strong> Promediamos los rangos de carbono (%C) para tener un indicador químico sólido y agrupamos decenas de descripciones de hornos en categorías limpias como <i>"Recocido"</i> o <i>"Templado"</i>.</li>
            <li><strong>Desarrollo Visual:</strong> Construimos esta aplicación utilizando algoritmos de regresión y análisis de varianza (ANOVA) para graficar automáticamente las relaciones ocultas en los datos.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 ¿Por qué lo hicimos? (Nuestra Hipótesis)")
    st.markdown("""
    <div class="leyenda-grafica">
        <p>En la industria, elegir el acero equivocado puede hacer que un puente colapse o que una herramienta se rompa en el primer uso. Nuestra hipótesis central fue: <strong>"El contenido de Carbono es el motor de la fuerza bruta de un material, pero el Tratamiento Térmico (el horneado) es el volante que decide si esa fuerza será rígida, frágil o flexible."</strong> Creamos esta app para comprobarlo visualmente.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 ¿Qué encontramos? (Conclusiones Generales)")
    st.markdown("""
    <div class="grid-conclusion">
        <div class="card-conclusion-item">
            <strong>🧂 El Poder del Carbono:</strong>
            <p>Descubrimos una tendencia innegable: a medida que sube el % de Carbono, la Resistencia (UTS/YS) y la Dureza se disparan. Sin embargo, la Elongación cae drásticamente. En resumen: más carbono hace al acero más fuerte, pero mucho más frágil y difícil de moldear.</p>
        </div>
        <div class="card-conclusion-item">
            <strong>🛀 La Magia del Recocido:</strong>
            <p>En los datos, el tratamiento térmico de "Recocido" (enfriar el metal muy lento) demostró ser el mejor amigo de la flexibilidad. Las muestras recocidas mostraron los valores más altos de elongación, preparándolas para ser trabajadas sin romperse.</p>
        </div>
        <div class="card-conclusion-item">
            <strong>❄️ Tensión en Frío (Cold Drawn):</strong>
            <p>Descubrimos que estirar el metal en frío (sin meterlo al horno) eleva masivamente su límite elástico. Es decir, las vigas procesadas así pueden cargar muchísimo más peso antes de doblarse para siempre.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2 — INVENTARIO BASE ──
with tab_inventario:
    st.markdown("### 📋 Inventario de Datos Analizados")
    st.markdown("Antes de graficar, necesitamos saber qué ingredientes tenemos en la cocina. Aquí puedes ver la cantidad de registros por grado de acero y la distribución de los tratamientos aplicados.")

    c_inv1, c_inv2 = st.columns(2)
    
    with c_inv1:
        st.subheader("Grados de Aceros Disponibles")
        df_sae_counts = aceros["SAE Grade"].value_counts().reset_index()
        df_sae_counts.columns = ["Grado (SAE)", "Registros"]
        st.dataframe(df_sae_counts, use_container_width=True, hide_index=True)
        
    with c_inv2:
        st.subheader("Tratamientos Analizados")
        df_treat_counts = aceros["Condition_simple"].value_counts().reset_index()
        df_treat_counts.columns = ["Tratamiento", "Registros"]
        df_treat_counts["Explicación"] = df_treat_counts["Tratamiento"].map(DESCRIPCIONES_TRATAMIENTOS)
        st.dataframe(df_treat_counts, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="conclusion-box">
        ✅ <strong>Conclusión de la Tabla:</strong> Observar estos conteos nos permite identificar qué aceros son los más comunes en la industria y nos asegura que tenemos suficientes muestras estadísticas para los tratamientos principales (como Normalizado, Recocido y Trabajado en Frío) para que las gráficas siguientes sean confiables.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 3 — EXPLORACIÓN (Matriz Química) ──
with tab_exploracion:
    
    st.markdown("### 🔬 Curva de Tendencia Multivariable Integrada")
    df_multivar = aceros.dropna(subset=["%C"] + PROPIEDADES_MECANICAS).copy()
    df_multivar["Elongation (%) x 10"] = df_multivar["Elongation (%)"] * 10
    
    st.markdown("""
    <div class="leyenda-grafica">
        📖 <strong>¿Cómo leer esta gráfica?</strong><br>
        • Cada punto es una pieza de acero. El eje X (abajo) es cuánto <strong>Carbono</strong> tiene. El eje Y (izquierda) es su puntaje en la propiedad.<br>
        • Para que pudieras ver la <strong>Elongación</strong> junto con las demás (que manejan números muy grandes), la multiplicamos por 10. ¡Es un truco visual para comparar tendencias juntas!
    </div>
    """, unsafe_allow_html=True)
    
    fig_multi = go.Figure()
    # Puntos con opacidad para evitar saturación
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["UTS (MPa)"], mode='markers', name='Fuerza Máxima (UTS)', marker=dict(color='#34d399', opacity=0.7)))
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["YS (MPa)"], mode='markers', name='Límite Elástico (YS)', marker=dict(color='#06b6d4', opacity=0.7)))
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["Hardness (HB)"], mode='markers', name='Dureza (HB)', marker=dict(color='#a78bfa', opacity=0.7)))
    fig_multi.add_trace(go.Scatter(x=df_multivar["%C"], y=df_multivar["Elongation (%) x 10"], mode='markers', name='Elongación (x10)', marker=dict(color='#f43f5e', opacity=0.7)))
    
    fig_multi.update_xaxes(title_text="Concentración de Carbono (%C)")
    fig_multi.update_yaxes(title_text="Magnitud de la Propiedad")
    st.plotly_chart(aplicar_layout_estetico(fig_multi), use_container_width=True, theme=None)

    st.markdown("""
    <div class="conclusion-box">
        ✅ <strong>Conclusión de la Tendencia:</strong> ¡Nuestra hipótesis se confirma! Observa cómo los puntos verdes (Fuerza) y morados (Dureza) suben como escalera al avanzar a la derecha (más carbono). En contraste, los puntos rojos (Elongación) caen en picada. <strong>Regla de oro: No puedes tener máxima dureza y máxima flexibilidad al mismo tiempo.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Análisis específico por propiedad individual
    st.markdown("### 🎯 Análisis Individual vs Carbono (Por Tratamiento)")
    prop_c = st.selectbox("Selecciona la propiedad física a aislar:", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    df_carb["Protocolo Térmico"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    st.markdown("""
    <div class="leyenda-grafica">
        📖 <strong>¿Cómo leer las Líneas de Tendencia?</strong><br>
        • Las líneas rectas cruzando los puntos se llaman <strong>Regresión Lineal</strong>. Son como el "promedio del camino".<br>
        • Si la línea apunta <strong>hacia arriba ↗️</strong>, significa que añadir Carbono mejora esta propiedad.<br>
        • Si la línea apunta <strong>hacia abajo ↘️</strong>, significa que el Carbono la empeora.<br>
        • Cada color representa un tratamiento térmico distinto para ver si el horno hace trampa o ayuda.
    </div>
    """, unsafe_allow_html=True)

    fig_c = px.scatter(
        df_carb, x="%C", y=prop_c, 
        color="Protocolo Térmico", 
        hover_data=["SAE Grade"], 
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Alphabet 
    )
    # Quitamos el título interno de plotly para evitar overlapping, ya lo pusimos con st.markdown arriba
    fig_c.update_layout(title="")
    st.plotly_chart(aplicar_layout_estetico(fig_c), use_container_width=True, theme=None)

    # REQUERIMIENTO SECCIÓN 5 PARTE 2: Diagrama de Cajas (Boxplots)
    st.markdown(f"### 📦 Boxplot: Dispersión de {prop_c} por Tratamiento")
    
    st.markdown("""
    <div class="leyenda-grafica">
        📖 <strong>¿Cómo leer un Boxplot (Cajas)?</strong><br>
        • La <strong>caja de color</strong> encierra al 50% de los aceros más comunes de ese grupo.<br>
        • La <strong>línea gruesa</strong> dentro de la caja es la "mediana" (el acero promedio).<br>
        • Las <strong>antenas (líneas arriba y abajo)</strong> muestran el rango total normal.<br>
        • Los <strong>puntos sueltos</strong> afuera de las antenas son <i>Outliers</i> o "bichos raros" que tuvieron valores extremos.
    </div>
    """, unsafe_allow_html=True)

    fig_box = px.box(
        df_carb, x="Protocolo Térmico", y=prop_c,
        color="Protocolo Térmico",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        points="all" # Muestra los puntos junto a la caja para más detalle
    )
    fig_box.update_layout(title="") # Sin título interno para evitar overlapping
    fig_box.update_xaxes(title_text="Tratamiento Aplicado")
    fig_box.update_yaxes(title_text=prop_c)
    st.plotly_chart(aplicar_layout_estetico(fig_box), use_container_width=True, theme=None)
    
    st.markdown("""
    <div class="conclusion-box">
        ✅ <strong>Conclusión de los Boxplots:</strong> Esta gráfica es brutalmente honesta. Te dice quién gana en promedio. Si seleccionas "Elongation", verás que la caja del "Recocido (Enfriado Lento)" está mucho más alta que las demás. Si seleccionas "YS", verás que la caja de "Estirado en Frío" domina. ¡El tratamiento define la especialidad del acero!
    </div>
    """, unsafe_allow_html=True)


# ── TAB 4 — TEMPERATURA (Protocolo Térmico) ──
with tab_temp:
    st.markdown("### 🔥 Módulo de Influencia Térmica // Grados Centígrados")
    st.markdown("Aquí analizamos exclusivamente los aceros donde sabemos exactamente a cuántos grados (Temperatura) se metieron al horno para procesos de Recocido o Normalizado.")
    
    prop_t = st.selectbox("Propiedad a evaluar contra el Calor:", PROPIEDADES_MECANICAS, key="sb_p_t")
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    st.markdown("""
    <div class="leyenda-grafica">
        📖 <strong>¿Cómo leer las gráficas Térmicas?</strong><br>
        • <strong>Eje X:</strong> Es el termómetro (°C). A la derecha es más caliente.<br>
        • <strong>Eje Y:</strong> Es la propiedad que seleccionaste arriba.<br>
        • <strong>Color de los puntos:</strong> Es la cantidad de Carbono. Amarillo/Claro significa MUCHO carbono, Morado/Oscuro significa POCO carbono.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔵 Análisis de Recocido (Annealed)")
    df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
    if not df_ann.empty:
        fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Viridis")
        fig_a.update_layout(title="")
        st.plotly_chart(aplicar_layout_estetico(fig_a), use_container_width=True, theme=None)
        
        st.markdown(f"""
        <div class="conclusion-box">
            ✅ <strong>Conclusión Recocido:</strong> Observa la línea de tendencia. Si apunta hacia abajo en propiedades de fuerza (UTS), significa que "hornear" a temperaturas más altas relaja tanto el material que le quita resistencia, pero a cambio, lo hace más suave.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No hay suficientes datos con temperatura numérica explícita para Recocido.")

    st.markdown("---")

    st.subheader("🟣 Análisis de Normalizado (Normalized)")
    df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
    if not df_norm.empty:
        fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Plasma")
        fig_n.update_layout(title="")
        st.plotly_chart(aplicar_layout_estetico(fig_n), use_container_width=True, theme=None)
        
        st.markdown(f"""
        <div class="conclusion-box">
            ✅ <strong>Conclusión Normalizado:</strong> Si notas puntos apilados verticalmente (como una torre en una misma temperatura), es porque en la industria se usan temperaturas estandarizadas (ej. exactamente 870°C o 900°C) para normalizar diferentes grados de acero. A la misma temperatura, los que quedan más arriba en dureza son los de color más claro (los que tienen más carbono).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No hay suficientes datos con temperatura numérica explícita para Normalizado.")


# ── TAB 5 — ¿QUÉ INFLUYE MÁS? (Control de Variables // ANOVA) ──
with tab_anova:
    st.markdown("### ⚖️ Algoritmo de ANOVA (Analysis of Variance)")
    st.markdown("Cuando una propiedad sube o baja... ¿A quién le echamos la culpa? ¿A la cantidad de carbono o al horneado? El ANOVA es un juez estadístico que reparte porcentajes de 'culpa' (influencia) a cada variable.")
    
    prop_anova = st.selectbox("Selecciona la propiedad para el juicio ANOVA:", PROPIEDADES_MECANICAS, key="sb_p_a")
    
    df_anova = aceros.dropna(subset=["%C", "Condition_simple", prop_anova]).copy()
    if len(df_anova) > 10:
        df_anova = df_anova.rename(columns={"%C": "Carbono", prop_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        mod = ols("Propiedad ~ Carbono + C(Tratamiento)", data=df_anova).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Influencia_%"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        
        pct_carbono = tab_a.loc["Carbono", "Influencia_%"]
        pct_tratamiento = tab_a.loc["C(Tratamiento)", "Influencia_%"]
        pct_otros = tab_a.loc["Residual", "Influencia_%"]
        
        st.subheader(f"Vector de Responsabilidad para: {prop_anova}")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Factor 1: Química (% Carbono)", "Factor 2: Protocolo (Tratamiento)", "Factor 3: Residual (Variaciones/Errores)"],
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
            showlegend=True, # Dejamos la leyenda de la dona a la derecha, por defecto
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

        st.markdown("<div class='conclusion-box'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Veredicto del Juez Estadístico")
        if pct_carbono > pct_tratamiento:
            st.markdown(f"**LA QUÍMICA GANA:** Para modificar **{prop_anova}**, el Carbono es el factor más poderoso (representa el **{pct_carbono:.1f}%** de la influencia total). El tratamiento térmico es secundario. Moraleja: ¡Tienes que cambiar el grado de acero SAE desde la fundición!")
        else:
            st.markdown(f"**EL HORNO GANA:** Para modificar **{prop_anova}**, no importa tanto el acero que compres, lo que realmente altera el resultado físico (con un **{pct_tratamiento:.1f}%** de influencia) es cómo lo tratas (enfriamiento, estirado, etc.). Moraleja: ¡Ajusta los procesos en tu fábrica!")
        st.markdown("</div>", unsafe_allow_html=True)
