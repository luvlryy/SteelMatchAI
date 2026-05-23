# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
VERSIÓN FINAL COMPLETA
Mantiene interfaz futurista original + mejoras requeridas
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

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

warnings.filterwarnings("ignore", category=UserWarning)

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

# ==========================================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================================

st.set_page_config(
    page_title="SteelMatch AI | Quantum Materials",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# TRADUCCIONES
# ==========================================================

TRADUCCIONES_TRATAMIENTOS = {
    "Annealed": "Recocido (Enfriado Lento)",
    "Normalized": "Normalizado (Enfriado al Aire)",
    "Hot Rolled": "Laminado en Caliente",
    "Cold Drawn": "Estirado en Frío",
    "Quenched": "Templado (Enfriado Rápido)",
    "Tempered": "Revenido",
    "As Rolled": "Laminado",
    "Other": "Otros"
}

TRADUCCIONES_INVERSAS = {
    v: k for k, v in TRADUCCIONES_TRATAMIENTOS.items()
}

# ==========================================================
# DESCRIPCIONES DE TRATAMIENTOS
# ==========================================================

DESCRIPCIONES_TRATAMIENTOS = {
    "Annealed": "Calentamiento y enfriamiento lento para aumentar ductilidad.",
    "Normalized": "Enfriamiento al aire para mejorar uniformidad mecánica.",
    "Hot Rolled": "Laminado en caliente para facilitar deformación.",
    "Cold Drawn": "Deformación en frío para incrementar resistencia.",
    "Quenched": "Enfriamiento rápido para aumentar dureza.",
    "Tempered": "Recalentamiento después del temple para reducir fragilidad.",
    "As Rolled": "Estado original después del laminado.",
    "Other": "Tratamientos diversos."
}

# ==========================================================
# LAYOUT PLOTLY
# ==========================================================

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(color="#ffffff", family="sans-serif", size=14),
    title_font=dict(color="#06b6d4", size=20),
    margin=dict(t=60, b=40, l=40, r=40)
)

# ==========================================================
# CSS ORIGINAL COMPLETO
# ==========================================================

st.markdown("""
<style>

.stApp, [data-testid="stSidebar"] {
    background-color: #020617 !important;
}

html, body, [class*="css"], p, span, label, li {
    color: #ffffff !important;
}

h1 {
    color: #06b6d4 !important;
    font-size: 3rem !important;
}

h2 {
    color: #06b6d4 !important;
}

.stButton>button {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    box-shadow: 0 0 15px #06b6d4;
}

.intro-seccion {
    background: rgba(15, 23, 42, 0.6);
    border: 2px solid #06b6d4;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.leyenda-grafica {
    background: rgba(2, 6, 23, 0.8);
    border-left: 5px solid #34d399;
    padding: 15px;
    border-radius: 8px;
}

.interpretacion-box {
    background: rgba(2, 6, 23, 0.8);
    border: 2px solid #34d399;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# FUNCIONES
# ==========================================================

def agrupar_tratamiento(texto):

    texto = str(texto).lower()

    if "annealed" in texto:
        return "Annealed"

    elif "normalized" in texto:
        return "Normalized"

    elif "hot rolled" in texto:
        return "Hot Rolled"

    elif "cold drawn" in texto:
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

    coincidencia = re.search(
        r"(\d+(?:\.\d+)?)\s*°\s*[Cc]",
        condicion
    )

    if coincidencia:
        return float(coincidencia.group(1))

    return np.nan


def aplicar_layout_estetico(fig):

    fig.update_layout(**PLOTLY_LAYOUT)

    fig.update_xaxes(
        gridcolor="rgba(6, 182, 212, 0.1)"
    )

    fig.update_yaxes(
        gridcolor="rgba(6, 182, 212, 0.1)"
    )

    return fig

# ==========================================================
# CARGA Y PREPROCESAMIENTO
# ==========================================================

@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo):

    aceros = pd.read_excel(nombre_archivo)

    aceros.columns = aceros.columns.astype(str).str.strip()

    aceros["SAE Grade"] = aceros["SAE Grade"].astype(str)

    aceros["Conditions"] = (
        aceros["Conditions"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    aceros["C (Min)"] = pd.to_numeric(
        aceros["C (Min)"],
        errors="coerce"
    )

    aceros["C (Max)"] = pd.to_numeric(
        aceros["C (Max)"],
        errors="coerce"
    )

    aceros["%C"] = (
        aceros["C (Min)"] +
        aceros["C (Max)"]
    ) / 2

    for col in PROPIEDADES_MECANICAS:

        aceros[col] = pd.to_numeric(
            aceros[col],
            errors="coerce"
        )

    aceros["Condition_simple"] = aceros[
        "Conditions"
    ].apply(agrupar_tratamiento)

    aceros["Temp_C"] = aceros[
        "Conditions"
    ].apply(extraer_temperatura)

    return aceros

# ==========================================================
# CARGA DE DATOS
# ==========================================================

aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

# ==========================================================
# SIDEBAR ORIGINAL
# ==========================================================

st.sidebar.header("⚙️ Configuración del Protocolo")

carbono = st.sidebar.slider(
    "Concentración de Carbono (%)",
    0.0,
    1.5,
    0.45,
    0.01
)

uts = st.sidebar.slider(
    "Resistencia Máxima (MPa)",
    200,
    1500,
    600
)

ys = st.sidebar.slider(
    "Límite Elástico (MPa)",
    100,
    1300,
    400
)

dureza = st.sidebar.slider(
    "Dureza (HB)",
    50,
    500,
    180
)

elongacion = st.sidebar.slider(
    "Elongación (%)",
    1,
    50,
    20
)

buscar = st.sidebar.button(
    "🚀 Ejecutar Algoritmo"
)

# ==========================================================
# TÍTULO
# ==========================================================

st.title(
    "SteelMatch AI // Características de los Aceros"
)

# ==========================================================
# TABS
# ==========================================================

tab_dataset, tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🧾 Dataset",
    "🏠 Módulo Central",
    "📊 Matriz Química",
    "🔥 Protocolo Térmico",
    "⚖️ Control Variables",
    "🔍 Requisitor"
])

# ==========================================================
# TAB DATASET
# ==========================================================

with tab_dataset:

    st.markdown("""
    <div class="intro-seccion">
        <h4>📂 Exploración General del Dataset</h4>
        <p>Verificación del dataset, tipos de aceros y tratamientos térmicos.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Primeras filas del dataset")

    st.dataframe(aceros.head())

    st.subheader("Nombres de columnas")

    st.write(aceros.columns.tolist())

    # ======================================================
    # TABLA DE ACEROS
    # ======================================================

    st.subheader("🛠️ Lista de Aceros SAE")

    tabla_aceros = (
        aceros.groupby("SAE Grade")
        .size()
        .reset_index(name="Número de registros")
        .sort_values("Número de registros", ascending=False)
    )

    st.dataframe(
        tabla_aceros,
        use_container_width=True
    )

    # ======================================================
    # TABLA DE TRATAMIENTOS
    # ======================================================

    st.subheader("🔥 Lista de Tratamientos")

    tabla_trat = (
        aceros.groupby("Condition_simple")
        .size()
        .reset_index(name="Número de registros")
    )

    tabla_trat["Descripción"] = tabla_trat[
        "Condition_simple"
    ].map(DESCRIPCIONES_TRATAMIENTOS)

    st.dataframe(
        tabla_trat,
        use_container_width=True
    )

# ==========================================================
# TAB INICIO
# ==========================================================

with tab_inicio:

    st.markdown("""
    <div class="intro-seccion">
        <h4>👋 Bienvenido a SteelMatch AI</h4>
        <p>
        Plataforma interactiva para analizar cómo el carbono
        y los tratamientos térmicos modifican las propiedades
        mecánicas de los aceros SAE.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Aceros Analizados",
        len(aceros)
    )

    c2.metric(
        "Grados SAE",
        aceros["SAE Grade"].nunique()
    )

    c3.metric(
        "Tratamientos",
        aceros["Condition_simple"].nunique()
    )

# ==========================================================
# TAB EXPLORACIÓN
# ==========================================================

with tab_exploracion:

    st.markdown("""
    <div class="intro-seccion">
        <h4>📊 Relación entre Carbono y Propiedades</h4>
        <p>
        Análisis de cómo cambia el comportamiento mecánico
        del acero conforme aumenta el contenido de carbono.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # NUEVA GRÁFICA COMPLETA
    # ======================================================

    st.subheader(
        "📈 Todas las Propiedades Mecánicas vs % Carbono"
    )

    df_multi = aceros.dropna(
        subset=["%C"] + PROPIEDADES_MECANICAS
    ).copy()

    df_plot = pd.DataFrame({
        "%C": df_multi["%C"],
        "UTS": df_multi["UTS (MPa)"],
        "YS": df_multi["YS (MPa)"],
        "Hardness": df_multi["Hardness (HB)"],
        "Elongation x10": df_multi["Elongation (%)"] * 10
    })

    df_long = df_plot.melt(
        id_vars="%C",
        var_name="Propiedad",
        value_name="Valor"
    )

    fig_all = px.scatter(
        df_long,
        x="%C",
        y="Valor",
        color="Propiedad",
        trendline="ols"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_all),
        use_container_width=True,
        theme=None
    )

    # ======================================================
    # RESPUESTAS AUTOMÁTICAS
    # ======================================================

    st.markdown("""
    <div class="interpretacion-box">
    <h3>🧠 Interpretación Automática</h3>

    <p>
    • El UTS y el YS aumentan al incrementar el porcentaje de carbono.<br><br>

    • La dureza también aumenta debido a la mayor presencia
    de cementita en la microestructura.<br><br>

    • La elongación disminuye porque el material se vuelve
    más frágil y menos dúctil.<br><br>

    • Existe una tendencia aproximadamente lineal entre
    carbono y resistencia mecánica.<br><br>

    • El aumento simultáneo de UTS y YS implica un acero
    más resistente a deformaciones y fractura.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # GRÁFICA INDIVIDUAL ORIGINAL
    # ======================================================

    prop_c = st.selectbox(
        "Selecciona propiedad:",
        PROPIEDADES_MECANICAS
    )

    fig_c = px.scatter(
        aceros.dropna(subset=["%C", prop_c]),
        x="%C",
        y=prop_c,
        color="Condition_simple",
        trendline="ols"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_c),
        use_container_width=True,
        theme=None
    )

    # ======================================================
    # NUEVO BOXPLOT
    # ======================================================

    st.subheader(
        "📦 Boxplot por Tratamiento"
    )

    fig_box = px.box(
        aceros.dropna(subset=[prop_c]),
        x="Condition_simple",
        y=prop_c,
        color="Condition_simple"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_box),
        use_container_width=True,
        theme=None
    )

    st.markdown("""
    <div class="interpretacion-box">
    <h3>📘 Interpretación del Boxplot</h3>

    <p>
    • Los tratamientos templados presentan
    mayores valores de dureza y resistencia.<br><br>

    • Los recocidos muestran mayor elongación
    y menor dureza.<br><br>

    • Una caja más alta indica mayor variabilidad
    del proceso.<br><br>

    • Los puntos aislados representan outliers.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB TEMPERATURA
# ==========================================================

with tab_temp:

    st.markdown("""
    <div class="intro-seccion">
        <h4>🔥 Influencia de Temperatura</h4>
        <p>
        Relación entre temperatura de tratamiento
        y propiedades mecánicas.
        </p>
    </div>
    """, unsafe_allow_html=True)

    prop_t = st.selectbox(
        "Propiedad:",
        PROPIEDADES_MECANICAS,
        key="temp_prop"
    )

    df_temp = aceros.dropna(
        subset=["Temp_C", prop_t]
    )

    # ======================================================
    # ANNEALED
    # ======================================================

    st.subheader("🔵 Annealed")

    df_ann = df_temp[
        df_temp["Condition_simple"] == "Annealed"
    ]

    fig_a = px.scatter(
        df_ann,
        x="Temp_C",
        y=prop_t,
        color="%C",
        trendline="ols"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_a),
        use_container_width=True,
        theme=None
    )

    # ======================================================
    # NORMALIZED
    # ======================================================

    st.subheader("🟣 Normalized")

    df_norm = df_temp[
        df_temp["Condition_simple"] == "Normalized"
    ]

    fig_n = px.scatter(
        df_norm,
        x="Temp_C",
        y=prop_t,
        color="%C",
        trendline="ols"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_n),
        use_container_width=True,
        theme=None
    )

    # ======================================================
    # EXPLICACIÓN NUEVA
    # ======================================================

    st.markdown("""
    <div class="interpretacion-box">
    <h3>🧠 Explicación de Líneas Verticales</h3>

    <p>
    Las líneas verticales aparecen porque varios aceros
    fueron tratados exactamente a la misma temperatura,
    pero poseen diferentes composiciones químicas (%C).<br><br>

    Esto demuestra que:
    <br><br>

    • La temperatura NO es el único factor importante.<br>
    • El contenido de carbono también modifica
    las propiedades mecánicas.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB ANOVA
# ==========================================================

with tab_anova:

    st.markdown("""
    <div class="intro-seccion">
        <h4>⚖️ Influencia Estadística (ANOVA)</h4>
        <p>
        Comparación entre influencia del carbono
        y tratamiento térmico.
        </p>
    </div>
    """, unsafe_allow_html=True)

    prop_anova = st.selectbox(
        "Propiedad:",
        PROPIEDADES_MECANICAS,
        key="anova_prop"
    )

    df_anova = aceros.dropna(
        subset=["%C", prop_anova]
    ).copy()

    df_anova = df_anova.rename(columns={
        "%C": "Carbono",
        prop_anova: "Propiedad",
        "Condition_simple": "Tratamiento"
    })

    modelo = ols(
        "Propiedad ~ Carbono + C(Tratamiento)",
        data=df_anova
    ).fit()

    tabla = sm.stats.anova_lm(
        modelo,
        typ=2
    )

    tabla["Influencia_%"] = (
        tabla["sum_sq"] /
        tabla["sum_sq"].sum()
    ) * 100

    st.dataframe(tabla)

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Carbono", "Tratamiento", "Residual"],
        values=tabla["Influencia_%"],
        hole=0.5
    )])

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# ==========================================================
# TAB RECOMENDADOR
# ==========================================================

with tab_recomendador:

    st.markdown("""
    <div class="intro-seccion">
        <h4>🔍 Recomendador Inteligente</h4>
        <p>
        Sistema de matching para encontrar
        el acero más parecido a los requisitos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    datos_modelo = aceros.dropna(
        subset=COLUMNAS_MODELO
    )

    scaler = MinMaxScaler()

    scaler.fit(
        datos_modelo[COLUMNAS_MODELO]
    )

    if buscar:

        usr = np.array([[
            carbono,
            uts,
            ys,
            dureza,
            elongacion
        ]])

        usr_norm = scaler.transform(usr)

        mat_norm = scaler.transform(
            datos_modelo[COLUMNAS_MODELO]
        )

        dist = euclidean_distances(
            usr_norm,
            mat_norm
        )

        idx = np.argmin(dist)

        match = datos_modelo.iloc[idx]

        similitud = 100 / (
            1 + dist[0][idx]
        )

        st.success(
            "🎉 Matching exitoso."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "SAE Grade",
            match["SAE Grade"]
        )

        c2.metric(
            "Compatibilidad",
            f"{similitud:.1f}%"
        )

        c3.metric(
            "% Carbono",
            f"{match['%C']:.2f}"
        )
