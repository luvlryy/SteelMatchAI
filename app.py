# ==========================================================
# STEELMATCH AI — VERSIÓN FINAL GITHUB
# Proyecto Integrador de Materiales
# ==========================================================
# Autor: Equipo de Proyecto
# Descripción:
# Plataforma interactiva para analizar cómo el contenido
# de carbono y los tratamientos térmicos afectan
# las propiedades mecánicas de los aceros SAE.
# ==========================================================

# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================

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

warnings.filterwarnings("ignore")

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
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="SteelMatch AI | Quantum Materials",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# DICCIONARIOS DE TRADUCCIÓN
# ==========================================================

TRADUCCIONES_TRATAMIENTOS = {
    "Annealed": "Recocido (Enfriado Lento)",
    "Normalized": "Normalizado (Enfriado al Aire)",
    "Hot Rolled": "Laminado en Caliente",
    "Cold Drawn": "Estirado en Frío",
    "Quenched": "Templado",
    "Tempered": "Revenido",
    "As Rolled": "Laminado",
    "Other": "Otros"
}

TRADUCCIONES_INVERSAS = {
    v: k for k, v in TRADUCCIONES_TRATAMIENTOS.items()
}

DESCRIPCIONES_TRATAMIENTOS = {
    "Annealed": "Enfriamiento lento para aumentar ductilidad.",
    "Normalized": "Enfriamiento al aire para mejorar uniformidad mecánica.",
    "Hot Rolled": "Laminado en caliente para facilitar deformación.",
    "Cold Drawn": "Deformación en frío para incrementar resistencia.",
    "Quenched": "Enfriamiento rápido para aumentar dureza.",
    "Tempered": "Recalentamiento posterior al temple.",
    "As Rolled": "Estado original del laminado.",
    "Other": "Otros tratamientos."
}

# ==========================================================
# ESTILO VISUAL FUTURISTA
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

h2, h3, h4 {
    color: #06b6d4 !important;
}

.stButton>button {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: none;
    padding: 0.7rem 1.5rem;
}

.stButton>button:hover {
    box-shadow: 0 0 20px #06b6d4;
}

.intro-seccion {
    background: rgba(15, 23, 42, 0.7);
    border: 2px solid #06b6d4;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
}

.interpretacion-box {
    background: rgba(2, 6, 23, 0.85);
    border: 2px solid #34d399;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

.grid-conclusion {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-top: 20px;
}

.card-conclusion-item {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid #06b6d4;
    border-radius: 12px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LAYOUT DE PLOTLY
# ==========================================================

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(color="#ffffff"),
    margin=dict(t=60, b=40, l=40, r=40)
)

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def aplicar_layout_estetico(fig):

    fig.update_layout(**PLOTLY_LAYOUT)

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.1)"
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.1)"
    )

    return fig


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

# ==========================================================
# CARGA Y PREPROCESAMIENTO DEL DATASET
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

    # ======================================================
    # CÁLCULO DEL PORCENTAJE DE CARBONO
    # ======================================================

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

    # ======================================================
    # CONVERSIÓN NUMÉRICA DE PROPIEDADES
    # ======================================================

    for col in PROPIEDADES_MECANICAS:

        aceros[col] = pd.to_numeric(
            aceros[col],
            errors="coerce"
        )

    # ======================================================
    # AGRUPACIÓN DE TRATAMIENTOS
    # ======================================================

    aceros["Condition_simple"] = aceros[
        "Conditions"
    ].apply(agrupar_tratamiento)

    # ======================================================
    # EXTRACCIÓN DE TEMPERATURA
    # ======================================================

    aceros["Temp_C"] = aceros[
        "Conditions"
    ].apply(extraer_temperatura)

    return aceros

# ==========================================================
# CARGA DEL DATASET
# ==========================================================

aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

# ==========================================================
# SIDEBAR
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
# TÍTULO PRINCIPAL
# ==========================================================

st.title("⚙️ SteelMatch AI // Quantum Materials")

# ==========================================================
# TABS PRINCIPALES
# ==========================================================

tab_dataset, tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🧾 Dataset",
    "🏠 Módulo Central",
    "📊 Matriz Química",
    "🔥 Protocolo Térmico",
    "⚖️ Control de Variables",
    "🔍 Requisitor"
])

# ==========================================================
# TAB DATASET
# ==========================================================

with tab_dataset:

    st.markdown("""
    <div class="intro-seccion">
        <h4>📂 Exploración General del Dataset</h4>

        <p>
        Este módulo permite validar el dataset utilizado
        y explorar los diferentes tipos de aceros
        y tratamientos térmicos presentes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📄 Primeras filas del dataset")

    st.dataframe(aceros.head())

    st.subheader("🧾 Columnas disponibles")

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

        <h4>👋 Bienvenido al Proyecto SteelMatch AI</h4>

        <p>
        Este proyecto fue desarrollado para analizar cómo
        el porcentaje de carbono (%C) y los tratamientos
        térmicos modifican las propiedades mecánicas
        de los aceros SAE.
        </p>

        <h4>🎯 Objetivos</h4>

        <ul>
            <li>Analizar la relación entre composición química y propiedades mecánicas.</li>
            <li>Evaluar el efecto de los tratamientos térmicos.</li>
            <li>Comparar resistencia, dureza y ductilidad.</li>
            <li>Aplicar ANOVA para identificar el factor dominante.</li>
        </ul>

        <h4>🧪 Metodología</h4>

        <ul>
            <li>Limpieza y procesamiento del dataset.</li>
            <li>Cálculo del porcentaje promedio de carbono.</li>
            <li>Visualización de tendencias mediante gráficas.</li>
            <li>Análisis estadístico con ANOVA.</li>
            <li>Desarrollo de un sistema de recomendación inteligente.</li>
        </ul>

        <h4>🧠 Hipótesis</h4>

        <ul>
            <li>El aumento del carbono incrementa resistencia y dureza.</li>
            <li>La ductilidad disminuye al aumentar el carbono.</li>
            <li>Los tratamientos térmicos alteran significativamente la microestructura.</li>
        </ul>

        <h4>📈 Resultados Generales</h4>

        <ul>
            <li>El UTS y YS aumentan con el %C.</li>
            <li>La dureza incrementa con tratamientos de temple.</li>
            <li>La elongación disminuye conforme aumenta la resistencia.</li>
            <li>El ANOVA mostró qué factor tiene mayor influencia.</li>
        </ul>

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
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # GRÁFICA GENERAL
    # ======================================================

    st.subheader("📈 Todas las Propiedades Mecánicas")

    df_multi = aceros.dropna(
        subset=["%C"] + PROPIEDADES_MECANICAS
    ).copy()

    df_plot = pd.DataFrame({
        "%C": df_multi["%C"],
        "UTS (MPa)": df_multi["UTS (MPa)"],
        "YS (MPa)": df_multi["YS (MPa)"],
        "Hardness (HB)": df_multi["Hardness (HB)"],
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
        use_container_width=True
    )

    st.markdown("""
    <div class="interpretacion-box">

    <h3>🧠 Interpretación General</h3>

    <p>
    • El UTS y YS aumentan conforme aumenta el carbono.<br><br>

    • La dureza también incrementa debido a la formación
    de fases más resistentes.<br><br>

    • La elongación disminuye porque el material
    se vuelve menos dúctil.<br><br>

    • Existe un balance entre resistencia y ductilidad.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # GRÁFICA INDIVIDUAL
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
        use_container_width=True
    )

    # ======================================================
    # BOXPLOT
    # ======================================================

    st.subheader("📦 Distribución por Tratamiento")

    fig_box = px.box(
        aceros.dropna(subset=[prop_c]),
        x="Condition_simple",
        y=prop_c,
        color="Condition_simple"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_box),
        use_container_width=True
    )

# ==========================================================
# TAB TEMPERATURA
# ==========================================================

with tab_temp:

    prop_t = st.selectbox(
        "Propiedad:",
        PROPIEDADES_MECANICAS,
        key="temp_prop"
    )

    df_temp = aceros.dropna(
        subset=["Temp_C", prop_t]
    )

    st.subheader("Normalized")

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
        use_container_width=True
    )

    st.markdown("""
    <div class="interpretacion-box">

    <h3>🧠 Explicación de Líneas Verticales</h3>

    <p>
    Las líneas verticales aparecen porque varios aceros
    fueron normalizados exactamente a la misma temperatura,
    pero poseen diferentes composiciones químicas.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB ANOVA
# ==========================================================

with tab_anova:

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

# ==========================================================
# TAB RECOMENDADOR
# ==========================================================

with tab_recomendador:

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

        st.success("🎉 Matching exitoso.")

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
