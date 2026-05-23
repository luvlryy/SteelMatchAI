# -*- coding: utf-8 -*-
"""
==============================================================
PROYECTO INTEGRADOR — STEELMATCH AI
VERSIÓN COMPLETA FINAL + MEJORAS DEL PROYECTO
==============================================================

MEJORAS INCLUIDAS:
✔ Tablas de aceros y tratamientos
✔ Descripciones de tratamientos
✔ Verificación del dataset
✔ Gráfica combinada de propiedades
✔ Boxplots por tratamiento
✔ Interpretaciones automáticas
✔ Explicación de líneas verticales
✔ Comentarios técnicos completos
✔ ANOVA
✔ Recomendador inteligente
✔ Interfaz futurista avanzada
==============================================================
"""

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

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURACIÓN GENERAL
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

# ==========================================================
# CONFIGURACIÓN VISUAL
# ==========================================================

st.set_page_config(
    page_title="SteelMatch AI",
    page_icon="⚙️",
    layout="wide"
)

# ==========================================================
# ESTILO FUTURISTA
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #020617;
    color: white;
}

h1, h2, h3 {
    color: #06b6d4 !important;
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
}

.stButton button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# DICCIONARIOS
# ==========================================================

TRADUCCIONES_TRATAMIENTOS = {
    "Annealed": "Recocido",
    "Normalized": "Normalizado",
    "Hot Rolled": "Laminado en caliente",
    "Cold Drawn": "Estirado en frío",
    "Quenched": "Templado",
    "Tempered": "Revenido",
    "As Rolled": "Laminado",
    "Other": "Otros"
}

DESCRIPCIONES = {
    "Annealed": "Calentamiento y enfriamiento lento para aumentar ductilidad.",
    "Normalized": "Enfriamiento al aire para mejorar uniformidad.",
    "Hot Rolled": "Procesado en caliente para facilitar deformación.",
    "Cold Drawn": "Deformación en frío para aumentar resistencia.",
    "Quenched": "Enfriamiento rápido para incrementar dureza.",
    "Tempered": "Recalentamiento posterior al temple.",
    "As Rolled": "Estado original del laminado.",
    "Other": "Otros tratamientos."
}

TRADUCCIONES_INVERSAS = {
    v: k for k, v in TRADUCCIONES_TRATAMIENTOS.items()
}

# ==========================================================
# LAYOUT PLOTLY
# ==========================================================

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(color="white"),
)

# ==========================================================
# FUNCIONES
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


# ==========================================================
# AGRUPAR TRATAMIENTOS
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


# ==========================================================
# EXTRAER TEMPERATURA
# ==========================================================

def extraer_temperatura(condicion):

    condicion = str(condicion)

    coincidencia = re.search(r"(\d+(?:\.\d+)?)\s*°\s*[Cc]", condicion)

    if coincidencia:
        return float(coincidencia.group(1))

    return np.nan


# ==========================================================
# CARGAR Y PREPROCESAR DATOS
# ==========================================================

@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo):

    aceros = pd.read_excel(nombre_archivo)

    aceros.columns = aceros.columns.astype(str).str.strip()

    # ======================================================
    # LIMPIEZA DE COLUMNAS
    # ======================================================

    aceros["SAE Grade"] = aceros["SAE Grade"].astype(str)

    aceros["Conditions"] = (
        aceros["Conditions"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ======================================================
    # CÁLCULO DEL %C
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
    # CONVERSIÓN NUMÉRICA
    # ======================================================

    for col in PROPIEDADES_MECANICAS:

        aceros[col] = pd.to_numeric(
            aceros[col],
            errors="coerce"
        )

    # ======================================================
    # AGRUPAR TRATAMIENTOS
    # ======================================================

    aceros["Condition_simple"] = aceros[
        "Conditions"
    ].apply(agrupar_tratamiento)

    # ======================================================
    # EXTRAER TEMPERATURA
    # ======================================================

    aceros["Temp_C"] = aceros[
        "Conditions"
    ].apply(extraer_temperatura)

    return aceros


# ==========================================================
# CARGA DE DATOS
# ==========================================================

aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙️ Configuración")

carbono = st.sidebar.slider(
    "Carbono %",
    0.0,
    1.5,
    0.45
)

uts = st.sidebar.slider(
    "UTS",
    200,
    1500,
    600
)

ys = st.sidebar.slider(
    "YS",
    100,
    1300,
    400
)

dureza = st.sidebar.slider(
    "Dureza",
    50,
    500,
    180
)

elongacion = st.sidebar.slider(
    "Elongación",
    1,
    50,
    20
)

buscar = st.sidebar.button(
    "Buscar acero"
)

# ==========================================================
# TÍTULO
# ==========================================================

st.title("⚙️ SteelMatch AI")

# ==========================================================
# TABS
# ==========================================================

tab_dataset, tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🧾 Dataset",
    "🏠 Inicio",
    "📊 Exploración",
    "🔥 Temperatura",
    "⚖️ ANOVA",
    "🔍 Recomendador"
])

# ==========================================================
# TAB DATASET
# ==========================================================

with tab_dataset:

    st.header("📂 Exploración del Dataset")

    # ======================================================
    # HEAD DEL DATASET
    # ======================================================

    st.subheader("Primeras filas")

    st.dataframe(aceros.head())

    # ======================================================
    # NOMBRES DE COLUMNAS
    # ======================================================

    st.subheader("Columnas")

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

    st.dataframe(tabla_aceros)

    # ======================================================
    # TABLA DE TRATAMIENTOS
    # ======================================================

    st.subheader("🔥 Tratamientos")

    tabla_trat = (
        aceros.groupby("Condition_simple")
        .size()
        .reset_index(name="Número de registros")
    )

    tabla_trat["Descripción"] = tabla_trat[
        "Condition_simple"
    ].map(DESCRIPCIONES)

    st.dataframe(tabla_trat)

# ==========================================================
# TAB INICIO
# ==========================================================

with tab_inicio:

    st.header("🏠 Introducción")

    st.markdown("""
    Este proyecto analiza cómo:
    
    - el porcentaje de carbono
    - el tratamiento térmico
    
    afectan las propiedades mecánicas de los aceros.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Registros",
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

    st.header("📊 Propiedades vs Carbono")

    # ======================================================
    # GRÁFICA COMBINADA
    # ======================================================

    st.subheader(
        "📈 Todas las Propiedades Mecánicas"
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
        use_container_width=True
    )

    # ======================================================
    # INTERPRETACIÓN
    # ======================================================

    st.markdown("""
    ### 🧠 Interpretación

    - UTS aumenta con el carbono.
    - YS aumenta con el carbono.
    - La dureza aumenta con el carbono.
    - La elongación disminuye.

    El acero se vuelve:
    - más resistente
    - más duro
    - menos dúctil
    """)

    # ======================================================
    # PROPIEDAD INDIVIDUAL
    # ======================================================

    prop = st.selectbox(
        "Selecciona propiedad",
        PROPIEDADES_MECANICAS
    )

    fig = px.scatter(
        aceros.dropna(subset=["%C", prop]),
        x="%C",
        y=prop,
        color="Condition_simple",
        trendline="ols"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig),
        use_container_width=True
    )

    # ======================================================
    # BOXPLOT
    # ======================================================

    st.subheader(
        "📦 Boxplot por Tratamiento"
    )

    fig_box = px.box(
        aceros.dropna(subset=[prop]),
        x="Condition_simple",
        y=prop,
        color="Condition_simple"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_box),
        use_container_width=True
    )

    st.markdown("""
    ### 📘 Interpretación del Boxplot

    - Temple → mayor dureza.
    - Recocido → mayor ductilidad.
    - Cajas altas → mayor variabilidad.
    - Outliers → valores atípicos.
    """)

# ==========================================================
# TAB TEMPERATURA
# ==========================================================

with tab_temp:

    st.header("🔥 Influencia de Temperatura")

    prop_temp = st.selectbox(
        "Propiedad",
        PROPIEDADES_MECANICAS,
        key="temp"
    )

    df_temp = aceros.dropna(
        subset=["Temp_C", prop_temp]
    )

    # ======================================================
    # ANNEALED
    # ======================================================

    st.subheader("Annealed")

    df_ann = df_temp[
        df_temp["Condition_simple"] == "Annealed"
    ]

    fig_ann = px.scatter(
        df_ann,
        x="Temp_C",
        y=prop_temp,
        color="%C",
        trendline="ols"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_ann),
        use_container_width=True
    )

    # ======================================================
    # NORMALIZED
    # ======================================================

    st.subheader("Normalized")

    df_norm = df_temp[
        df_temp["Condition_simple"] == "Normalized"
    ]

    fig_norm = px.scatter(
        df_norm,
        x="Temp_C",
        y=prop_temp,
        color="%C",
        trendline="ols"
    )

    st.plotly_chart(
        aplicar_layout_estetico(fig_norm),
        use_container_width=True
    )

    # ======================================================
    # EXPLICACIÓN
    # ======================================================

    st.markdown("""
    ### 🧠 Explicación de líneas verticales

    Varias muestras fueron tratadas
    exactamente a la misma temperatura,
    pero tienen composiciones químicas distintas.

    Por eso aparecen líneas verticales:
    - misma temperatura
    - diferentes propiedades
    """)

# ==========================================================
# TAB ANOVA
# ==========================================================

with tab_anova:

    st.header("⚖️ Influencia Estadística")

    prop_anova = st.selectbox(
        "Propiedad para ANOVA",
        PROPIEDADES_MECANICAS,
        key="anova"
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

    pct_carbono = tabla.loc[
        "Carbono",
        "Influencia_%"
    ]

    pct_trat = tabla.loc[
        "C(Tratamiento)",
        "Influencia_%"
    ]

    fig_pie = go.Figure(
        data=[go.Pie(
            labels=[
                "Carbono",
                "Tratamiento",
                "Residual"
            ],
            values=[
                pct_carbono,
                pct_trat,
                tabla.loc["Residual", "Influencia_%"]
            ],
            hole=0.5
        )]
    )

    st.plotly_chart(fig_pie)

    # ======================================================
    # INTERPRETACIÓN
    # ======================================================

    if pct_carbono > pct_trat:

        st.success(
            "El carbono influye más."
        )

    else:

        st.success(
            "El tratamiento influye más."
        )

# ==========================================================
# TAB RECOMENDADOR
# ==========================================================

with tab_recomendador:

    st.header("🔍 Recomendador Inteligente")

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
            "Acero encontrado."
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

        st.markdown("""
        ### 🧠 Conclusión del Matching

        El sistema encontró el acero
        más cercano a los requisitos
        solicitados utilizando distancia
        euclidiana normalizada.
        """)
