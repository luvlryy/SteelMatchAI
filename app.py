# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
Versión mejorada con:
  - Tema oscuro consistente en todas las gráficas Plotly
  - Extracción de temperatura más robusta (evita capturar SAE Grades)
  - ANOVA con aviso de R² bajo y residual separado en gráfica
  - Conclusión dedicada al final del Tab ANOVA
  - Estadísticas descriptivas en expanders
  - Validación de rango realista de temperatura (100–1300 °C)
  - Estética mejorada con CSS personalizado

Cómo ejecutar:
1. pip install streamlit pandas numpy scikit-learn plotly statsmodels openpyxl xlrd
2. Colocar este archivo junto al CSV/Excel del dataset o subirlo desde la app
3. streamlit run proyectoim_final.py
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

# ── MEJORA: paleta oscura global para todas las gráficas Plotly ──────────────
PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(15,23,42,0)",   # fondo transparente (mismo que .main)
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#e2e8f0", family="monospace"),
    title_font=dict(color="#38bdf8", size=18),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#334155", borderwidth=1),
)
# ─────────────────────────────────────────────────────────────────────────────


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
# ESTILO VISUAL — MEJORADO
# Cambios: gradiente en header, cards con glassmorphism,
# badges de color, tipografía monoespaciada para datos,
# animación de fade-in en secciones principales.
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

    /* ── Fondo general ── */
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

    /* ── Títulos ── */
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

    /* ── Botones ── */
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

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }

    /* ── Métricas ── */
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

    /* ── Cards de hallazgos ── */
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

    /* ── Badge de factor dominante ── */
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

    /* ── Separador ── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
        margin: 2rem 0;
    }

    /* ── Tabs ── */
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

    /* ── Dataframes ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(56,189,248,0.15);
    }

    /* ── Alerts / Info / Warning ── */
    .stInfo, .stSuccess, .stWarning {
        border-radius: 12px;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 10px;
        color: #e2e8f0;
    }

    /* ── Conclusión section ── */
    .conclusion-box {
        background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(99,102,241,0.08));
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-top: 2rem;
        animation: fadeInUp 0.6s ease forwards;
    }

    /* ── Propiedad badge actual ── */
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
</style>
""", unsafe_allow_html=True)


# ==========================================================
# FUNCIONES DE LIMPIEZA Y PREPROCESAMIENTO
# ==========================================================

def agrupar_tratamiento(texto):
    """
    Agrupa la columna Conditions en tratamientos generales.
    Evita que cada temperatura aparezca como una categoría distinta.
    """
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
    """
    MEJORA: Extracción de temperatura más robusta.
    - Solo activa el respaldo para Annealed/Normalized cuando
      el texto contiene explícitamente 'at' + número, evitando
      capturar SAE Grades como 1045 como si fueran °C.
    - Valida que el resultado esté en rango realista (100–1300°C).

    Extrae temperatura en °C desde textos como:
      'Annealed at 815 °C'  -> 815
      'Normalized at 870 C' -> 870
    Si no encuentra temperatura válida, devuelve NaN.
    """
    condicion = str(condicion)

    # Caso principal: número seguido de °C (con símbolo de grado explícito)
    coincidencia = re.search(r"(\d+(?:\.\d+)?)\s*°\s*[Cc]", condicion)
    if coincidencia:
        temp = float(coincidencia.group(1))
        # MEJORA: validar rango físico realista de tratamientos
        if 100 <= temp <= 1300:
            return temp
        return np.nan

    # Respaldo ESTRICTO: solo para Annealed/Normalized con patrón "at <número>"
    # Esto evita que un SAE Grade de 3-4 dígitos (ej. 1045) sea interpretado como temperatura.
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
    """
    MEJORA: Aplica el tema oscuro consistente a cualquier figura Plotly.
    Centraliza la configuración para que todas las gráficas sean coherentes
    con el fondo oscuro de la app (#0f172a).
    """
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(
        gridcolor="rgba(51,65,85,0.5)",
        zerolinecolor="rgba(56,189,248,0.2)"
    )
    fig.update_yaxes(
        gridcolor="rgba(51,65,85,0.5)",
        zerolinecolor="rgba(56,189,248,0.2)"
    )
    return fig


def leer_archivo_datos(nombre_archivo, archivo_bytes=None):
    """
    Lee archivos .csv, .xlsx o .xls.

    Puede leer:
    1. Un archivo local, por ejemplo:
       Steel- Property and Composition (Carbon Steel).csv
       Steel- Property and Composition (Carbon Steel).xlsx

    2. Un archivo subido desde la app con st.file_uploader.

    Requisitos:
    - Para .xlsx: instalar openpyxl
    - Para .xls: instalar xlrd
    """
    extension = nombre_archivo.lower().split(".")[-1]

    def obtener_fuente():
        if archivo_bytes is not None:
            return io.BytesIO(archivo_bytes)
        return nombre_archivo

    if extension == "csv":
        # Se intenta latin1 primero porque el dataset puede traer símbolos como °C.
        # Si falla, se intenta UTF-8.
        errores = []

        for encoding in ["latin1", "utf-8", "utf-8-sig"]:
            try:
                return pd.read_csv(obtener_fuente(), sep=",", encoding=encoding)
            except Exception as e:
                errores.append(f"{encoding}: {e}")

        st.error(
            "No se pudo leer el CSV con codificación latin1, utf-8 ni utf-8-sig. "
            "Revisa que el archivo no esté dañado."
        )

        with st.expander("Ver errores de lectura del CSV"):
            for error in errores:
                st.write(error)

        st.stop()

    elif extension in ["xlsx", "xls"]:
        try:
            return pd.read_excel(obtener_fuente())
        except ImportError:
            if extension == "xlsx":
                st.error(
                    "Para leer archivos .xlsx necesitas instalar openpyxl. "
                    "Ejecuta: pip install openpyxl"
                )
            else:
                st.error(
                    "Para leer archivos .xls necesitas instalar xlrd. "
                    "Ejecuta: pip install xlrd"
                )
            st.stop()
        except Exception as e:
            st.error(f"No se pudo leer el archivo de Excel: {e}")
            st.stop()

    else:
        st.error("Formato no compatible. Usa un archivo .csv, .xlsx o .xls.")
        st.stop()


@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo, archivo_bytes=None):
    """
    Carga CSV o Excel, limpia columnas principales y crea:
    - %C
    - Condition_simple
    - Temp_C

    El archivo debe contener estas columnas:
    SAE Grade, Conditions, C (Min), C (Max),
    UTS (MPa), YS (MPa), Hardness (HB), Elongation (%).
    """

    try:
        aceros = leer_archivo_datos(nombre_archivo, archivo_bytes)
    except FileNotFoundError:
        st.error(
            f"No se encontró el archivo '{nombre_archivo}'. "
            "Coloca el archivo de datos en la misma carpeta que este archivo de Streamlit, "
            "o súbelo desde el cargador de archivos de la barra lateral."
        )
        st.stop()

    # Limpiar nombres de columnas para evitar errores por espacios accidentales
    aceros.columns = aceros.columns.astype(str).str.strip()

    if "SAE Grade" in aceros.columns:
        aceros["SAE Grade"] = aceros["SAE Grade"].astype(str).str.strip()
    else:
        st.error("No se encontró la columna 'SAE Grade' en el archivo.")
        st.stop()

    columnas_necesarias = [
        "Conditions", "C (Min)", "C (Max)",
        "UTS (MPa)", "YS (MPa)", "Hardness (HB)", "Elongation (%)"
    ]

    faltantes = [col for col in columnas_necesarias if col not in aceros.columns]
    if faltantes:
        st.error(f"Faltan columnas necesarias en el archivo: {faltantes}")
        st.write("Columnas encontradas:")
        st.write(list(aceros.columns))
        st.stop()

    aceros["Conditions"] = (
        aceros["Conditions"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    aceros["C (Min)"] = pd.to_numeric(aceros["C (Min)"], errors="coerce")
    aceros["C (Max)"] = pd.to_numeric(aceros["C (Max)"], errors="coerce")
    aceros["%C"] = (aceros["C (Min)"] + aceros["C (Max)"]) / 2

    for col in PROPIEDADES_MECANICAS:
        aceros[col] = pd.to_numeric(aceros[col], errors="coerce")

    aceros["Condition_simple"] = aceros["Conditions"].apply(agrupar_tratamiento)
    aceros["Temp_C"] = aceros["Conditions"].apply(extraer_temperatura)

    return aceros


def interpretar_propiedad(propiedad):
    """Devuelve una explicación breve según la propiedad seleccionada."""
    if propiedad in ["UTS (MPa)", "YS (MPa)", "Hardness (HB)"]:
        return (
            "Para resistencia y dureza, valores mayores indican un material más fuerte "
            "o más resistente a la deformación, aunque normalmente con menor ductilidad."
        )
    elif propiedad == "Elongation (%)":
        return (
            "La elongación mide ductilidad. Valores mayores indican que el acero puede "
            "deformarse más antes de romperse."
        )
    return ""


def stats_card(df, col_num, label):
    """
    MEJORA: Genera estadísticas descriptivas (media, min, max, std)
    para mostrar en expanders en lugar de solo tabla cruda.
    """
    s = df[col_num].dropna()
    return {
        "Propiedad": label,
        "Media": f"{s.mean():.2f}",
        "Mín": f"{s.min():.2f}",
        "Máx": f"{s.max():.2f}",
        "Desv. Est.": f"{s.std():.2f}",
        "Registros": len(s)
    }


# ==========================================================
# CARGA DE DATOS
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Archivo de datos")

archivo_subido = st.sidebar.file_uploader(
    "Sube el dataset en CSV o Excel",
    type=["csv", "xlsx", "xls"]
)

if archivo_subido is not None:
    aceros = cargar_y_preprocesar_datos(
        archivo_subido.name,
        archivo_subido.getvalue()
    )
    st.sidebar.success(f"Archivo cargado: {archivo_subido.name}")
else:
    aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)
    st.sidebar.info(
        f"Usando archivo local: {DEFAULT_DATA_FILE}. "
        "También puedes subir un .csv, .xlsx o .xls desde aquí."
    )


# ==========================================================
# TÍTULO PRINCIPAL
# ==========================================================

st.title("⚙️ SteelMatch AI")
st.subheader("Análisis de propiedades mecánicas en aceros al carbono")

st.write("""
Esta web analiza cómo el porcentaje de carbono, el tratamiento térmico o mecánico
y la temperatura del tratamiento influyen en las propiedades mecánicas de aceros al carbono.
""")


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

    st.subheader("Columnas creadas en la limpieza")
    st.write("""
    - **%C**: promedio entre `C (Min)` y `C (Max)`.
    - **Condition_simple**: tratamiento agrupado.
    - **Temp_C**: temperatura extraída desde `Conditions` (validada entre 100 y 1300 °C).
    """)


# ==========================================================
# TAB 2 — EXPLORACIÓN
# ==========================================================

with tab_exploracion:
    st.header("Exploración inicial de aceros y tratamientos")

    st.subheader("Tabla de aceros")

    tabla_aceros = aceros["SAE Grade"].value_counts().reset_index()
    tabla_aceros.columns = ["SAE Grade", "Cantidad de datos"]
    st.dataframe(tabla_aceros, use_container_width=True)

    st.subheader("Tabla de tratamientos")

    tabla_tratamientos = aceros["Condition_simple"].value_counts().reset_index()
    tabla_tratamientos.columns = ["Tratamiento", "Cantidad de datos"]

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

    tabla_tratamientos["Descripción"] = tabla_tratamientos["Tratamiento"].map(descripciones)
    st.dataframe(tabla_tratamientos, use_container_width=True)

    st.subheader("Propiedades mecánicas vs %C")

    propiedad_carbono = st.selectbox(
        "Selecciona una propiedad para graficar contra %C",
        PROPIEDADES_MECANICAS,
        key="propiedad_carbono"
    )

    datos_carbono = aceros.dropna(subset=["%C", propiedad_carbono, "Condition_simple"])

    # MEJORA: tema oscuro aplicado con aplicar_layout_plotly()
    fig_carbono = px.scatter(
        datos_carbono,
        x="%C",
        y=propiedad_carbono,
        color="Condition_simple",
        hover_data=["SAE Grade", "Conditions"],
        trendline="ols",
        title=f"{propiedad_carbono} vs %C por tratamiento",
        labels={
            "%C": "Porcentaje de carbono promedio",
            propiedad_carbono: propiedad_carbono,
            "Condition_simple": "Tratamiento"
        }
    )
    fig_carbono = aplicar_layout_plotly(fig_carbono)
    st.plotly_chart(fig_carbono, use_container_width=True)

    st.info("""
    En general, al aumentar el contenido de carbono suelen aumentar la resistencia
    y la dureza, mientras que la ductilidad tiende a disminuir. El carbono dificulta
    el movimiento de dislocaciones y favorece estructuras más resistentes, pero menos deformables.
    """)


# ==========================================================
# TAB 3 — PARTE A: TEMPERATURA
# MEJORA: badge de propiedad activa, estadísticas en expander,
#         tema oscuro en gráficas, validación de datos vacíos mejorada.
# ==========================================================

with tab_temp:
    st.header("Parte A — Pregunta 3: Influencia de la temperatura")

    st.write("""
    Se analizan tratamientos con temperatura explícita: **Annealed** y **Normalized**.
    La columna `Temp_C` se extrae desde `Conditions` y se valida entre **100 y 1300 °C**
    para evitar capturar valores fuera de rango (como SAE Grades de 4 dígitos).
    """)

    propiedad_temp = st.selectbox(
        "Selecciona la propiedad mecánica para analizar contra temperatura",
        PROPIEDADES_MECANICAS,
        key="propiedad_temp"
    )

    # MEJORA: badge visual que indica qué propiedad está activa
    st.markdown(
        f'<span class="prop-badge">Analizando: {propiedad_temp}</span>',
        unsafe_allow_html=True
    )

    datos_temp = aceros.dropna(subset=["Temp_C", propiedad_temp, "Condition_simple"])

    # ─── Annealed ───────────────────────────────────────────────────────────────

    st.subheader("🔵 Annealed — Propiedad mecánica vs temperatura")

    annealed = datos_temp[datos_temp["Condition_simple"] == "Annealed"]

    if not annealed.empty:
        # MEJORA: tema oscuro consistente
        fig_annealed = px.scatter(
            annealed,
            x="Temp_C",
            y=propiedad_temp,
            color="%C",
            hover_data=["SAE Grade", "%C", "Conditions"],
            trendline="ols",
            title=f"{propiedad_temp} vs Temperatura — Annealed",
            labels={
                "Temp_C": "Temperatura del tratamiento (°C)",
                propiedad_temp: propiedad_temp,
                "%C": "% Carbono"
            },
            color_continuous_scale="Blues"
        )
        fig_annealed = aplicar_layout_plotly(fig_annealed)
        st.plotly_chart(fig_annealed, use_container_width=True)

        st.info(f"""
        **Interpretación metalúrgica — Annealed:**  
        {interpretar_propiedad(propiedad_temp)}

        En el recocido, al subir la temperatura normalmente **disminuyen resistencia y dureza**
        porque el acero libera esfuerzos internos, puede haber crecimiento de grano y se favorece
        una microestructura más blanda y estable (ferrita + perlita gruesa).  
        La **ductilidad (elongación) suele aumentar** porque el material se vuelve más deformable
        y menos frágil.
        """)

        # MEJORA: expander con estadísticas descriptivas en lugar de solo tabla cruda
        with st.expander("📋 Estadísticas descriptivas — Annealed"):
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            s = annealed[propiedad_temp].dropna()
            col_e1.metric("Media", f"{s.mean():.1f}")
            col_e2.metric("Mín", f"{s.min():.1f}")
            col_e3.metric("Máx", f"{s.max():.1f}")
            col_e4.metric("Registros", len(s))
            st.dataframe(
                annealed[["SAE Grade", "Conditions", "Temp_C", "%C", propiedad_temp]],
                use_container_width=True
            )
    else:
        st.warning("No hay datos de Annealed con temperatura válida (100–1300 °C).")

    st.markdown("---")

    # ─── Normalized ─────────────────────────────────────────────────────────────

    st.subheader("🟣 Normalized — Propiedad mecánica vs temperatura")

    normalized = datos_temp[datos_temp["Condition_simple"] == "Normalized"]

    if not normalized.empty:
        # MEJORA: tema oscuro consistente
        fig_normalized = px.scatter(
            normalized,
            x="Temp_C",
            y=propiedad_temp,
            color="%C",
            hover_data=["SAE Grade", "%C", "Conditions"],
            trendline="ols",
            title=f"{propiedad_temp} vs Temperatura — Normalized",
            labels={
                "Temp_C": "Temperatura del tratamiento (°C)",
                propiedad_temp: propiedad_temp,
                "%C": "% Carbono"
            },
            color_continuous_scale="Purples"
        )
        fig_normalized = aplicar_layout_plotly(fig_normalized)
        st.plotly_chart(fig_normalized, use_container_width=True)

        st.info("""
        **Interpretación metalúrgica — Normalized:**  
        En el normalizado, la tendencia varía más que en el recocido porque la propiedad final
        depende de la temperatura **y** del porcentaje de carbono y grado SAE.  
        El normalizado refina la microestructura austenítica, pero aceros con diferente composición
        tratados a la misma temperatura producen propiedades distintas (ferrita + perlita fina con
        tamaños de grano diferentes).
        """)

        # MEJORA: explicación de puntos verticales con tabla estadística enriquecida
        st.subheader("📌 ¿Por qué aparecen puntos alineados verticalmente?")

        st.write("""
        Los puntos alineados verticalmente **no son un error**: significan que varios registros
        comparten la misma `Temp_C` pero tienen diferente composición, grado SAE o propiedades
        mecánicas. La temperatura del tratamiento es la misma, pero cada acero responde distinto
        por su microestructura y %C.
        """)

        puntos_verticales = (
            normalized
            .groupby("Temp_C")
            .agg(
                registros=("SAE Grade", "count"),
                aceros_distintos=("SAE Grade", "nunique"),
                carbono_min=("%C", "min"),
                carbono_max=("%C", "max"),
                propiedad_min=(propiedad_temp, "min"),
                propiedad_max=(propiedad_temp, "max")
            )
            .reset_index()
            .sort_values("registros", ascending=False)
        )

        # CORRECCIÓN: se eliminó .background_gradient() porque requiere matplotlib
        # (dependencia opcional no instalada). Se muestra la tabla directamente;
        # el orden descendente por 'registros' ya resalta visualmente los casos más relevantes.
        st.dataframe(puntos_verticales, use_container_width=True)

        # MEJORA: estadísticas en expander
        with st.expander("📋 Estadísticas descriptivas — Normalized"):
            col_n1, col_n2, col_n3, col_n4 = st.columns(4)
            sn = normalized[propiedad_temp].dropna()
            col_n1.metric("Media", f"{sn.mean():.1f}")
            col_n2.metric("Mín", f"{sn.min():.1f}")
            col_n3.metric("Máx", f"{sn.max():.1f}")
            col_n4.metric("Registros", len(sn))
            st.dataframe(
                normalized[["SAE Grade", "Conditions", "Temp_C", "%C", propiedad_temp]],
                use_container_width=True
            )
    else:
        st.warning("No hay datos de Normalized con temperatura válida (100–1300 °C).")


# ==========================================================
# TAB 4 — PARTE B: ANOVA
# MEJORA: residual separado en gráfica, aviso de R² bajo,
#         tema oscuro, conclusión dedicada al final con cards.
# ==========================================================

with tab_anova:
    st.header("Parte B — Pregunta 4: ANOVA")

    st.write("""
    El ANOVA compara qué factor explica más la variación de una propiedad mecánica:
    el **porcentaje de carbono (%C)** o el **tratamiento térmico/mecánico**.
    """)

    propiedad_anova = st.selectbox(
        "Selecciona la propiedad mecánica para el ANOVA",
        PROPIEDADES_MECANICAS,
        key="propiedad_anova"
    )

    # MEJORA: badge de propiedad activa
    st.markdown(
        f'<span class="prop-badge">Analizando: {propiedad_anova}</span>',
        unsafe_allow_html=True
    )

    datos_anova = aceros.dropna(subset=["%C", "Condition_simple", propiedad_anova]).copy()

    if len(datos_anova) > 10 and datos_anova["Condition_simple"].nunique() > 1:

        datos_anova = datos_anova.rename(columns={
            "%C": "Carbono",
            propiedad_anova: "Propiedad",
            "Condition_simple": "Tratamiento"
        })

        modelo = ols("Propiedad ~ Carbono + C(Tratamiento)", data=datos_anova).fit()
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)

        tabla_anova["Influencia_%"] = (
            tabla_anova["sum_sq"] / tabla_anova["sum_sq"].sum()
        ) * 100

        tabla_anova_visual = (
            tabla_anova
            .reset_index()
            .rename(columns={"index": "Factor"})
        )

        # MEJORA: aviso cuando R² es bajo (modelo explica poco la variación total)
        r2 = modelo.rsquared
        if r2 < 0.4:
            st.warning(
                f"⚠️ El R² del modelo es **{r2:.2f}**, lo que indica que los factores "
                "analizados explican una parte limitada de la variación total. "
                "Pueden existir otros factores (composición de aleantes, procesos previos) "
                "que influyen en la propiedad."
            )

        st.subheader("Tabla ANOVA")
        st.dataframe(tabla_anova_visual, use_container_width=True)

        # MEJORA: separar residual de factores reales en la gráfica para no confundir
        tabla_factores = tabla_anova_visual[
            ~tabla_anova_visual["Factor"].str.lower().str.contains("residual")
        ].copy()

        tabla_residual = tabla_anova_visual[
            tabla_anova_visual["Factor"].str.lower().str.contains("residual")
        ].copy()

        # MEJORA: gráfica de factores reales (sin residual) con tema oscuro
        fig_anova = px.bar(
            tabla_factores,
            x="Factor",
            y="Influencia_%",
            title=f"Influencia de cada factor — {propiedad_anova}",
            labels={
                "Factor": "Factor analizado",
                "Influencia_%": "Variación explicada (%)"
            },
            text="Influencia_%",
            color="Factor",
            color_discrete_sequence=["#38bdf8", "#818cf8"]
        )
        fig_anova.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_anova = aplicar_layout_plotly(fig_anova)
        st.plotly_chart(fig_anova, use_container_width=True)

        # MEJORA: nota aclaratoria sobre el residual
        if not tabla_residual.empty:
            residual_pct = tabla_residual["Influencia_%"].values[0]
            st.caption(
                f"ℹ️ El **Residual** ({residual_pct:.1f}%) representa la variación no explicada "
                "por ninguno de los factores del modelo. Se omite de la gráfica para claridad."
            )

        # ─── Hallazgos principales ────────────────────────────────────────────

        factores_sin_residuo = tabla_factores.copy()
        factor_mayor = factores_sin_residuo.sort_values("Influencia_%", ascending=False).iloc[0]
        factor_menor = factores_sin_residuo.sort_values("Influencia_%", ascending=False).iloc[-1]

        # Obtener p-values para interpretación
        p_carbono = tabla_anova_visual.loc[
            tabla_anova_visual["Factor"].str.contains("Carbono", case=False), "PR(>F)"
        ].values
        p_tratamiento = tabla_anova_visual.loc[
            tabla_anova_visual["Factor"].str.contains("Tratamiento", case=False), "PR(>F)"
        ].values

        p_carbono_val = p_carbono[0] if len(p_carbono) > 0 else None
        p_tratamiento_val = p_tratamiento[0] if len(p_tratamiento) > 0 else None

        st.subheader("🏆 Hallazgos principales")

        # MEJORA: cards visuales con resultados clave en lugar de solo st.success
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown(f"""
            <div class="hallazgo-card">
                <h4 style="color:#38bdf8; margin:0 0 8px 0;">Factor dominante</h4>
                <span class="badge-dominante">{factor_mayor['Factor']}</span>
                <p style="margin:12px 0 0 0; color:#94a3b8; font-size:14px;">
                    Explica el <strong style="color:#e2e8f0;">{factor_mayor['Influencia_%']:.1f}%</strong>
                    de la variación en <em>{propiedad_anova}</em>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_h2:
            st.markdown(f"""
            <div class="hallazgo-card">
                <h4 style="color:#818cf8; margin:0 0 8px 0;">Factor secundario</h4>
                <span class="badge-secundario">{factor_menor['Factor']}</span>
                <p style="margin:12px 0 0 0; color:#94a3b8; font-size:14px;">
                    Explica el <strong style="color:#e2e8f0;">{factor_menor['Influencia_%']:.1f}%</strong>
                    de la variación en <em>{propiedad_anova}</em>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # p-values en cards
        st.markdown("<br>", unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)

        def formato_pvalue(p):
            if p is None:
                return "N/D"
            if p < 0.001:
                return "< 0.001 ✅ significativo"
            elif p < 0.05:
                return f"{p:.4f} ✅ significativo"
            else:
                return f"{p:.4f} ⚠️ no significativo"

        with col_p1:
            st.markdown(f"""
            <div class="hallazgo-card">
                <h4 style="color:#38bdf8; margin:0 0 6px 0;">p-value — %C (Carbono)</h4>
                <p style="font-size:16px; color:#e2e8f0; margin:0;">{formato_pvalue(p_carbono_val)}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_p2:
            st.markdown(f"""
            <div class="hallazgo-card">
                <h4 style="color:#818cf8; margin:0 0 6px 0;">p-value — Tratamiento</h4>
                <p style="font-size:16px; color:#e2e8f0; margin:0;">{formato_pvalue(p_tratamiento_val)}</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("""
        **Cómo interpretar el ANOVA:**  
        - Un **p-value < 0.05** indica que el factor tiene efecto estadísticamente significativo.  
        - Una **mayor suma de cuadrados** significa que ese factor explica más variación observada.  
        - Un **mayor F-value** indica que el efecto supera ampliamente al error aleatorio.  
        - Si el **tratamiento domina**: los procesos térmicos/mecánicos cambian fuertemente la propiedad.  
        - Si el **%C domina**: la composición química explica mejor los cambios observados.
        """)

        # MEJORA: sección de conclusión dedicada al final del Tab ANOVA
        st.markdown("---")
        st.subheader("📝 Conclusión — Parte B")

        nombre_factor = factor_mayor["Factor"]
        pct_factor = factor_mayor["Influencia_%"]
        nombre_secundario = factor_menor["Factor"]
        pct_secundario = factor_menor["Influencia_%"]
        r2_pct = r2 * 100

        conclusion_texto = f"""
        Para la propiedad **{propiedad_anova}**, el análisis ANOVA de dos factores indica que
        **{nombre_factor}** es el factor con mayor influencia, explicando aproximadamente
        **{pct_factor:.1f}%** de la variación total observada en el dataset.
        **{nombre_secundario}** contribuye con un **{pct_secundario:.1f}%**, siendo un factor
        {'relevante' if pct_secundario > 15 else 'secundario'} en la variación.

        En conjunto, el modelo explica el **{r2_pct:.1f}%** de la variación total (R²),
        {'lo que indica un buen ajuste para datos metalúrgicos.' if r2 >= 0.5
         else 'lo que sugiere que otros factores no incluidos (aleantes, proceso previo, enfriamiento) también contribuyen.'}
        """

        st.markdown(
            f'<div class="conclusion-box"><p style="line-height:1.8; font-size:15px;">{conclusion_texto}</p></div>',
            unsafe_allow_html=True
        )

    else:
        st.warning(
            "No hay datos suficientes o tratamientos suficientes para ANOVA con esta propiedad."
        )


# ==========================================================
# TAB 5 — RECOMENDADOR
# MEJORA: tema oscuro en gráficas de resultados si se agregan.
# ==========================================================

with tab_recomendador:
    st.header("🔍 Recomendador SteelMatch AI")

    st.write("""
    Ingresa las propiedades mecánicas deseadas y el tratamiento,
    y el sistema encontrará el acero más cercano en el dataset.
    """)

    datos_modelo = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])

    if datos_modelo.empty:
        st.warning("No hay datos suficientes para usar el recomendador.")
        st.stop()

    scaler = MinMaxScaler()
    scaler.fit(datos_modelo[COLUMNAS_MODELO])

    st.sidebar.header("🔍 Parámetros del recomendador")

    carbono = st.sidebar.slider("% de Carbono", 0.0, 1.5, 0.4, 0.01)
    uts = st.sidebar.slider("UTS (MPa)", 200, 1500, 600)
    ys = st.sidebar.slider("YS (MPa)", 100, 1300, 400)
    dureza = st.sidebar.slider("Dureza HB", 50, 500, 180)
    elongacion = st.sidebar.slider("Elongación (%)", 1, 50, 20)

    tratamientos_disponibles = ["Todos"] + sorted(
        datos_modelo["Condition_simple"].dropna().unique().tolist()
    )

    tratamiento = st.sidebar.selectbox("Tratamiento térmico", tratamientos_disponibles)
    buscar = st.sidebar.button("Buscar material")

    if buscar:
        datos_busqueda = datos_modelo.copy()

        if tratamiento != "Todos":
            datos_busqueda = datos_busqueda[
                datos_busqueda["Condition_simple"] == tratamiento
            ]

        if datos_busqueda.empty:
            st.warning("No hay datos disponibles para ese tratamiento.")
            st.stop()

        entrada_usuario = np.array([[carbono, uts, ys, dureza, elongacion]])
        entrada_usuario_normalizada = scaler.transform(entrada_usuario)
        X_busqueda = scaler.transform(datos_busqueda[COLUMNAS_MODELO])

        distancias = euclidean_distances(entrada_usuario_normalizada, X_busqueda)
        indice_mejor = np.argmin(distancias)
        mejor_material = datos_busqueda.iloc[indice_mejor]
        similitud = 100 / (1 + distancias[0][indice_mejor])

        st.success("✅ Material encontrado exitosamente")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("SAE Grade recomendado", mejor_material["SAE Grade"])
            st.metric("Tratamiento", mejor_material["Condition_simple"])

        with col2:
            st.metric("Similitud", f"{similitud:.2f}%")
            st.metric("% Carbono", f"{mejor_material['%C']:.3f}")

        with col3:
            st.metric("UTS", f"{mejor_material['UTS (MPa)']} MPa")
            st.metric("Dureza", f"{mejor_material['Hardness (HB)']} HB")

        st.subheader("📋 Propiedades completas")
        st.dataframe(mejor_material.to_frame(name="Valor"), use_container_width=True)

        st.subheader("🧠 Interpretación rápida")

        if mejor_material["%C"] < 0.3:
            st.info("""
            **Acero de bajo carbono:** mayor ductilidad y soldabilidad,
            con menor dureza y resistencia.
            """)
        elif mejor_material["%C"] < 0.6:
            st.info("""
            **Acero de medio carbono:** balance entre resistencia, dureza y ductilidad.
            """)
        else:
            st.info("""
            **Acero de alto carbono:** alta dureza y resistencia al desgaste,
            con menor ductilidad.
            """)


# ==========================================================
# CIERRE
# ==========================================================

st.markdown("---")

st.markdown("""
<div class="conclusion-box">
<p style="line-height:1.8; font-size:15px;">
<strong style="color:#38bdf8;">Conclusión general:</strong> El comportamiento mecánico de los aceros al carbono
depende de la composición química (especialmente el %C) y de los procesos térmicos o mecánicos aplicados.
El carbono tiende a aumentar resistencia y dureza. Los tratamientos modifican la microestructura y pueden
cambiar significativamente ductilidad, resistencia y dureza final. La temperatura del tratamiento añade
una capa adicional de control sobre la microestructura resultante.
</p>
</div>
""", unsafe_allow_html=True)
