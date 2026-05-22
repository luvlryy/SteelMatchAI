# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
Versión unificada, corregida de NameError y con traducción completa de la interfaz.
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
# CONFIGURACIÓN GENERAL Y PALETA DE COLORES
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

PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(15,23,42,0)",   
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
# DICCIONARIOS DE TRADUCCIÓN (MAPPING)
# ==========================================================

TRADUCCIONES_TRATAMIENTOS = {
    "Annealed": "Recocido",
    "Normalized": "Normalizado",
    "Hot Rolled": "Laminado en caliente",
    "Cold Drawn": "Estirado en frío",
    "Quenched": "Templado",
    "Tempered": "Revenido",
    "As Rolled": "Laminado en bruto",
    "Other": "Otros"
}

TRADUCCIONES_INVERSAS = {v: k for k, v in TRADUCCIONES_TRATAMIENTOS.items()}


# ==========================================================
# ESTILO VISUAL — CSS CUSTOM ENRIQUECIDO
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
        box-shadow: 0 4px 20px rgba(56,189,248,0.3);
        transition: all 0.3s ease;
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
        backdrop-filter: blur(8px);
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

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
        margin: 2rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,23,42,0.8);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
    }

    .conclusion-enriquecida {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 2px solid #38bdf8;
        border-radius: 20px;
        padding: 35px;
        margin-top: 3rem;
        box-shadow: 0 10px 40px rgba(56, 189, 248, 0.15);
    }
    .conclusion-enriquecida h3 {
        color: #38bdf8 !important;
        font-size: 24px !important;
        margin-top: 0px !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 12px;
    }
    .grid-conclusion {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .card-conclusion-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
    }
    .card-conclusion-item strong {
        color: #38bdf8;
        font-size: 16px;
    }
    .card-conclusion-item p {
        margin: 8px 0 0 0 !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        color: #cbd5e1 !important;
    }

    .prop-badge {
        display: inline-block;
        background: rgba(56,189,248,0.15);
        color: #38bdf8;
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
    [data-testid="metric-container"] * { color: #f8fafc !important; }
    .streamlit-expanderHeader { color: #f8fafc !important; }
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
        st.error("No se pudo procesar el archivo CSV.")
        st.stop()
    elif extension in ["xlsx", "xls"]:
        try: return pd.read_excel(obtener_fuente())
        except Exception as e:
            st.error(f"Error en lectura Excel: {e}")
            st.stop()
    st.stop()


@st.cache_data
def cargar_y_preprocesar_datos(nombre_archivo, archivo_bytes=None):
    aceros = leer_archivo_datos(nombre_archivo, archivo_bytes)
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
# MENÚ LATERAL: LOGICA INICIAL DE CONTROLADORES
# ==========================================================

st.sidebar.subheader("📁 Archivo de datos")
archivo_subido = st.sidebar.file_uploader("Cargar hoja de aceros (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    aceros = cargar_y_preprocesar_datos(archivo_subido.name, archivo_subido.getvalue())
else:
    aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración del Recomendador")

opcion_uso = st.sidebar.selectbox(
    "💡 ¿Para qué vas a usar el acero?",
    options=[
        "Personalizado (Ingresar valores manualmente)",
        "Que se pueda doblar, moldear o soldar fácilmente (Alta Ductilidad)",
        "Construcción general o soporte de estructuras (Equilibrado)",
        "Para herramientas, resortes o alta resistencia al desgaste (Alta Dureza)"
    ]
)

# CORRECCIÓN: Garantizar que deshabilitar_controles se defina en todas las opciones
if "Alta Ductilidad" in opcion_uso:
    val_c, val_uts, val_ys, val_dureza, val_elongacion = 0.15, 415, 230, 115, 33
    deshabilitar_controles = True
elif "Equilibrado" in opcion_uso or "Construcción" in opcion_uso:
    val_c, val_uts, val_ys, val_dureza, val_elongacion = 0.40, 620, 340, 180, 21
    deshabilitar_controles = True
elif "Alta Dureza" in opcion_uso:
    val_c, val_uts, val_ys, val_dureza, val_elongacion = 0.85, 930, 580, 290, 11
    deshabilitar_controles = True
else:
    val_c, val_uts, val_ys, val_dureza, val_elongacion = 0.45, 600, 400, 180, 20
    deshabilitar_controles = False

carbono = st.sidebar.slider("% de Carbono requerido", 0.0, 1.5, float(val_c), 0.01, disabled=deshabilitar_controles)
uts = st.sidebar.slider("UTS requerido (MPa)", 200, 1500, int(val_uts), disabled=deshabilitar_controles)
ys = st.sidebar.slider("YS requerido (MPa)", 100, 1300, int(val_ys), disabled=deshabilitar_controles)
dureza = st.sidebar.slider("Dureza requerida (HB)", 50, 500, int(val_dureza), disabled=deshabilitar_controles)
elongacion = st.sidebar.slider("Elongación requerida (%)", 1, 50, int(val_elongacion), disabled=deshabilitar_controles)

datos_modelo_limpio = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])
lista_originales = datos_modelo_limpio["Condition_simple"].dropna().unique().tolist()
lista_espanol = ["Todos"] + sorted([TRADUCCIONES_TRATAMIENTOS.get(t, t) for t in lista_originales])
tratamiento_elegido_espanol = st.sidebar.selectbox("Tratamiento térmico preferido", lista_espanol)

buscar = st.sidebar.button("Buscar material óptimo")


# ==========================================================
# ESTRUCTURA DE PESTAÑAS (TABS VISTA PRINCIPAL)
# ==========================================================

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio", "📊 Exploración", "🌡️ Temperatura", "🧪 ANOVA", "🔍 Recomendador"
])


# ── TAB 1 — INICIO ──
with tab_inicio:
    st.header("Objetivo del proyecto")
    st.write("Exploración sistemática y modelado matemático sobre la afectación física que inducen las aleaciones estables de Carbono en metales ferrosos.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros cargados", len(aceros))
    col2.metric("Tipos SAE distintos", aceros["SAE Grade"].nunique())
    col3.metric("Tratamientos agrupados", aceros["Condition_simple"].nunique())

    st.subheader("Muestra inicial de matriz limpia")
    st.dataframe(aceros.head(15), use_container_width=True)


# ── TAB 2 — EXPLORACIÓN ──
with tab_exploracion:
    st.header("Exploración inicial de aceros y tratamientos")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Distribución de datos por Grado SAE")
        t_aceros = aceros["SAE Grade"].value_counts().reset_index()
        t_aceros.columns = ["SAE Grade", "Cantidad"]
        st.dataframe(t_aceros, use_container_width=True)
        
    with col_t2:
        st.subheader("Tratamientos detectados")
        t_trats = aceros["Condition_simple"].value_counts().reset_index()
        t_trats.columns = ["Original", "Cantidad"]
        t_trats["Tratamiento (Español)"] = t_trats["Original"].map(TRADUCCIONES_TRATAMIENTOS)
        st.dataframe(t_trats[["Tratamiento (Español)", "Cantidad"]], use_container_width=True)

    st.subheader("Gráfica interactiva: Propiedades vs Composición Química")
    prop_c = st.selectbox("Elige la propiedad para contrastar con el %C", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    df_carb["Tratamiento"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    fig_c = px.scatter(df_carb, x="%C", y=prop_c, color="Tratamiento", hover_data=["SAE Grade"], trendline="ols")
    st.plotly_chart(aplicar_layout_plotly(fig_c), use_container_width=True)


# ── TAB 3 — TEMPERATURA ──
with tab_temp:
    st.header("Parte A — Influencia de la temperatura en el tratamiento")
    prop_t = st.selectbox("Propiedad mecánica a evaluar por temperatura", PROPIEDADES_MECANICAS, key="sb_p_t")
    
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    st.subheader("🔵 Análisis en Proceso de Recocido (Annealed)")
    df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
    if not df_ann.empty:
        fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Blues")
        st.plotly_chart(aplicar_layout_plotly(fig_a), use_container_width=True)
    else:
        st.warning("Faltan datos de temperatura para la muestra de Recocido.")

    st.subheader("🟣 Análisis en Proceso de Normalizado (Normalized)")
    df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
    if not df_norm.empty:
        fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Purples")
        st.plotly_chart(aplicar_layout_plotly(fig_n), use_container_width=True)
    else:
        st.warning("Faltan datos de temperatura para la muestra de Normalizado.")


# ── TAB 4 — ANOVA ──
with tab_anova:
    st.header("Parte B — Modelado Estadístico Inferencial (ANOVA)")
    prop_anova = st.selectbox("Propiedad criterio para ANOVA de dos factores", PROPIEDADES_MECANICAS, key="sb_p_a")
    
    df_anova = aceros.dropna(subset=["%C", "Condition_simple", prop_anova]).copy()
    if len(df_anova) > 10:
        df_anova = df_anova.rename(columns={"%C": "Carbono", prop_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        mod = ols("Propiedad ~ Carbono + C(Tratamiento)", data=df_anova).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Influencia_%"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        st.dataframe(tab_a, use_container_width=True)


# ── TAB 5 — RECOMENDADOR INTELIGENTE ──
with tab_recomendador:
    st.header("🔍 Recomendador Inteligente SteelMatch AI")
    st.write("Usa los controles del menú lateral de la pantalla para calibrar tu búsqueda manual o adaptada por casos prácticos.")

    if not datos_modelo_limpio.empty:
        scaler = MinMaxScaler()
        scaler.fit(datos_modelo_limpio[COLUMNAS_MODELO])

        if buscar:
            df_res = datos_modelo_limpio.copy()
            if tratamiento_elegido_espanol != "Todos":
                tag_or = TRADUCCIONES_INVERSAS.get(tratamiento_elegido_espanol, "Other")
                df_res = df_res[df_res["Condition_simple"] == tag_or]

            if df_res.empty:
                st.warning("No hay coincidencias para ese tratamiento térmico específico en el almacén actual.")
            else:
                usr_in = np.array([[carbono, uts, ys, dureza, elongacion]])
                usr_norm = scaler.transform(usr_in)
                mat_norm = scaler.transform(df_res[COLUMNAS_MODELO])

                dist = euclidean_distances(usr_norm, mat_norm)
                idx = np.argmin(dist)
                match_mat = df_res.iloc[idx]
                pct_sim = 100 / (1 + dist[0][idx])

                st.success("🎯 Búsqueda analítica completada")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Grado Comercial Encontrado", f"SAE {match_mat['SAE Grade']}")
                    st.metric("Tratamiento Sugerido", TRADUCCIONES_TRATAMIENTOS.get(match_mat['Condition_simple'], match_mat['Condition_simple']))
                with c2:
                    st.metric("Precisión Mecánica", f"{pct_sim:.2f}%")
                    st.metric("Carbono Real", f"{match_mat['%C']:.3f} %")
                with c3:
                    st.metric("Resistencia UTS Real", f"{match_mat['UTS (MPa)']} MPa")
                    st.metric("Dureza Brinel", f"{match_mat['Hardness (HB)']} HB")

                st.subheader("📋 Ficha técnica completa del material sugerido")
                st.dataframe(match_mat.to_frame(name="Especificación Técnica"), use_container_width=True)


# ==========================================================
# CONCLUSIÓN GENERAL DINÁMICA Y DE ALTO CONTRASTE
# ==========================================================
st.markdown("""
<div class="conclusion-enriquecida">
    <h3>📢 Conclusión General del Comportamiento de los Aceros</h3>
    <p style="font-size: 15px !important; color: #cbd5e1 !important; line-height: 1.7 !important; margin: 0;">
        El comportamiento mecánico de los aceros al carbono se rige bajo principios físicos de causa y efecto directos:
    </p>
    <div class="grid-conclusion">
        <div class="card-conclusion-item">
            <strong>📈 Si aumenta el porcentaje de Carbono (%C):</strong>
            <p>Aumentan de forma directa la <strong>dureza</strong> y la <strong>resistencia mecánica (UTS)</strong>, lo que permite fabricar piezas resistentes al desgaste como herramientas, cuchillas y matrices industriales.</p>
        </div>
        <div class="card-conclusion-item">
            <strong>🔵 Si se aplica un tratamiento de Recocido (Annealed):</strong>
            <p>Se reduce el esfuerzo interno, disminuyendo la dureza y <strong>aumentando la elongación (ductilidad)</strong>. Esto permite doblar, moldear y soldar el acero con facilidad sin riesgo de fracturas.</p>
        </div>
        <div class="card-conclusion-item">
            <strong>⚡ Si se aplica un tratamiento de Estirado en Frío (Cold Drawn):</strong>
            <p>Se deforma el material mecánicamente a temperatura ambiente, lo que <strong>eleva notablemente el límite elástico (YS)</strong>. Esto permite que soporte fuerzas mayores en estructuras sin deformarse permanentemente.</p>
        </div>
        <div class="card-conclusion-item">
            <strong>🌡️ Afectación por la Temperatura:</strong>
            <p>Incrementar la temperatura en procesos térmicos relaja la microestructura molecular del metal, permitiendo un control milimétrico sobre el balance final entre tenacidad y rigidez del componente.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
