# -*- coding: utf-8 -*-
"""
Proyecto Integrador Materiales — SteelMatch AI
Versión Definitiva Totalmente Corregida — Sin Errores de Sintaxis
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

# ── SILENCIAR ADVERTENCIAS MATEMÁTICAS EN LA INTERFAZ ──
warnings.filterwarnings("ignore", category=UserWarning)


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

    /* ── CUADROS DE GUÍA INTERACTIVA ── */
    .guia-didactica {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #6366f1;
        padding: 15px 20px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 20px;
    }
    
    .guia-didactica p {
        margin: 0 !important;
        font-size: 14px !important;
        color: #cbd5e1 !important;
    }

    /* ── CUADROS DE LEYENDA TÉCNICA EXPLICADA ── */
    .leyenda-grafica {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(56, 189, 248, 0.4);
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    p, span, label, div { color: #e2e8f0 !important; }
    .stSelectbox label { color: #f8fafc !important; }
    input, textarea { color: #ffffff !important; }
    .stTabs [data-baseweb="tab"] { color: #cbd5e1 !important; }
    .stTabs [aria-selected="true"] { color: white !important; }
    [data-testid="metric-container"] * { color: #f8fafc !important; }
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

    # ── CORREGIDO AQUÍ: Llamada correcta a la función en español ──
    aceros["Condition_simple"] = aceros["Conditions"].apply(agrupar_tratamiento)
    aceros["Temp_C"] = aceros["Conditions"].apply(extraer_temperatura)
    return aceros


# ==========================================================
# MENÚ LATERAL: MODOS DE SELECCIÓN Y ADVERTENCIAS FÍSICAS
# ==========================================================

st.sidebar.subheader("📁 Almacén de Datos")
archivo_subido = st.sidebar.file_uploader("Subir base de datos (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    aceros = cargar_y_preprocesar_datos(archivo_subido.name, archivo_subido.getvalue())
else:
    aceros = cargar_y_preprocesar_datos(DEFAULT_DATA_FILE)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración del Acero")

modo_usuario = st.sidebar.radio("👨‍💻 Elige tu perfil de uso:", ["Guiado (Por Niveles)", "Profesional (Control Manual)"])

if modo_usuario == "Guiado (Por Niveles)":
    deshabilitar_controles = True
    
    nivel_resistencia = st.sidebar.select_slider("💪 Nivel de Fuerza / Resistencia (UTS):", options=["Bajo", "Medio", "Alto"], value="Medio")
    nivel_dureza = st.sidebar.select_slider("💎 Nivel de Dureza Superficial (HB):", options=["Bajo", "Medio", "Alto"], value="Medio")
    nivel_flexibilidad = st.sidebar.select_slider("🎗️ Nivel de Flexibilidad / Ductilidad (Elongación):", options=["Bajo", "Medio", "Alto"], value="Medio")

    mapa_uts = {"Bajo": 380, "Medio": 600, "Alto": 950}
    mapa_ys = {"Bajo": 210, "Medio": 380, "Alto": 580}
    mapa_hb = {"Bajo": 105, "Medio": 180, "Alto": 300}
    mapa_elo = {"Bajo": 11, "Medio": 21, "Alto": 35}
    mapa_c = {"Bajo": 0.12, "Medio": 0.45, "Alto": 0.85}

    uts_val = mapa_uts[nivel_resistencia]
    ys_val = mapa_ys[nivel_resistencia]
    dureza_val = mapa_hb[nivel_dureza]
    elongacion_val = mapa_elo[nivel_flexibilidad]
    
    carbono_val = (mapa_c[nivel_resistencia] + mapa_c[nivel_dureza] + (1.0 - mapa_c[nivel_flexibilidad])) / 3
    carbono_val = np.clip(carbono_val, 0.05, 1.2)

    if nivel_resistencia == "Alto" or nivel_dureza == "Alto":
        st.sidebar.warning("⚠️ **Efecto Secundario:** Al buscar alta resistencia o dureza, el sistema requerirá mucho Carbono. **Pero ten en cuenta** que el material perderá flexibilidad molecular; será quebradizo ante golpes secos.")
    elif nivel_flexibilidad == "Alto":
        st.sidebar.success("✅ **Efecto Secundario:** El acero será fantástico para cortarse, moldearse y soldarse. **Pero ten en cuenta** que cederá ante fuerzas bajas y se deformará muy rápido bajo peso.")
    else:
        st.sidebar.caption("Has configurado una aleación balanceada estándar.")

else:
    deshabilitar_controles = False
    carbono_val, uts_val, ys_val, dureza_val, elongacion_val = 0.45, 600, 400, 180, 20
    st.sidebar.info("🧠 **Modo Profesional Activado:** Tienes el control absoluto de los deslizadores de laboratorio. Modifica cada parámetro bajo tus propios criterios de ingeniería.")

carbono = st.sidebar.slider("% de Carbono deseado", 0.0, 1.5, float(carbono_val), 0.01, disabled=deshabilitar_controles)
uts = st.sidebar.slider("Resistencia UTS (MPa)", 200, 1500, int(uts_val), disabled=deshabilitar_controles)
ys = st.sidebar.slider("Límite Elástico YS (MPa)", 100, 1300, int(ys_val), disabled=deshabilitar_controles)
dureza = st.sidebar.slider("Dureza Superficial (HB)", 50, 500, int(dureza_val), disabled=deshabilitar_controles)
elongacion = st.sidebar.slider("Elongación / Ductilidad (%)", 1, 50, int(elongacion_val), disabled=deshabilitar_controles)

datos_modelo_limpio = aceros.dropna(subset=COLUMNAS_MODELO + ["Condition_simple"])
lista_originales = datos_modelo_limpio["Condition_simple"].dropna().unique().tolist()
lista_espanol = ["Todos"] + sorted([TRADUCCIONES_TRATAMIENTOS.get(t, t) for t in lista_originales])
tratamiento_elegido_espanol = st.sidebar.selectbox("Tratamiento térmico preferido", lista_espanol)

buscar = st.sidebar.button("🚀 Buscar material óptimo")


# ==========================================================
# BANCO DE DATOS CURIOSOS ALEATORIOS
# ==========================================================

DATOS_CURIOSOS = [
    "💡 **Dato curioso:** ¡El acero no es un metal puro, es una mezcla de Hierro y Carbono! El carbono actúa como una cuña microscópica que traba los átomos de hierro para que no se deslicen libremente.",
    "💡 **Dato curioso:** El acero de los remaches del Titanic contenía altas impurezas de azufre. Con el agua congelada del Atlántico, el metal se volvió frágil como el vidrio en lugar de doblarse ante el impacto.",
    "💡 **Dato curioso:** Las legendarias katanas samurái combinaban dos tipos de acero: uno muy duro en el exterior para mantener el filo afilado, y uno blando en el núcleo para absorber los golpes sin romperse.",
    "💡 **Dato curioso:** En la Torre Eiffel, el hierro pudelado utilizado se comporta de forma similar al acero estructural moderno de bajo carbono, lo que le permite balancearse frente al viento sin fracturas."
]


# ==========================================================
# ESTRUCTURA DE PESTAÑAS (TABS VISTA PRINCIPAL)
# ==========================================================

st.title("SteelMatch AI")

tab_inicio, tab_exploracion, tab_temp, tab_anova, tab_recomendador = st.tabs([
    "🏠 Inicio", "📊 Exploración", "🌡️ Temperatura", "🧪 ANOVA", "🔍 Recomendador"
])


# ── TAB 1 — INICIO ──
with tab_inicio:
    st.header("¡Bienvenido al Laboratorio de Simulación de Aceros!")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>👋 <strong>¿De qué se trata este espacio?</strong> Imagina que el acero es como el personaje de un videojuego: 
        dependiendo de los ingredientes químicos que le des (% de Carbono) y el tipo de entrenamiento que reciba en el horno (Tratamiento Térmico), 
        cambiará drásticamente sus estadísticas de 'Ataque' (Resistencia) o 'Defensa' (Dureza). Aquí analizaremos matemáticamente esas combinaciones.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(random.choice(DATOS_CURIOSOS))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros analizados en total", len(aceros))
    col2.metric("Grados SAE identificados", aceros["SAE Grade"].nunique())
    col3.metric("Familias de tratamientos", aceros["Condition_simple"].nunique())

    st.subheader("📋 Tu Matriz de Datos Limpia")
    st.caption("Esta es la tabla base con la que el sistema trabaja internamente. Cada fila representa un acero probado en laboratorio.")
    st.dataframe(aceros.head(15), use_container_width=True)


# ── TAB 2 — EXPLORACIÓN ──
with tab_exploracion:
    st.header("📊 Exploración y Comportamiento General")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>💡 <strong>¿Qué estás viendo aquí?</strong> En esta sección ponemos a prueba la regla de oro de la metalurgia. 
        Analizaremos visualmente cómo el cambio en la química del material impacta en su rendimiento físico. 
        Usa el selector para ver cómo reaccionan las distintas propiedades mecánicas frente al Carbono.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Gráfica Interactiva: Química vs Propiedades Mecánicas")
    prop_c = st.selectbox("Elige el eje vertical (Y) para cruzarlo con el % de Carbono", PROPIEDADES_MECANICAS)
    
    df_carb = aceros.dropna(subset=["%C", prop_c, "Condition_simple"]).copy()
    
    # ── CORREGIDO AQUÍ: Eliminada la duplicación errónea de la línea 473 ──
    df_carb["Tratamiento"] = df_carb["Condition_simple"].map(TRADUCCIONES_TRATAMIENTOS)
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        📌 <strong>Guía rápida para interpretar el siguiente mapa visual:</strong><br>
        • 🔵 <strong>Cada Punto en el espacio:</strong> Representa una muestra real de acero probada en laboratorio. Su altura te indica el nivel de <strong>{prop_c}</strong> que alcanzó. Si pasas tu cursor sobre cualquiera, verás su etiqueta SAE comercial.<br>
        • 📈 <strong>La Línea Recta (Tendencia):</strong> Si la línea va <strong>hacia arriba</strong>, el Carbono potencia la propiedad. Si va <strong>hacia abajo</strong>, significa que el Carbono reduce esa característica a medida que añades más cantidad.
    </div>
    """, unsafe_allow_html=True)

    fig_c = px.scatter(df_carb, x="%C", y=prop_c, color="Tratamiento", hover_data=["SAE Grade"], trendline="ols")
    st.plotly_chart(aplicar_layout_plotly(fig_c), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Distribuciones de Muestras Registradas")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.caption("Cantidad de aceros disponibles agrupados por su Norma Comercial (SAE Grade).")
        t_aceros = aceros["SAE Grade"].value_counts().reset_index()
        t_aceros.columns = ["SAE Grade", "Cantidad"]
        st.dataframe(t_aceros, use_container_width=True)
        
    with col_t2:
        st.caption("Cantidad de aceros disponibles de acuerdo al Proceso Térmico que recibieron en fábrica.")
        t_trats = aceros["Condition_simple"].value_counts().reset_index()
        t_trats.columns = ["Original", "Cantidad"]
        t_trats["Tratamiento (Español)"] = t_trats["Original"].map(TRADUCCIONES_TRATAMIENTOS)
        st.dataframe(t_trats[["Tratamiento (Español)", "Cantidad"]], use_container_width=True)


# ── TAB 3 — TEMPERATURA ──
with tab_temp:
    st.header("🌡️ El Efecto del Horno (Influencia de la Temperatura)")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>🔥 <strong>¿Qué pasa aquí adentro?</strong> Cuando metemos el acero al fuego, los átomos internos se relajan y se acomodan. 
        Subir la temperatura tiende a eliminar el 'estrés' del metal (haciéndolo más flexible pero reduciendo un poco su rigidez extrema). 
        A continuación, evaluaremos por separado los dos tratamientos en horno más comunes del sector industrial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_t = st.selectbox("Propiedad mecánica a evaluar frente a los cambios de temperatura", PROPIEDADES_MECANICAS, key="sb_p_t")
    df_temp = aceros.dropna(subset=["Temp_C", prop_t, "Condition_simple"])
    
    st.markdown(f"""
    <div class="leyenda-grafica">
        📌 <strong>¿Cómo leer las gráficas térmicas de abajo?</strong><br>
        • El eje horizontal (X) muestra los grados de calor del horno. El eje vertical (Y) muestra el impacto directo en <strong>{prop_t}</strong>.<br>
        • 🎨 <strong>Código de color de los puntos:</strong> Los puntos oscuros tienen muy poco carbono (aceros dulces), mientras que los puntos brillantes están sumamente cargados de carbono.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔵 Proceso de Recocido (Annealed)")
    st.caption("Consiste en calentar el metal y enfriarlo de forma extremadamente lenta dentro del propio horno. Se usa para ablandar el material.")
    df_ann = df_temp[df_temp["Condition_simple"] == "Annealed"]
    if not df_ann.empty:
        fig_a = px.scatter(df_ann, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Blues")
        st.plotly_chart(aplicar_layout_plotly(fig_a), use_container_width=True)
    else:
        st.warning("No hay suficientes registros con temperatura explícita para la muestra de Recocido.")

    st.subheader("🟣 Proceso de Normalizado (Normalized)")
    st.caption("Calienta el metal a altas temperaturas y luego se extrae para enfriarse al aire libre. Busca un grano uniforme, balanceado y tenaz.")
    df_norm = df_temp[df_temp["Condition_simple"] == "Normalized"]
    if not df_norm.empty:
        fig_n = px.scatter(df_norm, x="Temp_C", y=prop_t, color="%C", trendline="ols", color_continuous_scale="Purples")
        st.plotly_chart(aplicar_layout_plotly(fig_n), use_container_width=True)
    else:
        st.warning("No hay suficientes registros con temperatura explícita para la muestra de Normalizado.")


# ── TAB 4 — ANOVA ──
with tab_anova:
    st.header("🧪 El Detective Estadístico (Análisis ANOVA)")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>🕵️‍♂️ <strong>¿Quién es el verdadero responsable del cambio?</strong> La palabra 'ANOVA' suena compleja, pero funciona como un detective matemático. 
        Su trabajo es revisar las variaciones de las propiedades e indicar, mediante porcentajes, qué tanta culpa tiene la química básica (% Carbono) 
        y qué tanta responsabilidad es del tratamiento en el horno. Mira la tabla final para conocer el veredicto científico.</p>
    </div>
    """, unsafe_allow_html=True)
    
    prop_anova = st.selectbox("Elige la propiedad criterio para ejecutar el juicio estadístico", PROPIEDADES_MECANICAS, key="sb_p_a")
    
    st.subheader("📊 Tabla de Influencia Porcentual Matemático")
    st.caption("Fíjate principalmente en la columna 'Influencia_%': el valor más alto dictamina quién domina el control de la propiedad.")

    df_anova = aceros.dropna(subset=["%C", "Condition_simple", prop_anova]).copy()
    if len(df_anova) > 10:
        df_anova = df_anova.rename(columns={"%C": "Carbono", prop_anova: "Propiedad", "Condition_simple": "Tratamiento"})
        mod = ols("Propiedad ~ Carbono + C(Tratamiento)", data=df_anova).fit()
        tab_a = sm.stats.anova_lm(mod, typ=2)
        tab_a["Influencia_%"] = (tab_a["sum_sq"] / tab_a["sum_sq"].sum()) * 100
        st.dataframe(tab_a, use_container_width=True)


# ── TAB 5 — RECOMENDADOR INTELIGENTE ──
with tab_recomendador:
    st.header("🔍 Recomendador de Materiales por Similitud Matemática")
    
    st.markdown("""
    <div class="guia-didactica">
        <p>🤖 <strong>¿Cómo funciona el buscador?</strong> Ajusta tus niveles preferidos (Baja, Mediana, Alta) en el panel izquierdo. 
        Al presionar el botón de búsqueda, el recomendador comparará tus deseos numéricos contra la base de datos real del almacén utilizando distancias vectoriales, 
        entregándote el acero comercial existente que mejor se adapta a tus requisitos.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📚 Glosario rápido de términos (¿Qué significan estas variables?)"):
        st.markdown("""
        * **UTS (Ultimate Tensile Strength):** ¿Cuánta fuerza de estiramiento resiste el acero antes de romperse por completo?
        * **YS (Yield Strength):** Límite elástico. Hasta este punto puedes estirar el acero y regresará a su forma original como una liga. Si lo superas, se deforma permanentemente.
        * **Hardness (Dureza HB):** Resistencia que tiene la cara del acero a ser rayada o abollada.
        * **Elongation (%):** Capacidad del material para estirarse como chicle sin partirse. Ideal si necesitas doblarlo en frío.
        """)

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

                st.success("🎯 Búsqueda analítica completada con éxito")
                
                st.subheader("🏆 Material Comercial Recomendado")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Grado Comercial Encontrado", f"SAE {match_mat['SAE Grade']}")
                    st.metric("Tratamiento Sugerido", TRADUCCIONES_TRATAMIENTOS.get(match_mat['Condition_simple'], match_mat['Condition_simple']))
                with c2:
                    st.metric("Compatibilidad del Match", f"{pct_sim:.2f}%")
                    st.metric("Porcentaje de Carbono Real", f"{match_mat['%C']:.3f} %")
                with c3:
                    st.metric("Resistencia UTS Real", f"{match_mat['UTS (MPa)']} MPa")
                    st.metric("Dureza Brinell Real", f"{match_mat['Hardness (HB)']} HB")

                st.markdown("---")
                st.subheader("📋 Ficha Técnica Oficial Completa")
                st.dataframe(match_mat.to_frame(name="Especificación Técnica"), use_container_width=True)


# ==========================================================
# CONCLUSIÓN GENERAL DINÁMICA Y DE ALTO CONTRASTE
# ==========================================================
st.markdown("""
<div class="conclusion-enriquecida">
    <h3>📢 Conclusión General del Comportamiento de los Aceros</h3>
    <p style="font-size: 15px !important; color: #cbd5e1 !important; line-height: 1.7 !important; margin: 0;">
        El comportamiento mecánico de los aceros al carbono se rige bajo principios físicos de causa y efecto directos. El mapa lógico se resume así:
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
