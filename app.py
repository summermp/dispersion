import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# Configuración de página
st.set_page_config(page_title="Dispersión de contaminantes", page_icon="🌊", 
                   initial_sidebar_state="expanded", layout='wide')

# Título principal
st.title("Dispersión de Plomo en el Río Rímac")
col1, col2 = st.columns([1,4])
with col1:
    st.image("estacion-chosica.jpg", width=200)
with col2:
    st.markdown("""La estación hidrológica Chosica es el punto de control más representativo de la cuenca del río Rímac y se ubica en el distrito de Lurigancho - Provincia de Lima
    
    CUENCA: RIMAC
    ALTITUD: 906 m.s.n.m
    LATITUD: 11º55'47.5
    LONGITUD: 76º41'22.8
    DEPART.: LIMA
    PROVINCIA: LIMA
    DISTRITO: LURIGANCHO
    """)


# Selector de modelo
modelo = st.radio("Seleccionar tipo de simulación:", 
                  ("Datos Aleatorios", "Datos Reales"), 
                  horizontal=True)

# =============================================
# MODELO CON DATOS ALEATORIOS
# =============================================
if modelo == "Datos Aleatorios":
    st.subheader("Simulación con Datos Aleatorios")
    
    # Contenedor expandible para parámetros
    with st.expander("⚙️ Parámetros del Modelo", expanded=True):
        dias = st.slider("Número de días", 7, 90, 30)
        puntos_por_dia = st.selectbox("Puntos por día", [12, 24, 48], index=1)
        n_puntos = dias * puntos_por_dia

    # Generación de datos sintéticos
    with st.spinner("Generando datos aleatorios..."):
        np.random.seed(42)
        fechas = pd.date_range(start="2023-06-01", periods=n_puntos, freq="h")
        
        data = pd.DataFrame({
            'fecha': fechas,
            'conc_descarga': stats.lognorm.rvs(s=0.5, loc=10, scale=80, size=n_puntos),
            'caudal': stats.norm.rvs(loc=15.0, scale=2.5, size=n_puntos),
            'velocidad': stats.norm.rvs(loc=0.7, scale=0.1, size=n_puntos),
            'dispersion': stats.norm.rvs(loc=30, scale=5, size=n_puntos)
        })

    # Mostrar datos
    with st.expander("📊 Ver datos generados"):
        st.metric("Total de registros", len(data))
        st.dataframe(data)

    # Modelo de dispersión
    def modelo_dispersion_plomo(C0, Q, U, D, x, t):
        dispersion_factor = np.exp(-(x - U * t)**2 / (4 * D * t))
        atenuacion = np.exp(-0.0001 * x)
        C = (C0 * dispersion_factor / np.sqrt(4 * np.pi * D * t)) * atenuacion
        return np.maximum(C, 0)

    # Simulación de distancias
    distancias = st.multiselect("Distancias de simulación (metros)", 
                               [500, 1000, 2000, 5000, 10000, 20000],
                               default=[500, 2000, 5000, 10000, 20000])
    
    for distancia in distancias:
        data[f'conc_{distancia}m'] = data.apply(
            lambda row: modelo_dispersion_plomo(
                C0=row['conc_descarga'],
                Q=row['caudal'],
                U=row['velocidad'],
                D=row['dispersion'],
                x=distancia,
                t=distancia / row['velocidad'] if row['velocidad'] > 0 else 1
            ), axis=1
        )

    # Visualizaciones
    st.subheader("Resultados de la Simulación")
    
    # Gráfico 1: Concentración en punto de descarga
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['fecha'], y=data['conc_descarga'], 
                             mode='lines', name='Descarga'))
    fig1.update_layout(title='Concentración en Punto de Descarga',
                      xaxis_title='Fecha', yaxis_title='μg/L', height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Dispersión aguas abajo
    fig2 = go.Figure()
    for distancia in distancias:
        fig2.add_trace(go.Scatter(x=data['fecha'], y=data[f'conc_{distancia}m'], 
                                 mode='lines', name=f'{distancia}m'))
    fig2.update_layout(title='Dispersión Aguas Abajo',
                      xaxis_title='Fecha', yaxis_title='μg/L', height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Perfil longitudinal
    distancias_km = np.linspace(0, 25, 100)
    perfil_promedio = [
        modelo_dispersion_plomo(
            C0=data['conc_descarga'].mean(),
            Q=data['caudal'].mean(),
            U=data['velocidad'].mean(),
            D=data['dispersion'].mean(),
            x=d*1000,
            t=(d*1000)/data['velocidad'].mean()
        ) for d in distancias_km
    ]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=distancias_km, y=perfil_promedio, 
                             mode='lines', name='Promedio'))
    fig3.update_layout(title='Perfil Longitudinal Promedio',
                      xaxis_title='Distancia (km)', yaxis_title='μg/L', height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Análisis de excedencias
    st.subheader("Análisis de Excedencias")
    limite_plomo = st.number_input("Límite de plomo (μg/L)", value=10.0, min_value=0.1, step=0.5)
    
    cols = st.columns(len(distancias))
    for i, distancia in enumerate(distancias):
        excedencias = data[data[f'conc_{distancia}m'] > limite_plomo]
        porcentaje = len(excedencias) / len(data) * 100
        cols[i].metric(f"A {distancia}m", f"{porcentaje:.1f}%", "excedencias")

# =============================================
# MODELO CON DATOS REALES
# =============================================
else:
    st.subheader("Dispersion del plomo en el Río Rímac")
    
    # Cargar datos
    @st.cache_data
    def cargar_datos():
        url = "https://raw.githubusercontent.com/summermp/dispersion/main/Plomo.xlsx"
        df = pd.read_excel(url, engine='openpyxl')
        df.columns = df.columns.str.strip()
        df["AÑO"] = df["AÑO"].astype(int)
        return df
    
    try:
        df = cargar_datos()
        st.success("Datos cargados correctamente")
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()
    
    # Mostrar datos
    with st.expander("📊 Ver datos reales"):
        st.metric("Total de registros", len(df))
        st.dataframe(df)
    
    # Parámetros del modelo
    with st.expander("⚙️ Configurar parámetros físicos", expanded=True):
        c1, c2, c3 = st.columns(3)
        x = c1.number_input("Distancia desde la fuente (m)", value=200, min_value=10)
        A_rio = c2.number_input("Área transversal del río (m²)", value=35.0, min_value=0.1)
        D_base = c3.number_input("Coeficiente de dispersión (m²/s)", value=85.0, min_value=0.1)
    
    # Cálculos del modelo
    df["u"] = (df["CAUDAL (m³/s)"] / A_rio) * np.random.uniform(0.85, 0.95, len(df))
    df["t"] = x / df["u"]
    
    # Función de dispersión
    def calcular_dispersion(C0, t, x, u, D):
        if t <= 0 or u <= 0 or D <= 0:
            return 0
        term1 = C0 / np.sqrt(4 * np.pi * D * t)
        term2 = np.exp(-((x - u * t) ** 2) / (4 * D * t))
        return term1 * term2

    # Aplicar modelo
    modeladas = []
    for _, row in df.iterrows():
        C0 = row["CONCENTRACION mg/L"]
        t = row["t"]
        u = row["u"] + np.random.normal(0, 0.02)
        D = D_base + np.random.normal(0, 5)
        modeladas.append(calcular_dispersion(C0, t, x, u, D))
    
    df["CONC_MODELADA"] = modeladas
    
    # Escalado de resultados
    max_real = df["CONCENTRACION mg/L"].max()
    max_model = df["CONC_MODELADA"].max()
    if max_model > 0:
        df["CONC_MODELADA"] *= max_real / max_model
    
    # Explicación del modelo
    with st.expander("📘 Explicación del modelo físico", expanded=True):
        st.markdown("### Ecuación de advección-dispersión 1D")
        st.latex(r'''
        C(x, t) = \frac{C_0}{\sqrt{4 \pi D t}} \cdot \exp\left(-\frac{(x - ut)^2}{4Dt} \right)
        ''')
        st.markdown("""
        **Donde:**
        - $C_0$: concentración inicial en la fuente (mg/L)
        - $x$: distancia desde el punto de vertido (m)
        - $u$: velocidad del agua (m/s)
        - $D$: coeficiente de dispersión (m²/s)
        - $t$: tiempo desde la descarga (s)
        """)
    
    # Visualización de resultados
    st.subheader("Comparación de Resultados")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["AÑO"], y=df["CONCENTRACION mg/L"],
        mode="lines+markers", name="Medida (real)",
        line=dict(color="royalblue", width=3),
        marker=dict(symbol="circle", size=7)
    ))
    fig.add_trace(go.Scatter(
        x=df["AÑO"], y=df["CONC_MODELADA"],
        mode="lines+markers", name="Modelada (simulada)",
        line=dict(color="darkorange", width=3, dash="dash"),
        marker=dict(symbol="square", size=7)
    ))
    fig.update_layout(
        title="Dispersión Modelada vs Medida",
        xaxis_title="Año",
        yaxis_title="Concentración de Pb (mg/L)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación de resultados
    st.subheader("Interpretación de Resultados")
    st.markdown("""
    - **Línea azul**: Valores reales medidos en el río
    - **Línea naranja**: Simulación del modelo físico
    - Las diferencias se deben a factores no incluidos en el modelo
    - El modelo permite prever el comportamiento general de la contaminación
    """)
    
    # Mostrar datos procesados
    with st.expander("📈 Ver datos procesados"):
        # st.dataframe(df[["AÑO", "CONCENTRACION mg/L", "CONC_MODELADA"]].style.format("{:.4f}"))
        df["AÑO"] = df["AÑO"].astype(int).astype(str)
        st.dataframe(df[["AÑO", "CONCENTRACION mg/L", "CONC_MODELADA"]].set_index("AÑO"))