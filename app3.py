import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import os
import requests

# -------- CONFIGURACIÓN DE LA PÁGINA --------
st.set_page_config(page_title="Análisis Financiero & Descripción IA", page_icon="📈", layout="wide")

# -------- ESTILOS PERSONALIZADOS --------
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #000000;
            color: #000000;
        }
        .stButton>button {
            color: white;
            background: linear-gradient(90deg, #007BFF, #00D4FF);
            border: none;
            padding: 0.6em 2em;
            font-weight: bold;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 10px;
            border-radius: 12px;
            border: 2px solid #ccc;
            text-align: center;
            color: #333;
        }
        .stMarkdown {
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            color: #black;
        }
    </style>
""", unsafe_allow_html=True)

# -------- CONEXIÓN A GEMINI (GOOGLE GENAI) --------
genai.configure(api_key="AIzaSyAL8weo8gSEtaBtrs7WDoLAyIS42Id02M4")  
model = genai.GenerativeModel("gemini-1.5-flash")

# -------- SIDEBAR --------
with st.sidebar:
    st.markdown("""
        <style>
            /* Sidebar container con degradado y textura */
            .css-1d391kg {
                background: linear-gradient(135deg, #0b121f, #121a2a);
                background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
                padding: 30px 30px 40px 30px;
                border-radius: 20px;
                box-shadow: 0 15px 40px rgba(0, 70, 110, 0.7);
                color: #c7d0e0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            /* Título principal azul grisáceo con glow */
            .sidebar-title {
                font-size: 30px;
                font-weight: 800;
                color: #6da4d1;
                text-align: center;
                margin-bottom: 15px;
                text-shadow: 0 0 10px #4a6fa9;
                letter-spacing: 2px;
                font-style: italic;
                user-select: none;
            }

            /* Subtítulo elegante */
            .sidebar-sub {
                font-size: 15px;
                font-style: italic;
                color: #8fa6cc;
                margin-bottom: 35px;
                text-align: center;
                letter-spacing: 0.8px;
                user-select: none;
            }

            /* Secciones con bordes dobles y color azul pastel */
            .sidebar-section {
                font-size: 20px;
                font-weight: 600;
                color: #a1b3d1;
                margin-top: 32px;
                margin-bottom: 15px;
                border-bottom: 3px double #4a6fa9;
                padding-bottom: 8px;
                letter-spacing: 1px;
                user-select: none;
            }

            /* Inputs con sombra interior y animación */
            input[type="text"], input[type="date"] {
                background: #152233 !important;
                color: #d2d9e6 !important;
                border: 2.5px solid #4a6fa9 !important;
                border-radius: 18px !important;
                padding: 14px 20px !important;
                font-size: 18px !important;
                text-align: center !important;
                box-shadow: inset 0 0 12px #3a5478;
                transition: border-color 0.4s ease, box-shadow 0.4s ease;
                user-select: text;
            }
            input[type="text"]:focus, input[type="date"]:focus {
                border-color: #6da4d1 !important;
                box-shadow: 0 0 18px #6da4d1 !important;
                outline: none !important;
            }

            /* Botón moderno con degradado y animación */
            .stButton > button {
                background: linear-gradient(90deg, #5a7fa6, #3b5e8c);
                color: #f0f5ff;
                font-weight: 700;
                padding: 16px 0;
                border-radius: 25px;
                border: none;
                font-size: 22px;
                letter-spacing: 1.3px;
                box-shadow: 0 8px 30px rgba(58, 94, 140, 0.8);
                width: 100%;
                margin-top: 30px;
                transition: background 0.5s ease, box-shadow 0.5s ease;
                user-select: none;
            }
            .stButton > button:hover {
                background: linear-gradient(90deg, #3b5e8c, #5a7fa6);
                box-shadow: 0 12px 40px rgba(58, 94, 140, 1);
                cursor: pointer;
            }

            /* Checkboxes estilizados */
            div.stCheckbox > label > div {
                color: #9db3d8 !important;
                font-size: 17px !important;
                font-weight: 600 !important;
                user-select: none;
                transition: color 0.3s ease;
            }
            div.stCheckbox > label > div:hover {
                color: #6da4d1 !important;
                cursor: pointer;
            }

            /* Separador personalizado */
            hr {
                border: none;
                border-top: 1.5px solid #3a5478;
                margin: 30px 0;
                width: 90%;
                margin-left: auto;
                margin-right: auto;
            }
        </style>

        <div class='sidebar-title'>📊 FINTECH AI</div>
        <div class='sidebar-sub'>Comienza el Análisis: </div>

        <div class='sidebar-section'>💼 Ticker de la Empresa</div>
    """, unsafe_allow_html=True)

    ticker = st.text_input("", "AAPL")

    st.markdown("<div class='sidebar-section'>📆 Fechas a Consultar</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Desde:", datetime.today() - timedelta(days=5*365))
    with col2:
        end_date = st.date_input("Hasta:", datetime.today())

    st.markdown("<div class='sidebar-section'>⚙️ Opciones Adicionales</div>", unsafe_allow_html=True)
    show_comparativa = st.checkbox("Comparar con sector", value=True)
    show_descripcion = st.checkbox("Mostrar descripción de la empresa", value=True)
    show_historico = st.checkbox("Ver histórico de precios", value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    submit = st.button("🚀 Generar Análisis")

    st.caption("© 2025 VanniaAPP - Todos los derechos reservados.")


# -------- FUNCIÓN DE VALIDACIÓN --------
def validar_ticker(ticker):
    ticker = ticker.upper()  # Normalizar siempre
    try:
        data = yf.Ticker(ticker)
        info = data.info
        # Algunas veces info puede estar vacío o tener claves parciales
        if info and "shortName" in info and info['shortName'] is not None:
            return info
    except Exception as e:
        # Puedes imprimir/loggear e si quieres debug
        pass
    return None

# -------- FLUJO PRINCIPAL --------
if submit:
    info = validar_ticker(ticker)

    if info:
        st.title(f"📊 **Análisis Financiero de {info['shortName']}** 📈")
        st.markdown(f"""
        **Sector:** {info.get('sector', 'N/A')}  
        **Industria:** {info.get('industry', 'N/A')}
        """)
        st.markdown("""
        <style>
        @keyframes moveLine {
        0% {background-position: 0 0;}
        100% {background-position: 100% 0;}
        }
        .animated-line {
        height: 3px;
        background: linear-gradient(270deg, #00cfff, #005f99, #00cfff);
        background-size: 200% 100%;
        animation: moveLine 3s linear infinite;
        border-radius: 10px;
        margin: 25px 0 40px 0;
        }
        </style>
        <div class="animated-line"></div>
        """, unsafe_allow_html=True)

        # -------- ANÁLISIS FUNDAMENTAL --------
        st.markdown("### 💡 Datos Fundamentales Clave")

        col1, col2, col3 = st.columns(3)

        datos_clave = [
            ("Precio Actual (USD)", f"${info.get('currentPrice', 'N/A')}"),
            ("Ganancias por Acción (EPS)", info.get('epsTrailingTwelveMonths', 'N/A')),
            ("Precio/Ganancias (P/E)", info.get('trailingPE', 'N/A')),
        ]

        for i, (nombre, valor) in enumerate(datos_clave):
            col = [col1, col2, col3][i]
            col.markdown(f"""
                <div style="
                    background: #121212; 
                    border-radius: 10px; 
                    padding: 20px; 
                    margin-bottom: 20px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    text-align: center;
                ">
                    <h4 style="margin: 0; color: #00C2FF;">{nombre}</h4>
                    <p style="font-size: 1.6rem; font-weight: bold; color: #ffffff; margin: 10px 0 0 0;">{valor}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("----")

        # -------- DESCRIPCIÓN GENERADA POR IA --------
        with st.spinner("🧠 **Buscando información importante de la empresa...**"):
            prompt = f"Resume profesionalmente a qué se dedica la empresa {info['longName']}, cuyo ticker es {ticker}. Dame un resumen profesional de la empresa."
            try:
                response = model.generate_content(prompt)
                st.markdown("### 💡 **Descripción de la Empresa**")
                st.write(response.text)
            except Exception as e:
                st.error(f"❌ **Error al generar la descripción con IA: {str(e)}**")

        # -------- DATOS HISTÓRICOS --------
        df = yf.download(ticker, start=start_date, end=end_date)
        if 'Adj Close' in df.columns:
            df['Price'] = df['Adj Close']
        else:
            df['Price'] = df['Close']
        df.fillna(method='ffill', inplace=True)

        # -------- EMPRESAS DEL MISMO SECTOR --------
        sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN'],
            'Healthcare': ['JNJ', 'PFE', 'MRK', 'GILD', 'ABBV'],
            'Financials': ['JPM', 'GS', 'BAC', 'WFC', 'C'],
            'Consumer Discretionary': ['DIS', 'MCD', 'NKE', 'TGT', 'AMZN'],
            'Other': []
        }
        tickers_comparativa = sector_tickers.get(info.get('sector', 'Other'), [])
        sector_data = {}

        for sec_ticker in tickers_comparativa:
            data = yf.download(sec_ticker, start=start_date, end=end_date)
            if 'Adj Close' in data.columns:
                data['Price'] = data['Adj Close']
            else:
                data['Price'] = data['Close']
            data.fillna(method='ffill', inplace=True)
            sector_data[sec_ticker] = data

        # -------- GRÁFICO HISTÓRICO PRINCIPAL --------
        st.markdown("### 📉 Evolución Histórica de Precios")
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Price'],
                mode="lines",
                name=f"{ticker} (Cierre Ajustado)",
                line=dict(color="#E91E63")
            ))
            fig.update_layout(
                title=f"Precio histórico de {ticker.upper()}",
                xaxis_title="Fecha",
                yaxis_title="Precio en USD",
                xaxis=dict(tickformat="%b %Y", tickangle=45),
                template="plotly_white",
                height=480
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No se hallaron datos suficientes para mostrar el gráfico.")

        st.divider()

        # ----- ANÁLISIS TÉCNICO -----------
        st.markdown("### 📊 Análisis Técnico")

        # Cálculo de medias móviles
        df['SMA_20'] = df['Price'].rolling(window=20).mean()
        df['SMA_50'] = df['Price'].rolling(window=50).mean()

        # RSI (Relative Strength Index)
        delta = df['Price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Gráfico con SMA
        fig_ta = go.Figure()
        fig_ta.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='Precio'))
        fig_ta.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20 días'))
        fig_ta.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50 días'))
        fig_ta.update_layout(title="📈 Precio vs Medias Móviles", template="plotly_white")
        st.plotly_chart(fig_ta, use_container_width=True)

        # Gráfico RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi.update_layout(title="📉 Índice de Fuerza Relativa (RSI)", yaxis=dict(range=[0, 100]), template="plotly_white")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # -------- CONCLUSIÓN AUTOMÁTICA DEL ANÁLISIS TÉCNICO --------
        st.markdown("### 🧠 Conclusión del Análisis Técnico")

        ultima_rsi = df['RSI'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        ultimo_precio = df['Price'].iloc[-1]

        conclusiones = []

        # RSI Interpretation
        if ultima_rsi < 30:
            conclusiones.append("🔵 El RSI indica que la acción está en zona de **sobreventa** (posible rebote al alza).")
        elif ultima_rsi > 70:
            conclusiones.append("🔴 El RSI indica una **sobrecompra** (riesgo de corrección a la baja).")
        else:
            conclusiones.append("🟡 El RSI se encuentra en zona neutral, sin señales claras de compra o venta.")

        # SMA Interpretation
        if sma_20 > sma_50:
            conclusiones.append("📈 La media móvil de 20 días está por encima de la de 50 días: señal de **tendencia alcista**.")
        elif sma_20 < sma_50:
            conclusiones.append("📉 La media móvil de 20 días está por debajo de la de 50 días: señal de **tendencia bajista**.")
        else:
            conclusiones.append("➖ Las medias móviles están alineadas: **no hay tendencia clara**.")

        # Comparación de precio actual vs. medias móviles
        if ultimo_precio > sma_20:
            conclusiones.append("✅ El precio actual está por encima de su media de corto plazo (SMA 20), lo que sugiere **fortaleza reciente**.")
        else:
            conclusiones.append("⚠️ El precio actual está por debajo de su media de corto plazo (SMA 20), lo que podría indicar **debilidad en el corto plazo**.")

        # Mostrar conclusiones
        for c in conclusiones:
            st.markdown(c)
        st.markdown("---")

        # -------- SIMULACIÓN MONTECARLO --------
        st.markdown("### 🔮 Simulación Monte Carlo - Predicción a 30 días")

        # Número de simulaciones y días
        num_simulaciones = 1000  # aumentamos para más profundidad
        num_dias = 30

        # Calcular log-returns
        df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        media = df['log_return'].mean()
        volatilidad = df['log_return'].std()

        # Precio inicial
        precio_actual = df['Price'].iloc[-1]

        # Simular trayectorias
        simulaciones = np.zeros((num_dias, num_simulaciones))

        for sim in range(num_simulaciones):
            precio = precio_actual
            for dia in range(num_dias):
                shock = np.random.normal(loc=media, scale=volatilidad)
                precio *= np.exp(shock)
                simulaciones[dia, sim] = precio

        # Crear gráfico
        fig_mc = go.Figure()
        for i in range(0, num_simulaciones, int(num_simulaciones/50)):  # para no saturar el gráfico
            fig_mc.add_trace(go.Scatter(
                x=list(range(1, num_dias+1)),
                y=simulaciones[:, i],
                mode='lines',
                line=dict(width=0.5),
                opacity=0.15,
                showlegend=False
            ))

        fig_mc.update_layout(
            title="🔁 Proyecciones de Precio (Monte Carlo)",
            xaxis_title="Días en el Futuro",
            yaxis_title="Precio Estimado (USD)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # -------- CONCLUSIÓN DE LA SIMULACIÓN --------
        precio_finales = simulaciones[-1]
        p_min = round(np.percentile(precio_finales, 5), 2)
        p_max = round(np.percentile(precio_finales, 95), 2)
        p_median = round(np.median(precio_finales), 2)

        st.markdown(f"""
        🧠 Basado en {num_simulaciones} simulaciones de Monte Carlo:
        - 📉 Hay un 90% de probabilidad de que el precio esté entre **${p_min}** y **${p_max}** en 30 días.
        - 📍 El precio mediano estimado es **${p_median}**.
        """)


        # -------- ESCENARIOS Y PLANIFICACIÓN --------
        st.markdown("### 🔍 Proyección y Planificación a 30 días")

        esc_opt = p_max
        esc_neutro = p_median
        esc_pes = p_min
        precio_act = df['Price'].iloc[-1]

        diff_opt = round((esc_opt / precio_act - 1) * 100, 2)
        diff_neu = round((esc_neutro / precio_act - 1) * 100, 2)
        diff_pes = round((esc_pes / precio_act - 1) * 100, 2)

        st.markdown(f"""
        - 🚀 **Escenario Optimista:** Precio podría alcanzar **${esc_opt}**, un alza potencial de **{diff_opt}%**. ¡Momento para aprovechar oportunidades!  
        - ⚖️ **Escenario Neutral:** Precio se mantendría cerca de **${esc_neutro}**, variando alrededor de **{diff_neu}%**. Tiempo para observar y consolidar posiciones.  
        - ⚠️ **Escenario Pesimista:** Precio podría caer hasta **${esc_pes}**, una baja de **{diff_pes}%**. Precaución, considera proteger tus inversiones.
        """)

        st.markdown("""
        **Recomendaciones clave:**

        - Si el escenario optimista se sostiene y la volatilidad es controlada, considera **mantener o aumentar posiciones** con gestión de riesgo.  
        - Ante un posible escenario pesimista, **define límites de pérdidas (stops)** y revisa tu exposición.  
        - Usa siempre señales técnicas y fundamentales como guía para ajustar tu estrategia.
        """)
        st.markdown("---")
    

        # -------- ANÁLISIS IA DE LA SIMULACIÓN --------
        with st.spinner("🧠 Generando análisis con IA..."):
            prompt_mc = f"""
            Analiza de manera breve los resultados de la simulación Monte Carlo para la acción con precio actual de ${precio_actual:.2f}.
            La simulación realizó {num_simulaciones} trayectorias para los próximos {num_dias} días.
            Los percentiles 5% y 95% son ${p_min} y ${p_max} respectivamente, con un precio mediano estimado de ${p_median}.
            Considera volatilidad histórica y posibles riesgos para un análisis completo y profesional.

            Proporciona una conclusión clara para inversionistas, incluyendo posibles escenarios y recomendaciones de manera breve.
            """

            try:
                respuesta_mc = model.generate_content(prompt_mc)
                st.markdown("### 🤖 Análisis IA de la Simulación")
                st.write(respuesta_mc.text)
            except Exception as e:
                st.error(f"❌ Error al generar análisis con IA: {str(e)}")
        st.markdown("---")        


        # -------- COMPARATIVA CON EMPRESAS DEL SECTOR --------
        st.markdown("### 🏢 Tabla Comparativa con Empresas del Sector")

        fig_sector = go.Figure()
        for sec_ticker, sec_df in sector_data.items():
            if not sec_df.empty:
                sec_df_norm = sec_df['Price'] / sec_df['Price'].iloc[0]  # Normalizar
                fig_sector.add_trace(go.Scatter(
                    x=sec_df.index,
                    y=sec_df_norm,
                    mode='lines',
                    name=f'{sec_ticker}'
                ))

        if not df.empty:
            df_norm = df['Price'] / df['Price'].iloc[0]  # Normalizar ticker principal
            fig_sector.add_trace(go.Scatter(
                x=df.index,
                y=df_norm,
                mode='lines',
                name=f'{ticker} (Principal)',
                line=dict(color='red', width=2)
            ))
            fig_sector.update_layout(
                title=f"Comparativa de {ticker.upper()} vs empresas del sector {info.get('sector', 'N/A')}",
                xaxis_title="Fecha",
                yaxis_title="Rendimiento Normalizado (Base 1)",
                template="plotly_dark",
                legend_title="Empresas",
                xaxis=dict(showgrid=True, zeroline=True),
                yaxis=dict(showgrid=True, zeroline=True),
                height=500
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.error("❌ No se encontraron datos válidos para el ticker principal.")

                
    
    # ------ RENDIMIENTOS ANUALIZADOS ------
    st.markdown("### 📈 Rendimientos Anualizados")

    if 'df' in locals() and not df.empty:
        fecha_actual = df.index[-1]
    else:
        st.error("❌ No se han cargado datos para el ticker principal.")
        fecha_actual = None  # O maneja el flujo que necesites

    precios = df['Price']

    def calcular_rendimiento_anualizado(fecha_objetivo):
        df_filtrado = df[df.index >= fecha_objetivo]
        if not df_filtrado.empty:
            precio_inicio = df_filtrado['Price'].iloc[0]
            precio_final = precios.iloc[-1]
            años = (df_filtrado.index[-1] - df_filtrado.index[0]).days / 365.25
            return round(((precio_final / precio_inicio) ** (1 / años) - 1) * 100, 2)
        return None

    rendimiento_1a = calcular_rendimiento_anualizado(fecha_actual - pd.DateOffset(years=1))
    rendimiento_3a = calcular_rendimiento_anualizado(fecha_actual - pd.DateOffset(years=3))
    rendimiento_5a = calcular_rendimiento_anualizado(fecha_actual - pd.DateOffset(years=5))

    col1, col2, col3 = st.columns(3)
    col1.metric("🔁 Último año", f"{rendimiento_1a}%" if rendimiento_1a else "N/A")
    col2.metric("📊 Últimos 3 años", f"{rendimiento_3a}%" if rendimiento_3a else "N/A")
    col3.metric("📈 Últimos 5 años", f"{rendimiento_5a}%" if rendimiento_5a else "N/A")

    # ------ VOLATILIDAD RIESGO ---------
    st.markdown("### ⚠️ Volatilidad Anualizada")

    # Asegúrate de tener log_return en el DataFrame
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))

    vol_diaria = df['log_return'].std()
    vol_anual = vol_diaria * np.sqrt(252)

    st.metric("📉 Volatilidad Anualizada", f"{round(vol_anual * 100, 2)}%")
    st.markdown("---")

    # -------- RECOMENDACIÓN FINAL --------
    st.markdown("## 🧾 Recomendación Final")

    # Variables clave del análisis
    precio_actual = df['Price'].iloc[-1]
    r1 = rendimiento_1a or 0
    r3 = rendimiento_3a or 0
    r5 = rendimiento_5a or 0
    tendencia_tecnica = "neutral"

    if sma_20 > sma_50 and ultima_rsi < 70:
        tendencia_tecnica = "alcista"
    elif sma_20 < sma_50 and ultima_rsi > 30:
        tendencia_tecnica = "bajista"

    riesgo = vol_anual
    sim_min = p_min
    sim_max = p_max
    sim_mediana = p_median

    # Generar recomendación
    if r1 > 0 and tendencia_tecnica == "alcista" and sim_mediana > precio_actual and riesgo < 0.35:
        mensaje = f"""
        ✅ **La empresa {info['shortName']} muestra señales positivas para los inversionistas.**  
        Presenta un crecimiento anual de {r1}% en el último año, una tendencia técnica alcista y una proyección favorable en los próximos 30 días según la simulación de Monte Carlo (precio estimado mediano: ${sim_mediana}).  
        La volatilidad se mantiene en un rango razonable de {round(riesgo*100,2)}%, lo que la convierte en una opción atractiva de inversión a mediano plazo.
        """
    elif r1 < 0 and tendencia_tecnica == "bajista" and sim_mediana < precio_actual:
        mensaje = f"""
        ⚠️ **Actualmente, {info['shortName']} presenta señales de cautela para nuevos inversionistas.**  
        El rendimiento anualizado ha sido negativo ({r1}%), la tendencia técnica es bajista y el modelo de predicción estima que el precio podría seguir descendiendo en las próximas semanas.  
        Además, la volatilidad anualizada del {round(riesgo*100,2)}% sugiere un riesgo elevado. Puede ser prudente esperar una mejor oportunidad.
        """
    else:
        mensaje = f"""
        ℹ️ **La empresa {info['shortName']} presenta señales mixtas.**  
        Aunque ha tenido un rendimiento anual de {r1}%, la tendencia técnica es {tendencia_tecnica}, y la simulación de Monte Carlo proyecta un precio estimado mediano de ${sim_mediana}, que está{' por encima' if sim_mediana > precio_actual else ' por debajo'} del valor actual.  
        Con una volatilidad del {round(riesgo*100,2)}%, se recomienda analizar el perfil de riesgo del inversionista antes de tomar una decisión.
        """

    st.markdown(mensaje)
    st.markdown("---")


    # -------- FUNCION PARA OBTENER NOTICIAS FINANCIERAS --------
    def obtener_noticias_finnhub(ticker, api_key, max_noticias=5):
        hoy = datetime.today()
        hace_1_mes = hoy - timedelta(days=30)
        url = (
            f'https://finnhub.io/api/v1/company-news?symbol={ticker}'
            f'&from={hace_1_mes.strftime("%Y-%m-%d")}'
            f'&to={hoy.strftime("%Y-%m-%d")}'
            f'&token={api_key}'
        )
        response = requests.get(url)
        if response.status_code != 200:
            return []
        data = response.json()
        if isinstance(data, list):
            noticias = []
            for noticia in data[:max_noticias]:
                titulo = noticia.get('headline', 'Sin título')
                resumen = noticia.get('summary', '')
                fecha = noticia.get('datetime', 0)
                fecha_str = datetime.utcfromtimestamp(fecha).strftime('%Y-%m-%d') if fecha else ''
                noticias.append(f"{fecha_str} - {titulo}. {resumen}")
            return noticias
        return []

    # -------- INICIO DE LA SECCIÓN DE NOTICIAS Y ANALISIS --------
    finnhub_key = "d0ndc09r01qi1cvdmb80d0ndc09r01qi1cvdmb8g"  # Tu API Key Finnhub
    with st.spinner("📰 Cargando noticias recientes y análisis IA..."):
        noticias = obtener_noticias_finnhub(ticker, finnhub_key)

    if noticias:
        texto_noticias = "\n".join(noticias)
        prompt_noticias = f"""
        Analiza estas noticias recientes para la empresa con ticker {ticker}:
        {texto_noticias}

        Resume en 2 puntos clave el sentimiento general, los riesgos principales y las oportunidades para un inversionista. 
        Presenta la información clara, profesional, breve y concisa.
        """

        try:
            respuesta_noticias = model.generate_content(prompt_noticias)
            texto_limpio = respuesta_noticias.text.strip()
            
            # Contenedor visual elegante para mostrar resultados
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                border-radius: 20px;
                padding: 30px 40px;
                margin-top: 50px;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7);
                color: #f0f6fc;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.7;
                font-size: 1.15rem;
            ">
                <h2 style="
                    text-align: center; 
                    color: #ffb347; 
                    margin-bottom: 25px; 
                    font-weight: 800;
                    letter-spacing: 1.5px;
                    text-shadow: 0 0 8px #ffb347;
                ">📰 Análisis IA de Noticias y Sentimiento</h2>
                <p style="white-space: pre-wrap;">{texto_limpio}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error generando análisis IA de noticias: {str(e)}")
    else:
        st.info("No se encontraron noticias recientes para este ticker.")



    # -------- PIE DE PÁGINA --------
    st.markdown("""---""")

    st.markdown("""
    <div style='text-align: center; font-size: 16px; padding-top: 20px; color: #black;'>
    Hecho con 💙 por <strong>Vannia Fernanda Martin Medina</strong> · 2025<br>
    Esta app combina análisis financiero tradicional 📊 con el poder de la IA 🤖<br>
    <em>Se sugiere que la información aquí mostrada se utilice con responsabilidad.</em><br><br>
    </div>
    """, unsafe_allow_html=True)          

    # -------- PÁGINA DE INICIO --------
else:
    st.markdown("""
    <div style='
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
        height: 80vh; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        max-width: 800px;
        margin: 3rem auto;
        text-align: center;
    '>
        <h1 style='font-size: 3.5rem; margin-bottom: 0.3rem;'>🚀 Bienvenido al <br> <strong>Análisis Financiero Avanzado</strong></h1>
        <p style='font-size: 1.5rem; margin-bottom: 2rem; font-weight: 300;'>
            Combina análisis técnico tradicional con el poder de la <strong>Inteligencia Artificial</strong>
        </p>
    """, unsafe_allow_html=True)
