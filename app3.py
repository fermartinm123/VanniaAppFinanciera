import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import os

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
            .sidebar-title {
                font-size: 26px;
                font-weight: 800;
                color: #00C2FF;
                margin-bottom: 5px;
                font-family: 'Segoe UI', sans-serif;
            }
            .sidebar-sub {
                font-size: 16px;
                color: #black;
                font-style: italic;
                margin-bottom: 15px;
            }
            .sidebar-section {
                font-size: 18px;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 10px;
                color: #black;
            }
        </style>
        <div class='sidebar-title'>📊 Análisis Financiero </div>
        <div class='sidebar-sub'>Potenciado con IA</div>
    """, unsafe_allow_html=True)

    # Ticker Input
    st.markdown("<div class='sidebar-section'>💼 Ticker de la Empresa</div>", unsafe_allow_html=True)
    ticker = st.text_input("Ingresa el símbolo bursátil de tu elección (Ej. AAPL)", "AAPL")

    # Fechas
    st.markdown("<div class='sidebar-section'>📆 Fechas a Consultar</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Desde:", datetime.today() - timedelta(days=5*365))
    with col2:
        end_date = st.date_input("Hasta:", datetime.today())

    # Opciones adicionales
    st.markdown("<div class='sidebar-section'>⚙️ Opciones Adicionales</div>", unsafe_allow_html=True)
    show_comparativa = st.checkbox("Comparar con sector", value=True)
    show_descripcion = st.checkbox("Mostrar descripción de la empresa", value=True)
    show_historico = st.checkbox("Ver histórico de precios", value=True)

    # Botones decorativos
    st.markdown("<div class='sidebar-section'>📁 Exportar</div>", unsafe_allow_html=True)
    st.button("⬇️ Descargar CSV")
    st.button("📄 Generar PDF")

    # Submit
    st.markdown("---")
    submit = st.button("🚀 Generar Análisis")

    st.caption("© 2025 VanniaAPP - Todos los derechos reservados.")


# -------- FUNCIÓN DE VALIDACIÓN --------
def validar_ticker(ticker):
    try:
        data = yf.Ticker(ticker)
        info = data.info
        if "shortName" in info:
            return info
    except:
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

        # -------- SIMULACIÓN MONTECARLO --------
        st.markdown("### 🔮 Simulación Monte Carlo - Predicción a 30 días")

        # Número de simulaciones y días
        num_simulaciones = 500
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
        for i in range(num_simulaciones):
            fig_mc.add_trace(go.Scatter(
                x=list(range(1, num_dias+1)),
                y=simulaciones[:, i],
                mode='lines',
                line=dict(width=0.5),
                opacity=0.2,
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
        st.markdown("### 📌 Conclusión de la Simulación")

        precio_finales = simulaciones[-1]
        p_min = round(np.percentile(precio_finales, 5), 2)
        p_max = round(np.percentile(precio_finales, 95), 2)
        p_median = round(np.median(precio_finales), 2)

        st.markdown(f"""
        🧠 Basado en {num_simulaciones} simulaciones de Monte Carlo:
        - 📉 Hay un 90% de probabilidad de que el precio esté entre **${p_min}** y **${p_max}** en 30 días.
        - 📍 El precio mediano estimado es **${p_median}**.
        """)

        # Análisis sencillo de tendencia esperada
        if p_median > precio_actual:
            st.success("📈 Las simulaciones sugieren una **tendencia alcista** en el próximo mes.")
        elif p_median < precio_actual:
            st.warning("📉 Las simulaciones indican una posible **corrección o baja** en el precio.")
        else:
            st.info("➖ Las simulaciones no muestran una dirección clara del precio.")


        # -------- COMPARATIVA CON EMPRESAS DEL SECTOR --------
        st.markdown("### 🏢 Tabla Comparativa con Empresas del Sector")

        fig_sector = go.Figure()
        for sec_ticker, sec_df in sector_data.items():
            if not sec_df.empty:
                fig_sector.add_trace(go.Scatter(
                    x=sec_df.index,
                    y=sec_df['Price'],
                    mode='lines',
                    name=f'{sec_ticker}'
                ))

        # Agrega el ticker principal
        if not df.empty:
            fig_sector.add_trace(go.Scatter(
                x=df.index,
                y=df['Price'],
                mode='lines',
                name=f'{ticker} (Principal)',
                line=dict(color='red', width=2)
            ))
            fig_sector.update_layout(
                title=f"Comparativa de {ticker.upper()} vs empresas del sector {info.get('sector', 'N/A')}",
                xaxis_title="Fecha",
                yaxis_title="Precio Ajustado (USD)",
                template="plotly_dark",
                legend_title="Empresas",
                xaxis=dict(showgrid=True, zeroline=True),
                yaxis=dict(showgrid=True, zeroline=True),
                height=500
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.error("❌ No se encontraron datos válidos para el ticker principal.")
    else:
        st.error("❌ **Ticker inválido. Por favor, verifica el símbolo ingresado.**")
    
    # ------ RENDIMIENTOS ANUALIZADOS ------
    st.markdown("### 📈 Rendimientos Anualizados")

    fecha_actual = df.index[-1]
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

    # -------- PIE DE PÁGINA --------
    st.markdown("""---""")

    st.markdown("""
    <div style='text-align: center; font-size: 16px; padding-top: 20px; color: #black;'>
    Hecho con 💙 por <strong>Vannia Fernanda Martin Medina</strong> · 2025<br>
    Esta app combina análisis financiero tradicional 📊 con el poder de la IA 🤖<br>
    <em>Se sugiere que la información aquí mostrada se utilice con responsabilidad.</em><br><br>
    📧 ¿Sugerencias o mejoras? Jajajaja no creo! | Me merezco un 10 Profe! </em><br><br>
    Le deseo Bonitas Vacaciones 🏝️☀️🌊
    </div>
    """, unsafe_allow_html=True)

    

    




