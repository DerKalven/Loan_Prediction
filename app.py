import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import time

st.set_page_config(
    page_title="Loan Status Predictor - Data Science Portfolio",
    page_icon="💳",
    layout="wide"
)

# Header con información del proyecto
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: #1f77b4; margin-bottom: 10px;'>💳 Loan Status Prediction App</h1>
    <p style='font-size: 18px; color: #2c3e50; font-weight: 500;'>Predictor de Estado de Préstamos usando Machine Learning</p>
    <p style='font-size: 14px; color: #e74c3c; font-weight: bold;'>⚠️ Modelo creado con fines educativos</p>
</div>
""", unsafe_allow_html=True)


# Cargar modelos con mejor manejo de errores
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("model.pkl")

        # Verificar si el scaler está entrenado
        try:
            _ = scaler.mean_  # Esto fallará si no está fitted
            st.success("✅ Modelos cargados correctamente")
            return scaler, model, True
        except AttributeError:
            st.warning("⚠️ El scaler no está entrenado. Usando normalización manual.")
            return scaler, model, False

    except FileNotFoundError as e:
        st.error(f"❌ Error: No se pudo cargar el archivo: {e}")
        return None, None, False


# Función de animación combinada (texto + emojis)
def animated_result(is_approved):
    """Animación combinada de texto y emojis"""
    progress_text = st.empty()

    if is_approved:
        # Secuencia para aprobado
        animations = [
            ("🔍", "Analizando datos..."),
            ("⚖️", "Evaluando perfil crediticio..."),
            ("✅", "Verificando información..."),
            ("🎉", "¡PRÉSTAMO APROBADO!"),
            ("🥳", "¡Felicitaciones!")
        ]
        color = "green"
    else:
        # Secuencia para rechazado
        animations = [
            ("🔍", "Analizando datos..."),
            ("⚖️", "Evaluando perfil crediticio..."),
            ("📊", "Revisando criterios..."),
            ("❌", "PRÉSTAMO NO APROBADO"),
            ("💪", "¡No te desanimes, sigue mejorando!")
        ]
        color = "red"

    # Mostrar cada paso de la animación
    for emoji, text in animations:
        progress_text.markdown(
            f"""
            <div style='text-align: center; padding: 20px;'>
                <h1 style='font-size: 80px; margin: 0;'>{emoji}</h1>
                <h3 style='color: {color}; margin: 10px 0;'>{text}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.8)

    progress_text.empty()


scaler, model, scaler_fitted = load_models()

if scaler is None or model is None:
    st.stop()

# Información del dataset y modelo
with st.expander("📊 Información del Dataset y Modelo", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Dataset")
        st.markdown(
            "**Fuente:** [Kaggle - Loan Status Prediction](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction)")
        st.markdown("**Descripción:** Dataset para predicción de aprobación de préstamos")
        st.markdown("**Variables principales:**")
        st.markdown("- Estado Civil")
        st.markdown("- Ingresos Anuales")
        st.markdown("- Nivel Educativo")
        st.markdown("- Monto del Préstamo")
        st.markdown("- Historial Crediticio")

    with col2:
        st.markdown("### 🤖 Modelo de Machine Learning")
        st.markdown("**Algoritmo:** Support Vector Classifier (SVC)")
        st.markdown("**Parámetros:**")
        st.markdown("- `C = 0.05` (Parámetro de regularización)")
        st.markdown("- `kernel = 'linear'` (Kernel lineal)")
        st.markdown("**Preprocesamiento:** StandardScaler")
        st.markdown("**Finalidad:** Proyecto educativo de Data Science")

st.divider()

# Inputs del usuario
col1, col2, col3 = st.columns(3)

with col1:
    married = st.selectbox("Client is Married?", ["No", "Yes"])
    education = st.selectbox("Client is graduated?", ["No", "Yes"])

with col2:
    income = st.number_input("Annual Income", min_value=0, value=10000, step=500)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000, step=500)

with col3:
    credit_history = st.selectbox("Credit History?", ["No", "Yes"])

# Selector de tipo de animación (removido)
# animation_type = st.radio(
#     "Elige el tipo de animación:",
#     ["Clásica (Nieve/Globos)", "Emojis Animados", "Texto Animado", "Sin Animación"]
# )

predictbutton = st.button("Predict the Approval!", type="primary")

st.divider()

# Procesar predicción solo cuando se presiona el botón
if predictbutton:
    try:
        # Conversión de variables categóricas
        marriedstatus = 1 if married == "Yes" else 0
        educationstatus = 1 if education == "Yes" else 0
        credit_historystatus = 1 if credit_history == "Yes" else 0

        # Crear array con todas las características
        features = np.array([[marriedstatus, income, educationstatus, loan_amount, credit_historystatus]], dtype=float)

        # Aplicar escalado
        if scaler_fitted:
            X_scaled = scaler.transform(features)
        else:
            X_scaled = features
            st.info("ℹ️ Usando datos sin escalar")

        # Hacer predicción
        prediction = model.predict(X_scaled)
        probability = None

        # Intentar obtener probabilidades si está disponible
        try:
            probability = model.predict_proba(X_scaled)
        except:
            pass

        # ANIMACIÓN COMBINADA (Texto + Emojis)
        animated_result(prediction[0] == 1)

        # Mostrar resultado final con información adicional
        st.markdown("### 📊 Resultados de la Predicción")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction[0] == 1:
                st.success("🎉 ¡APROBADO!")
                st.markdown("#### 🎊 ¡Felicitaciones!")
                st.markdown("Tu préstamo ha sido **aprobado** según el modelo SVC.")
            else:
                st.error("❌ NO APROBADO")
                st.markdown("#### 💡 Resultado del Análisis")
                st.markdown("El modelo SVC **no recomienda** aprobar este préstamo.")

        with col2:
            if probability is not None:
                prob_approved = probability[0][1] * 100
                st.metric("Probabilidad de Aprobación", f"{prob_approved:.1f}%")

                # Barra de progreso visual
                st.progress(prob_approved / 100)

                # Interpretación de probabilidad
                if prob_approved < 30:
                    st.markdown("🔻 **Probabilidad Baja**")
                elif prob_approved < 60:
                    st.markdown("🔶 **Probabilidad Moderada**")
                else:
                    st.markdown("🔺 **Probabilidad Alta**")

        with col3:
            st.markdown("#### 📋 Modelo Utilizado")
            st.markdown("**SVC (Support Vector Classifier)**")
            st.markdown("- C = 0.05")
            st.markdown("- Kernel = Linear")
            st.markdown("- Preprocesamiento: StandardScaler")

        # Recomendaciones basadas en resultado
        st.markdown("### 💡 Interpretación y Recomendaciones")

        if prediction[0] == 1:
            st.info("""
            **✅ Perfil Aprobado:** El modelo identifica que este perfil cumple con los criterios 
            típicos para la aprobación de préstamos según los patrones encontrados en el dataset de entrenamiento.
            """)
        else:
            st.warning("""
            **❌ Perfil No Aprobado:** El modelo sugiere que este perfil presenta características 
            que históricamente se asocian con mayor riesgo de impago según el dataset de entrenamiento.

            **Posibles mejoras para futuras solicitudes:**
            - Mejorar el historial crediticio
            - Aumentar los ingresos
            - Considerar un monto de préstamo menor
            """)

        # Disclaimer importante
        st.markdown("### ⚠️ Importante")
        st.warning("""
        **Este es un modelo educativo** basado en datos de Kaggle y no debe utilizarse para 
        decisiones reales de préstamos. Los algoritmos de ML pueden tener sesgos y las decisiones 
        financieras reales requieren análisis más complejos y regulados.
        """)

        # Mostrar valores de entrada para debug
        with st.expander("🔍 Valores de entrada (debug)"):
            st.write(f"Married: {marriedstatus}")
            st.write(f"Income: {income}")
            st.write(f"Education: {educationstatus}")
            st.write(f"Loan Amount: {loan_amount}")
            st.write(f"Credit History: {credit_historystatus}")

    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")

        with st.expander("🐛 Detalles del error"):
            import traceback

            st.code(traceback.format_exc())

# Información adicional
with st.sidebar:
    st.markdown("### 👨‍💻 Información del Proyecto")

    st.markdown("#### 📊 Dataset")
    st.markdown("**Fuente:** Kaggle")
    st.markdown("**Nombre:** Loan Status Prediction")
    st.markdown("🔗 [Ver Dataset](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction)")

    st.markdown("#### 🤖 Modelo")
    st.markdown("**Algoritmo:** SVC")
    st.markdown("**Parámetros:**")
    st.markdown("- C = 0.05")
    st.markdown("- kernel = 'linear'")

    st.markdown("#### 🎯 Variables del Modelo")
    st.markdown("1. Estado Civil")
    st.markdown("2. Ingresos Anuales")
    st.markdown("3. Educación")
    st.markdown("4. Monto del Préstamo")
    st.markdown("5. Historial Crediticio")

    if scaler_fitted:
        st.success("Scaler: ✅ Entrenado")
    else:
        st.warning("Scaler: ⚠️ No entrenado")

    st.divider()

    st.markdown("#### 🎬 Animación")
    st.markdown("**Secuencia animada** que combina emojis con mensajes de progreso")
    st.markdown("- ✅ **Aprobado:** 🔍 → ⚖️ → ✅ → 🎉 → 🥳")
    st.markdown("- ❌ **Rechazado:** 🔍 → ⚖️ → 📊 → ❌ → 💪")

    st.divider()

    st.markdown("#### ⚠️ Aviso Legal")
    st.markdown(
        "**Este modelo fue creado con fines educativos.** No debe utilizarse para decisiones reales de préstamos.")

    # Información de contacto (opcional - puedes personalizarlo)
    st.markdown("#### 📫 Portfolio")
    st.markdown("🐙 [GitHub](https://github.com)")
    st.markdown("💼 [LinkedIn](https://linkedin.com)")
    st.markdown("📧 email@ejemplo.com")
