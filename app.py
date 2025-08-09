import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import time

st.set_page_config(layout="wide")


# Cargar modelos con mejor manejo de errores
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("Scaler.pkl")
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

st.title("Loan Status Prediction App")
st.caption("This app helps you to predict a Loan Status")
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

        # Mostrar resultado final
        col1, col2 = st.columns(2)

        with col1:
            if prediction[0] == 1:
                st.success("🎉 ¡APROBADO!")
                st.markdown("### 🎊 ¡Felicitaciones! Tu préstamo ha sido aprobado.")
            else:
                st.error("❌ NO APROBADO")
                st.markdown("### 💡 No te desanimes, puedes mejorar tu perfil crediticio.")

        with col2:
            if probability is not None:
                prob_approved = probability[0][1] * 100
                st.metric("Probabilidad de Aprobación", f"{prob_approved:.1f}%")

                # Barra de progreso visual
                st.progress(prob_approved / 100)

                # Recomendaciones basadas en probabilidad
                if prob_approved < 30:
                    st.warning("🔻 Probabilidad muy baja. Considera mejorar tu historial crediticio.")
                elif prob_approved < 60:
                    st.info("🔶 Probabilidad moderada. Podrías mejorar algunos aspectos.")
                else:
                    st.success("🔺 ¡Excelente probabilidad!")

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
    st.header("ℹ️ Información del Modelo")
    st.write("**Variables utilizadas:**")
    st.write("- Estado Civil")
    st.write("- Ingresos Anuales")
    st.write("- Educación")
    st.write("- Monto del Préstamo")
    st.write("- Historial Crediticio")

    if scaler_fitted:
        st.success("Scaler: ✅ Entrenado")
    else:
        st.warning("Scaler: ⚠️ No entrenado")

