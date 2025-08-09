import joblib
import numpy as np

# Cargar el scaler
scaler = joblib.load("Scaler.pkl")

# Verificar si está entrenado
print("¿Está el scaler entrenado?")
try:
    # Intentar acceder a los atributos que se crean después del fit
    print(f"Mean: {scaler.mean_}")
    print(f"Scale: {scaler.scale_}")
    print("✅ El scaler SÍ está entrenado")
except AttributeError:
    print("❌ El scaler NO está entrenado")

# Ver todos los atributos del scaler
print("\nAtributos del scaler:")
print([attr for attr in dir(scaler) if not attr.startswith('_')])