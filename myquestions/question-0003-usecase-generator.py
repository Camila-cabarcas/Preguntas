import numpy as np
import pandas as pd
import random

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression


def generar_caso_de_uso_entrenar_con_seleccion():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función entrenar_con_seleccion.
    """

    # ---------------------------------------------------------
    # 1. Configuración aleatoria de dimensiones
    # ---------------------------------------------------------

    n_samples = random.randint(40, 120)
    n_features = random.randint(5, 10)

    # k_features debe ser menor que n_features
    k_features = random.randint(1, n_features - 1)

    # ---------------------------------------------------------
    # 2. Generar datos aleatorios
    # ---------------------------------------------------------

    X = np.random.randn(n_samples, n_features)

    # Crear una relación lineal con ruido
    coef_reales = np.random.randn(n_features)
    y = X @ coef_reales + np.random.randn(n_samples) * 0.5

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------

    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "k_features": k_features
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------

    # A. Selección de características
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_selected = selector.fit_transform(X, y)

    # B. Entrenar modelo de regresión
    modelo = LinearRegression()
    modelo.fit(X_selected, y)

    output_data = (selector, modelo)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":

    entrada, salida_esperada = generar_caso_de_uso_entrenar_con_seleccion()

    print("=== INPUT ===")
    print("Shape de X:", entrada["X"].shape)
    print("Shape de y:", entrada["y"].shape)
    print("Número de features a seleccionar:", entrada["k_features"])

    print("\n=== OUTPUT ESPERADO ===")
    selector, modelo = salida_esperada

    print("Features seleccionadas:", selector.get_support())
    print("Coeficientes del modelo:", modelo.coef_)
