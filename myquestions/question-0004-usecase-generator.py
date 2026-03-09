import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def generar_caso_de_uso_clasificador_balanceado():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función clasificador_balanceado.
    """

    # ---------------------------------------------------------
    # 1. Configuración aleatoria de dimensiones
    # ---------------------------------------------------------

    n_samples = random.randint(120, 300)
    n_features = random.randint(5, 12)

    # ---------------------------------------------------------
    # 2. Generar datos aleatorios
    # ---------------------------------------------------------

    X = np.random.randn(n_samples, n_features)

    # Generar clases desbalanceadas (aprox 10% fallos)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------

    input_data = {
        "X": X.copy(),
        "y": y.copy()
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    modelo = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    f1 = f1_score(y_test, y_pred)

    output_data = float(f1)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":

    entrada, salida_esperada = generar_caso_de_uso_clasificador_balanceado()

    print("=== INPUT ===")
    print("Shape de X:", entrada["X"].shape)
    print("Shape de y:", entrada["y"].shape)

    print("\n=== OUTPUT ESPERADO ===")
    print("F1-Score esperado:", salida_esperada)
