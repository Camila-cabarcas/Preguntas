import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# =====================================================
# NUEVA FUNCIÓN
# =====================================================
def entrenar_y_evaluar_knn(df, k):

    # Separar X e y
    X_data = df.drop(columns=["target"])

    y_data = df["target"]

    # Escalar datos
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_data)

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_data,
        test_size=0.2,
        random_state=42
    )

    # Entrenar modelo
    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    return acc


def generar_caso_de_uso_entrenar_y_evaluar_knn():

    n = random.randint(20, 60)

    X = np.random.randn(n, 3)

    y = np.random.randint(0, 2, n)

    df = pd.DataFrame(
        X,
        columns=["f1", "f2", "f3"]
    )

    df["target"] = y

    k = random.randint(1, 5)

    input_data = {
        "df": df.copy(),
        "k": k
    }

    # =====================================================
    # USAR LA NUEVA FUNCIÓN
    # =====================================================
    acc = entrenar_y_evaluar_knn(df, k)

    return input_data, acc


# =====================================================
# EJEMPLO DE USO
# =====================================================
if __name__ == "__main__":

    entrada, salida = (
        generar_caso_de_uso_entrenar_y_evaluar_knn()
    )

    print("=== INPUT ===")
    print(entrada)

    print("\n=== ACCURACY ===")
    print(salida)
