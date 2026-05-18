import numpy as np
import pandas as pd

# NUEVAS IMPORTACIONES
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def eliminar_multicolinealidad(df, threshold):
    # Seleccionar columnas numéricas
    df_numeric = df.select_dtypes(include=[np.number])

    # Calcular matriz de correlación absoluta
    corr_matrix = df_numeric.corr().abs()

    # Tomar solo la parte superior de la matriz
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Identificar columnas a eliminar
    columnas_eliminadas = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    # Eliminar columnas
    df_filtrado = df_numeric.drop(columns=columnas_eliminadas)

    return df_filtrado, columnas_eliminadas


# =====================================================
# FUNCIÓN ORIGINAL (NO LA ELIMINES)
# =====================================================
def generar_caso_de_uso_eliminar_multicolinealidad():

    np.random.seed()

    # Número aleatorio de filas y columnas
    n_filas = np.random.randint(30, 100)
    n_cols = np.random.randint(4, 8)

    data = {}

    # Generar columnas base
    for i in range(n_cols):
        data[f"col_{i}"] = np.random.randn(n_filas)

    df = pd.DataFrame(data)

    # Introducir correlación artificial
    if n_cols >= 2:
        col_base = np.random.choice(df.columns)
        col_nueva = np.random.choice(df.columns)

        if col_base != col_nueva:
            df[col_nueva] = df[col_base] * (
                0.8 + 0.2 * np.random.rand()
            )

    # Threshold aleatorio
    threshold = np.random.uniform(0.7, 0.95)

    # Crear input
    input_data = {
        "df": df,
        "threshold": threshold
    }

    # Calcular output esperado
    df_filtrado, columnas_eliminadas = eliminar_multicolinealidad(
        df,
        threshold
    )

    output_data = (df_filtrado, columnas_eliminadas)

    return input_data, output_data


# =====================================================
# NUEVA FUNCIÓN
# =====================================================
def clasificar_pulsos_radio(df, target_col):

    # Separar variables y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Eliminar multicolinealidad
    X_filtrado, columnas_eliminadas = eliminar_multicolinealidad(
        X,
        threshold=0.9
    )

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtrado,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5))
    ])

    # Entrenamiento
    pipeline.fit(X_train, y_train)

    # Predicciones
    y_pred = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "modelo": pipeline.named_steps["modelo"],
        "accuracy": accuracy
    }


# =====================================================
# NUEVO CASO DE USO
# =====================================================
def generar_caso_de_uso_clasificar_pulsos_radio():

    np.random.seed()

    n_filas = np.random.randint(80, 150)

    df = pd.DataFrame({
        "frecuencia_central": np.random.uniform(100, 500, n_filas),
        "ancho_banda": np.random.uniform(10, 50, n_filas),
        "flujo": np.random.uniform(0.1, 10, n_filas),
        "relacion_señal_ruido": np.random.uniform(1, 100, n_filas),
    })

    # Correlación artificial
    df["flujo_correlacionado"] = (
        df["flujo"] * (0.8 + 0.2 * np.random.rand())
    )

    # Objetivo binario
    df["objetivo"] = np.random.randint(0, 2, n_filas)

    input_data = {
        "df": df,
        "target_col": "objetivo"
    }

    output_data = clasificar_pulsos_radio(
        df,
        "objetivo"
    )

    return input_data, output_data


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    # CASO ORIGINAL
    input_data, output_data = (
        generar_caso_de_uso_eliminar_multicolinealidad()
    )

    print("INPUT:")
    print(input_data)

    print("\nOUTPUT:")
    print(output_data)

    # NUEVO CASO
    input_data_2, output_data_2 = (
        generar_caso_de_uso_clasificar_pulsos_radio()
    )

    print("\n==============================")
    print("CASO DE USO - CLASIFICAR PULSOS")
    print("==============================")

    print("\nINPUT:")
    print(input_data_2)

    print("\nOUTPUT:")
    print(output_data_2)
