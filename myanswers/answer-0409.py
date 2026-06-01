import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def clasificar_pulsos_radio(df: pd.DataFrame, target_col: str) -> dict:
    """
    Limpia datos, construye pipeline de preprocesamiento y clasifica pulsos
    de radio como provenientes de estrellas de neutrones o de otras fuentes.

    Args:
        df: DataFrame con columnas frecuencia_central, ancho_banda, flujo,
            relacion_señal_ruido y la columna objetivo.
        target_col: Nombre de la columna objetivo.

    Returns:
        dict: {'modelo': KNeighborsClassifier, 'accuracy': float}
    """
    # Separar features y target
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # 1. Imputar valores nulos con la mediana
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # 2. Transformar a distribución uniforme para reducir efecto de outliers
    qt = QuantileTransformer(output_distribution="uniform", random_state=42)
    X_qt = qt.fit_transform(X_imp)

    # 3. Escalar a media 0 y desviación estándar 1
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X_qt)

    # 4. Entrenar clasificador KNN
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_proc, y)

    accuracy = round(float(modelo.score(X_proc, y)), 4)

    return {"modelo": modelo, "accuracy": accuracy}


# ── Prueba rápida ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 120

    frecuencia_central = rng.gamma(shape=2.0, scale=200.0, size=n)
    ancho_banda        = rng.exponential(scale=15.0, size=n)
    flujo              = rng.gamma(shape=1.5, scale=5.0, size=n)
    snr                = rng.exponential(scale=8.0, size=n)
    es_neutron         = ((flujo > np.median(flujo)) & (snr > np.median(snr))).astype(int)

    df = pd.DataFrame({
        "frecuencia_central":  frecuencia_central,
        "ancho_banda":         ancho_banda,
        "flujo":               flujo,
        "relacion_señal_ruido": snr,
        "es_neutron":          es_neutron,
    })

    # Introducir ~10 % de NaN en features
    feature_cols = [c for c in df.columns if c != "es_neutron"]
    for col in feature_cols:
        mask = rng.random(n) < 0.10
        df.loc[mask, col] = np.nan

    resultado = clasificar_pulsos_radio(df.copy(), "es_neutron")
    print(f"Modelo  : {resultado['modelo']}")
    print(f"Accuracy: {resultado['accuracy']}")
