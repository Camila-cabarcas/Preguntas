import numpy as np
import random
from sklearn.datasets import make_classification

# NUEVAS IMPORTACIONES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def generar_caso_de_uso_calibrar_y_comparar():

    perfiles = {
        "diagnostico_medico": dict(
            n_samples=400,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            weights=[0.7, 0.3]
        ),

        "deteccion_spam": dict(
            n_samples=600,
            n_features=15,
            n_informative=8,
            n_redundant=4,
            weights=[0.8, 0.2]
        ),

        "aprobacion_credito": dict(
            n_samples=300,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            weights=[0.6, 0.4]
        ),

        "fallo_equipo": dict(
            n_samples=500,
            n_features=12,
            n_informative=4,
            n_redundant=5,
            weights=[0.85, 0.15]
        ),
    }

    nombre, params = random.choice(list(perfiles.items()))
    seed = random.randint(0, 999)

    X, y = make_classification(
        random_state=seed,
        n_clusters_per_class=1,
        flip_y=random.uniform(0.02, 0.08),
        **{k: v for k, v in params.items() if k != "weights"},
    )

    # Aplicar desbalance aproximado
    print(f"Perfil: '{nombre}'")

    print(
        f"X shape: {X.shape} | Distribución de clases: "
        f"{dict(zip(*np.unique(y, return_counts=True)))}"
    )

    return X, y


# =====================================================
# NUEVA FUNCIÓN
# =====================================================
def calibrar_y_comparar(X, y):

    # 1. División train/test 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    # 2. Estandarización
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # 3. Random Forest
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X_train_scaled, y_train)

    # Probabilidades sin calibrar
    probs_sin_calibrar = modelo.predict_proba(X_test_scaled)[:, 1]

    # 4. Calibración isotónica
    modelo_calibrado = CalibratedClassifierCV(
        estimator=modelo,
        cv="prefit",
        method="isotonic"
    )

    modelo_calibrado.fit(X_test_scaled, y_test)

    # Probabilidades calibradas
    probs_calibradas = (
        modelo_calibrado.predict_proba(X_test_scaled)[:, 1]
    )

    # 5. Brier Score
    brier_sin_calibrar = brier_score_loss(
        y_test,
        probs_sin_calibrar
    )

    brier_calibrado = brier_score_loss(
        y_test,
        probs_calibradas
    )

    # Mejora
    mejora = brier_sin_calibrar - brier_calibrado

    return {
        "brier_sin_calibrar": brier_sin_calibrar,
        "brier_calibrado": brier_calibrado,
        "mejora": mejora
    }


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    # CASO DE USO ORIGINAL
    X, y = generar_caso_de_uso_calibrar_y_comparar()

    # NUEVA FUNCIÓN
    resultado = calibrar_y_comparar(X, y)

    print("\n==============================")
    print("RESULTADOS CALIBRACIÓN")
    print("==============================")

    print(resultado)
