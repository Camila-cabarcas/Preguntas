import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analizar_churn():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función analizar_churn.
    """

    # 1. Configuración aleatoria del número de clientes
    n_rows = random.randint(10, 30)   # Entre 10 y 30 clientes

    # 2. Generar datos aleatorios
    pago_mensual = np.random.uniform(20, 200, n_rows)
    antiguedad = np.random.randint(1, 48, n_rows)
    cancelo = np.random.choice(["Si", "No"], size=n_rows)

    df = pd.DataFrame({
        "pago_mensual": pago_mensual,
        "antiguedad": antiguedad,
        "cancelo": cancelo
    })

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------

    input_data = {
        'df': df.copy()  # Copia para evitar modificar el original
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Aquí replicamos la lógica que debería tener analizar_churn
    # ---------------------------------------------------------

    # A. Calcular promedio de pago mensual
    promedio_pago = df["pago_mensual"].mean()

    # B. Filtrar clientes valiosos
    clientes_valiosos = df[
        (df["pago_mensual"] > promedio_pago) &
        (df["antiguedad"] > 12)
    ]

    # C. Calcular tasa de cancelación
    if len(clientes_valiosos) == 0:
        tasa_cancelacion = 0.0
    else:
        tasa_cancelacion = (clientes_valiosos["cancelo"] == "Si").mean()

    output_data = round(float(tasa_cancelacion), 4)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":

    # Generamos un caso
    entrada, salida_esperada = generar_caso_de_uso_analizar_churn()

    print("=== INPUT (Diccionario) ===")
    print("DataFrame (primeras 5 filas):")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print("Tasa de cancelación esperada:", salida_esperada)
