import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analizar_churn():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función analizar_churn.
    """

    # 1. Número aleatorio de clientes
    n = random.randint(10, 30)

    # 2. Crear datos aleatorios
    pago_mensual = np.random.uniform(10, 200, n)
    antiguedad = np.random.randint(1, 36, n)

    cancelo = np.random.choice(["Si", "No"], size=n)

    df = pd.DataFrame({
        "pago_mensual": pago_mensual,
        "antiguedad": antiguedad,
        "cancelo": cancelo
    })

    # -------------------------------------------------
    # INPUT
    # -------------------------------------------------

    input_data = {
        "df": df.copy()
    }

    # -------------------------------------------------
    # OUTPUT ESPERADO
    # -------------------------------------------------

    promedio = df["pago_mensual"].mean()

    clientes_valiosos = df[
        (df["pago_mensual"] > promedio) &
        (df["antiguedad"] > 12)
    ]

    if len(clientes_valiosos) == 0:
        tasa = 0.0
    else:
        tasa = (clientes_valiosos["cancelo"] == "Si").mean()

    output_data = round(float(tasa), 4)

    return input_data, output_data


# Ejemplo de uso
if __name__ == "__main__":

    entrada, salida = generar_caso_de_uso_analizar_churn()

    print("INPUT:")
    print(entrada["df"].head())

    print("\nOUTPUT ESPERADO:")
    print(salida)
