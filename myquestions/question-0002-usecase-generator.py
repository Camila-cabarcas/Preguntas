import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_normalizar_por_categoria():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función normalizar_por_categoria.
    """

    # 1. Configuración aleatoria del número de productos
    n_rows = random.randint(10, 30)

    # 2. Generar datos aleatorios
    producto_id = np.arange(1, n_rows + 1)

    categorias_posibles = ["Electrónica", "Ropa", "Hogar", "Deportes"]
    categoria = np.random.choice(categorias_posibles, size=n_rows)

    precio = np.round(np.random.uniform(5, 500, n_rows), 2)

    df = pd.DataFrame({
        "producto_id": producto_id,
        "categoria": categoria,
        "precio": precio
    })

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------

    input_data = {
        "df": df.copy()
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Aquí replicamos la lógica que debería tener
    #    normalizar_por_categoria
    # ---------------------------------------------------------

    df_expected = df.copy()

    # calcular precio máximo por categoría
    max_por_categoria = df_expected.groupby("categoria")["precio"].transform("max")

    # calcular precio relativo
    df_expected["precio_relativo"] = df_expected["precio"] / max_por_categoria

    output_data = df_expected

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":

    # Generamos un caso
    entrada, salida_esperada = generar_caso_de_uso_normalizar_por_categoria()

    print("=== INPUT (Diccionario) ===")
    print("DataFrame (primeras 5 filas):")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print("DataFrame con precio_relativo:")
    print(salida_esperada.head())
