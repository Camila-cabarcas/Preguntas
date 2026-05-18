import pandas as pd
import numpy as np
import random


# =====================================================
# NUEVA FUNCIÓN
# =====================================================
def resumir_partidas_por_mapa(df):

    # Eliminar filas con NaN
    df_clean = df.dropna().copy()

    # Crear columna auxiliar
    df_clean['victoria_local'] = (
        df_clean['equipo_ganador'] ==
        df_clean['equipo_local']
    )

    # Agrupar y resumir
    resumen = (
        df_clean.groupby('mapa', as_index=False)
        .agg(
            total_partidas=('puntaje_local', 'count'),

            promedio_puntaje_local=(
                'puntaje_local',
                'mean'
            ),

            promedio_puntaje_visitante=(
                'puntaje_visitante',
                'mean'
            ),

            tasa_victoria_local=(
                'victoria_local',
                'mean'
            )
        )
        .sort_values(
            'tasa_victoria_local',
            ascending=False
        )
        .reset_index(drop=True)
    )

    return resumen


def generar_caso_de_uso_resumir_partidas_por_mapa():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función resumir_partidas_por_mapa(df).
    """

    # --- Configuración aleatoria ---
    mapas_posibles = [
        'Dust2',
        'Mirage',
        'Inferno',
        'Nuke',
        'Overpass',
        'Ancient',
        'Vertigo',
        'Cache',
        'Train',
        'Cobblestone'
    ]

    n_mapas = random.randint(3, 6)

    mapas_elegidos = random.sample(
        mapas_posibles,
        n_mapas
    )

    n_partidas = random.randint(12, 30)

    mapas = np.random.choice(
        mapas_elegidos,
        size=n_partidas
    )

    equipos = ['A', 'B']

    equipo_loc = np.random.choice(
        equipos,
        size=n_partidas
    )

    equipo_gan = np.random.choice(
        equipos,
        size=n_partidas
    )

    pts_local = np.random.randint(
        5,
        20,
        size=n_partidas
    ).astype(float)

    pts_visit = np.random.randint(
        5,
        20,
        size=n_partidas
    ).astype(float)

    df = pd.DataFrame({
        'mapa': mapas,
        'equipo_ganador': equipo_gan,
        'equipo_local': equipo_loc,
        'puntaje_local': pts_local,
        'puntaje_visitante': pts_visit
    })

    # Introducir NaNs aleatorios (~10%)
    for col in ['puntaje_local', 'puntaje_visitante']:

        mask = np.random.choice(
            [True, False],
            size=n_partidas,
            p=[0.10, 0.90]
        )

        df.loc[mask, col] = np.nan

    # Algunos NaN en columnas de texto
    for col in ['equipo_ganador', 'equipo_local']:

        idx = np.random.choice(
            df.index,
            size=max(1, n_partidas // 12),
            replace=False
        )

        df.loc[idx, col] = np.nan

    # --- Input ---
    input_data = {'df': df.copy()}

    # =====================================================
    # USAR LA NUEVA FUNCIÓN
    # =====================================================
    output_data = resumir_partidas_por_mapa(df)

    return input_data, output_data


# =====================================================
# EJEMPLO DE USO
# =====================================================
if __name__ == "__main__":

    entrada, salida_esperada = (
        generar_caso_de_uso_resumir_partidas_por_mapa()
    )

    print("=== INPUT ===")
    print(entrada['df'])

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
