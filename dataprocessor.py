import numpy as np
import pandas as pd

def datos_test_train(df, test_size : float, seed = None):
    r"""
    Esta función procesa los datos y crea dos conjuntos:
    uno de entrenamiento, data_train, y otro de prueba, data_test.

    Parámetros
    ----------
        - df : str or pd.DataFrame
            ruta a un archivo `.csv` que contiene los datos
        - test_size : float
            proporción de los datos de prueba. Número entre 0 y 1
        - seed : int, opcional
            semilla para la mezcla aleatoria. Si se indica, la mezcla
            es siempre la misma (reproducible)

    Devuelve
    --------
        - data_train, data_test : pd.DataFrame
    """
    msg =  "The test size must be a number between 0 and 1"
    assert 0 < test_size < 1, msg
    
    # Read the data    
    df = pd.read_csv(df) if isinstance(df, str) else df.copy()
    
    N = df.shape[0] # number of observations

    # Generador aleatorio propio: con la misma semilla, la misma mezcla
    rng = np.random.default_rng(seed)

    # Mezcla aleatoria de los datos
    df = df.iloc[rng.permutation(N)].reset_index(drop=True)

    # Define test_size and samples
    n = int(N * test_size)
    data_train = df.iloc[np.arange(N-n)].reset_index(drop=True)
    data_test =  df.iloc[np.arange(N-n, N)].reset_index(drop=True)

    return data_train, data_test
