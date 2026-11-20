import os
import sys
import json
import csv

from utils import *
from gepkarva import gen, Cromosoma
from dataprocessor import datos_test_train
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

def añadir_fila(ruta, fila, cabecera=None):
    """Añade `fila` (lista de valores) al final del csv. Si el archivo no existe,
    lo crea y escribe antes la cabecera (si se da)."""
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo and cabecera is not None:
            w.writerow(cabecera)
        w.writerow(fila)

def best_pobl(poblacion, train, p=90):
    r"""
    Parametros
    ----------
        - poblacion: lista de Cromosomas
        - X : matriz de datos (variables de los cromosomas)
        - y : target
    Returns
    -------
        - i : Indice del mejor individuo de la población
        - px : El percentil p de los mejores individuos
            Sobre los que podemos sesgar mutaciones y recombinaciones.
        - scores : lista de los scores de cada individuo.
    """
    scores = []
    X, y = train
    for c in poblacion:
        Xc = c.evaluar(X)
        datos = np.c_[X, Xc]
        regresion = LinearRegression().fit(datos, y)
        scores.append(regresion.score(datos, y))

    px = np.percentile(scores, p)
    i = np.argmax(scores)
    return poblacion[i], scores[i], px

def test_cromosoma(c, regresion, test):
    X, y = test
    datos = np.c_[X, c.evaluar(X)]
    return regresion.score(datos, y)


def ejecutar_experimento(setup, name_exp, data, rng):
    r"""
    Cada experimento se ejecuta con los dos datasets. Para un experimento
    debemos 
    """

    p_mut = 0.2 
    p_recomb_s = 0.3; p_recomb_m = 0.1; p_recomb_c = 0.3
    p_trans_r = 0.1

    n_samples = setup['n_samples']; n_attr = setup['n_attr']
    h = setup['h']
    n_pobl = setup['n_pobl']; n_gener = setup['n_gener']

    p_mut = setup.get('p_mut', p_mut) # Prob mutacion
    p_recomb_s = setup.get('p_recomb_s', 
                           p_recomb_s) # Prob  recombinacion simple
    p_recomb_m = setup.get('p_recomb_m',  
                          p_recomb_m) # Prob recombinacion multipunto
    p_recomb_c = setup.get('p_recomb_c', 
                           p_recomb_c) # Prob recombinacion cromosoma
    p_trans_r = setup.get('p_trans_r', 
                          p_trans_r) # Transposicion de raiz

    train0, _ = data[0]
    n_vars = train0[0].shape[1]
    
    c_plantilla = Cromosoma(n_vars=n_vars, h=h, n_gen=n_attr)
    
    mejor_cromosoma = None
    mejor_score = 0

    for sample in range(n_samples):

        resultados = {'sample': sample, 
                      'generacion': 0, 
                      'score_tr_base': 0,
                      'score_te_base': 0,
                      'score_tr': None, 
                      'score_te': None,
                      'cromosoma': None,}
        
        train, test = data[sample]
        regresion_base = LinearRegression().fit(train[0], train[1])
        resultados['score_tr_base'] = regresion_base.score(train[0], train[1])
        resultados['score_te_base'] = regresion_base.score(test[0], test[1])

        poblacion0 = []
        for _ in range(n_pobl):
            c = Cromosoma(n_vars=n_vars, h=h, n_gen=n_attr)
            c.aleatorio(rng)
            poblacion0.append(c)

        c_best, score, px = best_pobl(poblacion0, train)
        if score >= mejor_score:
            mejor_cromosoma = c_best
            mejor_score = score

        for generacion in range(1, n_gener):
            # Aplicamos operaciones genéticas sin competicion y
            # de manera aleatoria y sin elitismo

            # Recombinaciones simples
            recombinations = np.flatnonzero(
                rng.random(len(poblacion0)) < p_recomb_s)
            
            pairs_recomb = rng.permutation(recombinations)
            for j in range(len(recombinations)):
                c1 = poblacion0[recombinations[j]]
                c2 = poblacion0[pairs_recomb[j]]
                recombinacion_simple(c1, c2, rng)

            # Recombinaciones multipunto
            recombinations = np.flatnonzero(
            rng.random(len(poblacion0)) < p_recomb_m)
            pairs_recomb = rng.permutation(recombinations)
            for j in range(len(recombinations)):
                c1 = poblacion0[recombinations[j]]
                c2 = poblacion0[pairs_recomb[j]]
                recombinacion_multipunto(c1, c2, rng)

            # Recombinaciones cromosoma
            recombinations = np.flatnonzero(
            rng.random(len(poblacion0)) < p_recomb_m)
            pairs_recomb = rng.permutation(recombinations)
            for j in range(len(recombinations)):
                c1 = poblacion0[recombinations[j]]
                c2 = poblacion0[pairs_recomb[j]]
                recombinacion_de_cromosomas(c1, c2, rng)

            # Mutaciones
            mutations = np.flatnonzero(
            rng.random(len(poblacion0)) < p_mut)
            for j in mutations:
                g = rng.integers(0,c_plantilla.n_gen)
                mutacion(poblacion0[j].sec_genes[g], p_mut, rng)

            c_gener, score_gener, _ = best_pobl(poblacion0, train)
            if score_gener > mejor_score:
                mejor_cromosoma = c_gener
                mejor_score = score_gener
                resultados['generacion'] = generacion
        
        resultados['score_tr'] = mejor_score
        # guardamos_el_mejor_cromosoma
        name_mejor_cromosoma = name_exp + f'/Cromosomas/C{sample}'
        mejor_cromosoma.guardar_cromosoma(name_mejor_cromosoma)
        resultados['cromosoma'] = name_mejor_cromosoma

        # Evaluamos en test
        Xtr, ytr = train
        Xc = mejor_cromosoma.evaluar(Xtr)
        X = np.c_[Xtr, Xc]
        regresion = LinearRegression().fit(X, ytr)
        score_test = test_cromosoma(mejor_cromosoma, regresion, test)
        resultados['score_te'] = score_test

        añadir_fila(name_exp + '.csv', 
                    list(resultados.values()),
                    list(resultados.keys()))



if __name__ == '__main__':

    # ====== SETUP ==============
    ruta_experimentos = 'experimentos_setups/experimento1/' # Definir carpeta que queremos ejecutar
    EXPERIMENTOS = load_experimentos(ruta_experimentos)

    n_exp = 0
    seed = 123981
    rng = np.random.default_rng(seed)
    dfDiabetes = pd.read_csv("rawdata/Diabetes.csv")
    dfCalifornia = pd.read_csv("rawdata/California.csv")

    proporcion = 0.2
    samples = 5

    for setup in EXPERIMENTOS:

        n_samples = setup['n_samples']
        DIABETES, CALIFORNIA = [], []
        for _ in range(n_samples):
            dfD_train, dfD_test = datos_test_train(dfDiabetes, 
                                                   test_size=proporcion,
                                                   rng = rng)
            dfC_train, dfC_test = datos_test_train(dfCalifornia, 
                                                   test_size=proporcion,
                                                   rng=rng)

            colsD = dfDiabetes.columns
            X_Dtr = dfD_train[colsD[:-1]].to_numpy()
            y_Dtr = dfD_train[colsD[-1]].to_numpy()
            X_Dte = dfD_test[colsD[:-1]].to_numpy()
            y_Dte = dfD_test[colsD[-1]].to_numpy()

            colsC = dfCalifornia.columns
            X_Ctr = dfC_train[colsC[:-1]].to_numpy()
            y_Ctr = dfC_train[colsC[-1]].to_numpy()
            X_Cte = dfC_test[colsC[:-1]].to_numpy()
            y_Cte = dfC_test[colsC[-1]].to_numpy()

            DIABETES.append(((X_Dtr, y_Dtr), (X_Dte, y_Dte)))
            CALIFORNIA.append(((X_Ctr, y_Ctr), (X_Cte, y_Cte)))


        name_exp = setup.get('name', n_exp)
        # Ejecutamos para diabetes
        ejecutar_experimento(
                setup,
                f'experimentos/Diabetes/{name_exp}/{name_exp}',
                DIABETES,
                rng,
            )
        # Ejecutamos para california
        ejecutar_experimento(
                setup,
                f'experimentos/California/{name_exp}/{name_exp}',
                CALIFORNIA,
                rng,
            )
        n_exp += 1