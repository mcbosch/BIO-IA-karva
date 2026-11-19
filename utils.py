import json
import os
import numpy as np

from gepkarva import gen, Cromosoma


def load_from_json(name: str):
    """Lee genetic_data/<name>.json (creado con Cromosoma.save_as_json) y devuelve el Cromosoma."""
    from gepkarva import Cromosoma, DATA_DIR  # import local: gepkarva ya importa utils
    ruta = os.path.join(DATA_DIR, f"{name}.json")
    with open(ruta, "r", encoding="utf-8") as f:
        d = json.load(f)
    c = Cromosoma(d["n_vars"], d["h"], funciones=d["funciones"], constantes=d["constantes"])
    c.karva = np.array(d["karva"], dtype=int)
    return c


def aplicar(f, args):
    """Aplica la función `f` (símbolo) a sus argumentos ya evaluados (vectores de N valores)."""
    a = args[0]
    if f == "+":
        return a + args[1]
    if f == "-":
        return a - args[1]
    if f == "*":
        return a * args[1]
    if f == "/":
        b = args[1]
        ok = np.abs(b) > 0.1
        return np.where(ok, a / np.where(ok, b, 1.0), 1.0)
    if f == "Q":
        return np.sqrt(np.abs(a))
    if f == "L":
        ok = np.abs(a) > 1e-6
        return np.where(ok, np.log(np.where(ok, np.abs(a), 1.0)), 0.0)
    if f == "ReLU":
        return np.maximum(0.0, a)

    raise ValueError(f"Función sin implementar: {f}")

#============================================================================
# Operadores genéticos
#============================================================================
r"""
    Aqui definiremos as operaciones genéticas necesarias para trabajar con nuestros cromosomas.

    - mutacion
    - transposicion_de_elementos
    - transposicion_de_cromosomas
    - recombinacion_simple
    - recombinacion_multi_punto
    - recombinacion_de_cromosomas

    Se asumen cromosomas generales, es decir, pueden tener distintos tamaños pero todos
    son de la clase Cromosoma
"""
# ------------------------------------------------------------------ mutación
def mutacion(gen, p, rng):
    r"""
    Mutación puntual. Cada posición de cada gen muta con probabilidad `p` (p=0: nada muta,
    p=1: todas las posiciones se vuelven a sortear).

    Se protege la estructura cabeza-cola: en la cabeza el nuevo símbolo puede ser cualquiera
    (función, variable o constante); en la cola sólo puede ser un terminal.

    Devuelve un individuo nuevo.
    """

    k = gen.karva
    for i in np.flatnonzero(rng.random(gen.largo_gen) < p):
        k[i] = rng.integers(0 if i < gen.h else gen.n_f, gen.n_simb)

# ------------------------------------------------------------------ transposiciones
def transposicion_de_elementos(g1, g2, rng, long=3):
    r"""
    Transposición de elementos IS. Se elige al azar una secuencia de entre 1 y `max_len`
    símbolos de un gen (de cabeza o cola) y se inserta en la cabeza de otro gen del mismo
    individuo (puede ser el mismo), en una posición >= 1 para no tocar la raíz.

    Parametros
    ----------
        - g1: gen del cromosoma 
        - g2: gen del cromosoma
        - rng: np.random.default_rng

    El gen g1 sufre una modificación
    NOTA: El gen 1 y 2 pueden ser el mismo. 
    """
    assert isinstance(g1, gen)
    assert isinstance(g2, gen)
    if g1.h < long:
        return g1

    if g2 == None:
        g2 = g1

    karva2 = g2.karva

    # Tomamos una secuencia de la cola de karva 2
    inicio_sec = rng.integers(0, g2.largo_gen - long + 1)
    sec = karva2[inicio_sec, inicio_sec+long]
    posicion_trasposicion = rng.integers(0, g2.plantilla.h - long + 1)

    new_head = g1.karva[:posicion_trasposicion] + sec + g1.karva[posicion_trasposicion:]
    g1.karva[:g1.h] = new_head[:g1.h]
    return g1

def transposicion_de_raiz(g1, g2, rng, long=3):
    r"""
    Consiste en seleccionar una secuencia que empiece con unna función y 
    colocarla en la raíz del cromosoma.

    Parametros
    ----------
        - g1: gen de un cromosoma
        - g2: gen de un cromosoma

    El gen g1 sufre una modificación
    """
    if g1.h < long:
        return g1
    if g2 == None:
        g2 = g1

    karva2 = g2.karva

    # Tomamos una secuencia de las posiciones que son funciones
    pos_f = []
    for f in g1.karva:
        if f < g2.plantilla.n_f:
            # es una funcion
            pos_f.append(f)

    if len(pos_f) == 0:
        return g1
    
    inicio_sec = rng.choice(pos_f)
    sec = karva2[inicio_sec, inicio_sec+long]

    new_head = sec + g1.karva[inicio_sec:]
    g1.karva[:g1.h] = new_head[:g1.h]
    return g1

# --------------------------------------------------------------- Recombinaciones
def recombinacion_simple(c1, c2, rng):
    r"""
    Hace una recombinacion a partir de una posición 
    devolviendo dos nuevos cromosomas.

    Parametros
    ----------

    """
    sec1 = c1.toarray()
    sec2 = c2.toarray()
    posicion_recombinacion = rng.integers(1,len(sec1)-1)
    new_sec = sec1[posicion_recombinacion:]
    sec1[posicion_recombinacion:]= sec2[posicion_recombinacion:]
    sec2[posicion_recombinacion:]= new_sec
    c1.update(sec1)
    c2.update(sec2)


def recombinacion_multipunto(self, c1, c2, rng):
    r"""
    Hace una recombinacion de dos cromosomas entre dos posiciones
    """
    sec1 = c1.toarray()
    sec2 = c2.toarray()
    posicion_recombinacion1 = rng.integers(1,len(sec1)-2)
    posicion_recombinacion2 = rng.integers(posicion_recombinacion1,
                                           len(sec1)-1)

    new_sec = sec1[posicion_recombinacion1:posicion_recombinacion2]
    sec1[posicion_recombinacion1:posicion_recombinacion2]= sec2[posicion_recombinacion1:posicion_recombinacion2]
    sec2[posicion_recombinacion1:posicion_recombinacion2]= new_sec
    c1.update(sec1)
    c2.update(sec2)

def recombinacion_de_cromosomas(self, c1, c2, rng):
    i = rng.integers(0, c1.n_gen)
    cromosoma_intercambiado = c1.sec_genes[i]
    c1.sec_genes[i] = c2.sec_genes[i]
    c2.sec_genes[i] = cromosoma_intercambiado

    
