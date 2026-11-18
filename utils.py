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
def mutacion(self, gen, p, rng):
    r"""
    Mutación puntual. Cada posición de cada gen muta con probabilidad `p` (p=0: nada muta,
    p=1: todas las posiciones se vuelven a sortear).

    Se protege la estructura cabeza-cola: en la cabeza el nuevo símbolo puede ser cualquiera
    (función, variable o constante); en la cola sólo puede ser un terminal.

    Devuelve un individuo nuevo.
    """

    k = np.array(gen.karva, dtype=int)
    for i in np.flatnonzero(rng.random(self.largo_gen) < p):
        k[i] = rng.integers(0 if i < self.h else self.n_f, self.n_simb)

# ------------------------------------------------------------------ transposiciones
def transposicion_de_elementos(g1, g2, rng, long=3):
    r"""
    Transposición de elementos IS. Se elige al azar una secuencia de entre 1 y `max_len`
    símbolos de un gen (de cabeza o cola) y se inserta en la cabeza de otro gen del mismo
    individuo (puede ser el mismo), en una posición >= 1 para no tocar la raíz.

    Parametros
    ----------
        - g1: Cromosoma 1
        - g2: Cromosoma 2
        - rng: np.random.default_rng

    NOTA: El cromosoma 1 y 2 pueden ser el mismo.
    """
    assert isinstance(g1, gen)
    assert isinstance(g2, gen)
    nuevo = g1.copy()
    if g1.h < long:
        return nuevo

    if g2 == None:
        g2 = g1

    karva2 = g2.karva

    # Tomamos una secuencia de la cola de karva 2
    inicio_sec = rng.integers(0, g2.largo_gen - long + 1)
    sec = karva2[inicio_sec, inicio_sec+long]
    posicion_trasposicion = rng.integers(0, g2.plantilla.h - long + 1)

    new_head = nuevo.karva[:posicion_trasposicion] + sec + nuevo.karva[posicion_trasposicion:]
    nuevo.karva[:g1.h] = new_head[:g1.h]
    return nuevo

def transposicion_de_raiz(g1, g2, rng, long=3):
    r"""
    Consiste en seleccionar una secuencia que empiece con unna función y 
    colocarla en la raíz del cromosoma.
    """
    nuevo = g1.copy()
    if g1.h < long:
        return nuevo
    if g2 == None:
        g2 = g1

    karva2 = g2.karva

    # Tomamos una secuencia de las posiciones que son funciones
    pos_f = []
    for f in nuevo.karva:
        if f < g2.plantilla.n_f:
            # es una funcion
            pos_f.append(f)

    if len(pos_f) == 0:
        return nuevo
    
    inicio_sec = rng.choice(pos_f)
    sec = karva2[inicio_sec, inicio_sec+long]

    new_head = sec + nuevo.karva[inicio_sec:]
    nuevo.karva[:g1.h] = new_head[:g1.h]
    return nuevo

# --------------------------------------------------------------- Recombinaciones
def recombinacion_simple(c1, c2, rng):
    r"""
    Hace una recombinacion a partir de una posición 
    devolviendo dos nuevos cromosomas.
    """
    # nuevo_c1 = c1.copy()
    # nuevo_c2 = c2.copy()
    # posicion_recombinacion = rng.integers(0,self.plantilla.largo_gen)
    # locus_1 = nuevo_c1.karva[posicion_recombinacion:]
    # locus_2 = nuevo_c2.karva[posicion_recombinacion:]
    # nuevo_c1.karva[posicion_recombinacion:] = locus_2
    # nuevo_c2.karva[posicion_recombinacion:] = locus_1
    # return nuevo_c1, nuevo_c2

def recombinacion_multipunto(self, c1, c2, rng):
    r"""
    Hace una recombinacion de dos cromosomas entre dos posiciones
    """
    # nuevo_c1 = c1.copy()
    # nuevo_c2 = c2.copy()
    # p1, p2 = np.sort(list(rng.integers(0,self.plantilla.largo_gen,2)))
    # locus_1 = nuevo_c1.karva[p1:p2]
    # locus_2 = nuevo_c2.karva[p1:p2]
    # nuevo_c1.karva[p1:p2] = locus_2
    # nuevo_c2.karva[p1:p2] = locus_1
    # return nuevo_c1, nuevo_c2

def recombinacion_de_cromosomas(self):
    pass
