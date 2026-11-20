import json
import os
import numpy as np

from gepkarva import gen, Cromosoma


def load_from_json(name: str):
    """Lee genetic_data/<name>.json (creado con Cromosoma.save_as_json) y devuelve el Cromosoma."""
    from gepkarva import gen  # import local
    ruta = f"{name}.json"
    with open(ruta, "r", encoding="utf-8") as f:
        d = json.load(f)
    # FIX: save_as_json guarda un `gen` (no un Cromosoma), así que se reconstruye un `gen`
    c = gen(d["n_vars"], d["h"], funciones=d["funciones"], constantes=d["constantes"])
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
    if f == "Pow":
        # TODO
        pass
    if f=="H": # Huevera
        return np.sin(a)*np.cos(args[1])
    if f=="P": # Paraboloide
        return a**2 + args[1]**2
    if f=="Max": # Rosenbluck
        x = np.asarray(a).ravel()   # (n,) o (n,1) -> (n,)
        y = np.asarray(args[1]).ravel()
        return np.maximum(x, y)
    if f=="Min":
        x = np.asarray(a).ravel()   # (n,) o (n,1) -> (n,)
        y = np.asarray(args[1]).ravel()
        return np.minimum(x, y)
    if f=="Gauss":
        return np.exp(-(a**2+args[1]**2))
    if f=="R":
        d = a**2 + args[1]**2
        return np.divide(a, d, out=np.zeros_like(a), where=(d != 0))
    
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
    # TODO -- Sesgar a funciones no lineales.
    w = np.ones(gen.n_simb)
    w[2:gen.n_f] = gen.peso_nl
    pw = w / w.sum()
    k = gen.karva
    for i in np.flatnonzero(rng.random(gen.largo_gen) < p):

        if i < gen.h:
            # Higher probability to mutate as a function
            new = rng.choice(np.arange(gen.n_simb), p=pw, size=1)[0]
        else:
            new = rng.integers(gen.n_f, gen.n_simb)
        
        k[i] = new

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
    sec = karva2[inicio_sec:inicio_sec+long]  # FIX: slice, no indexado 2D
    posicion_trasposicion = rng.integers(1, g1.h)  # FIX: no existe `plantilla`; pos >= 1 (no toca la raíz)

    # FIX: `+` sobre arrays suma elementos; hay que concatenar
    new_head = np.concatenate([g1.karva[:posicion_trasposicion], sec, g1.karva[posicion_trasposicion:]])
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
    # FIX: se guardan las POSICIONES de funciones de g2 (antes se guardaba el símbolo, y `plantilla` no existe)
    for pos, f in enumerate(karva2):
        if f < g2.n_f:
            # es una funcion
            pos_f.append(pos)

    if len(pos_f) == 0:
        return g1
    
    inicio_sec = rng.choice(pos_f)
    sec = karva2[inicio_sec:inicio_sec+long]  # FIX: slice

    new_head = np.concatenate([sec, g1.karva[:g1.h]])  # FIX: concatenar; la secuencia va en la raíz
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
    new_sec = sec1[posicion_recombinacion:].copy()  # FIX: .copy(); si no es una vista y se sobrescribe
    sec1[posicion_recombinacion:]= sec2[posicion_recombinacion:]
    sec2[posicion_recombinacion:]= new_sec
    c1.update(sec1)
    c2.update(sec2)


def recombinacion_multipunto(c1, c2, rng):  # FIX: sobraba `self`
    r"""
    Hace una recombinacion de dos cromosomas entre dos posiciones
    """
    sec1 = c1.toarray()
    sec2 = c2.toarray()
    posicion_recombinacion1 = rng.integers(1,len(sec1)-2)
    posicion_recombinacion2 = rng.integers(posicion_recombinacion1,
                                           len(sec1)-1)

    new_sec = sec1[posicion_recombinacion1:posicion_recombinacion2].copy()  # FIX: .copy() (vista)
    sec1[posicion_recombinacion1:posicion_recombinacion2]= sec2[posicion_recombinacion1:posicion_recombinacion2]
    sec2[posicion_recombinacion1:posicion_recombinacion2]= new_sec
    c1.update(sec1)
    c2.update(sec2)

def recombinacion_de_cromosomas(c1, c2, rng):  # FIX: sobraba `self`
    i = rng.integers(0, c1.n_gen)
    cromosoma_intercambiado = c1.sec_genes[i]
    c1.sec_genes[i] = c2.sec_genes[i]
    c2.sec_genes[i] = cromosoma_intercambiado


# ======================================================================
# FUNCIONES PARA EJECUTAR EXPERIMENTOS
# ======================================================================

def guardar_experimento(dd, path):
    r"""
    Esta función recibe un diccionario dd con el setup de un experimento,
    y guarda el setup en la carpeta path, como <dd['name']>.json (name es
    obligatorio). Además, se asegura que todas las funciones 
    necesarias para el experimento estén guardadas.

    Parametros necesarios en un SetUP
    1. n_samples : número de pruebas remezclando los datos
    2. n_attr : número de genes de cada individuo (número de atributos que crearemos)
    3. h : tamño de la cabeza de los genes
    4. n_pobl : tamaño de la población inicial
    5. n_gener : número de generaciones para evolucionar

    Parametros opcionales
    1. p_mut
    2. p_recomb_s
    3. p_recomb_m
    4. p_recomb_c
    5. p_trans_r
    """
    vars_obligatorias = ['n_samples', 
                         'n_attr',
                         'h',
                         'n_pobl',
                         'n_gener'
                         ]
    for v in vars_obligatorias:
        assert v in dd, f"Hay que definir {v}"

    assert 'name' in dd, "Hay que definir name"

    # path es una carpeta; el archivo se llama <name>.json
    os.makedirs(path, exist_ok=True)
    ruta = os.path.join(path, f"{dd['name']}.json")
    with open(ruta, "w", encoding="utf-8") as f:
            json.dump(dd, f, indent=2)

def cargar_experimento(ruta):
    r"""
    Esta función lee un experimento, guardado en formato .json y lo devuelve
    en formato diccionario de python.
    """
    with open(ruta, "r", encoding="utf-8") as f:
            dd = json.load(f)
    return dd

def load_experimentos(path):
    r"""
    Esta función lee de una carpeta todos los experimentos guardados que
    queremos ejecutar y devuelve una lista donde en cada posición tenemos
    un diccionario con el setup del  experimento.

    Para mayor información en como deben estar guardados los experimenots, 
    lea la documentación de as funciones `guardar_experimento` y 
    `cargar_experimento`. De todas formas sólo recomendamos cargar los 
    experimentos con datos ques se hayan guardado con las funciones ya que 
    nos aseguran el correcto funcionamiento.
    """
    experimentos = []
    for nombre in sorted(os.listdir(path)):
        if nombre.endswith(".json"):
            experimentos.append(cargar_experimento(os.path.join(path, nombre)))
    return experimentos
