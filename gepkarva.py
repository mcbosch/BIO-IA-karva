import json
import os
import numpy as np
import networkx as nx

from utils import *

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genetic_data")

HPFUNCION = {
            "+": 2,     # 0 
            "-": 2,     # 1
            "*": 2,     # 2
            "/": 2,     # 3
            "Q": 1,     # 4
            "L": 1,     # 5
            "ReLU": 1,  # 6
            }
FUNCIONES_DEFECTO = tuple(HPFUNCION)
CONSTANTES_DEFECTO = (1.0, 2.0, 0.5, np.pi)

class gen:

    def __init__(self, 
                 n_vars, 
                 h,
                 funciones = FUNCIONES_DEFECTO,
                 constantes = CONSTANTES_DEFECTO,
                 ):
        r"""
        Esta funcion nos define un objeto Python que codificará un gen
        (un atributo) en lenguaje karva. Notemos que para definir exactamente un
        gen deberemos utilizar ka función `aleatorio`; esto solo nos define 
        la estructura. Por lo tanto lo podemos entender como una estructura de gen
        """


        self.n_vars, self.h = n_vars, h #
        self.funciones = tuple(funciones)
        self.constantes = tuple(constantes)
        self.aridades = np.array([HPFUNCION[f] for f in self.funciones])
        self.n_f = len(self.funciones) # Numero de funciones
        self.n_term = n_vars + len(self.constantes) # Nodos terminales
        self.n_simb = self.n_f + self.n_term  # Numero de simbolos totales
        self.nombres = [f'X{i+1}' for i in range(self.n_vars)] # Nombres de las variables

        # cola: garantiza que todo gen se pueda leer completo
        self.t = self.h * (int(self.aridades.max()) - 1) + 1
        self.largo_gen = self.h + self.t

        # correspondencia entero -> funcion / variable / constante
        self.correspondencia = list(self.funciones) + self.nombres + list(self.constantes)

        # secuencia karva
        self.karva = []

    def aleatorio(self, rng):
        """gen aleatorio válido, shape (n_genes*largo_gen,): cabeza con cualquier
        símbolo, cola sólo con terminales."""
        # Definimos un gen aleatorio con nodos terminales
        c = rng.integers(self.n_f, self.n_simb, size = self.largo_gen)
        # Añadimos posibles funciones a la cabeza de cada gen.
        c[np.arange(self.h)] = rng.integers(0, self.n_simb, size=self.h)
        self.karva = c
        return c
    
    def adj(self):
        """
        Lee el gen en lenguaje karva y devuelve un diccionario de adyacencia que podemos
        entender como grafo. Este posteriormente se podra literalmente expresar y graficar cómo 
        un grafo.

        Ejemplo, gen = [+, Q, x, a, b]:  '+' -> hijos [1, 2];  'Q' -> hijos [3]  => +(Q(a), x)
        Ejemplo return: {0: [1, 2], 1: [3]}
        
        El tamaño del head y del tail nos asegura que no tendremoos problemas de indices.
        """
        hijos, sig, i = {}, 1, 0
        if len(self.karva) == 0:
            return {}
        
        while i < sig:
            if self.karva[i] < self.n_f:
                # Estamos en un nodo función
                # Miramos el número de hijos esperados:
                k = int(self.aridades[self.karva[i]])
                # añadimos los hijos
                hijos[i] = list(range(sig, sig + k))
                # actualizamos nodos leidos
                sig += k
            elif i == 0:
                return {0:[]}
            i += 1
        return hijos

    def evaluar(self, Z):
        """
        Evalúa un gen sobre Z (N, n_vars). Devuelve un vector (N,).
        Hacemos un clip en 1e6, -1e6 para evitar problemas de overflow.
        """
        hijos = self.adj()
        n_f, n_vars = self.n_f, self.n_vars

        def ev(i):
            s = self.karva[i]
            if i not in hijos: # terminal
                if s < n_f + n_vars:
                    return Z[:, s - n_f] #   variable
                return np.full(len(Z), self.constantes[s - n_f - n_vars])   # constante
            # Es una funcion --> se aplica a sus hijos
            return aplicar(self.funciones[s], [ev(j) for j in hijos[i]])

        with np.errstate(all="ignore"):
            out = ev(0)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


    def es_lineal(self):
        r"""
        Revisa que el gen con no sea lineal.
        """
        for f in self.karva:
            if 2<= f < self.n_f:
                # Es una funcion no lineal:
                return False
        return True

    def grafo(self):
        r"""
        Esta función nos devuelve un grafo en networkx que nos servirá para 
        graficar las operaciones y entender mejor que ha aprendido.
        """
        adj = self.adj()
        G = nx.from_dict_of_lists(adj, create_using=nx.DiGraph)
        for i in G.nodes:
            s = self.correspondencia[self.karva[i]]
            G.nodes[i]["simbolo"] = f"{s:.4g}" if isinstance(s, float) else str(s)

        return G

    @staticmethod
    def _posiciones_arbol(G, raiz=0):
        """Layout de árbol: y = -profundidad, x = hojas consecutivas, padre centrado."""
        pos, sig_x = {}, [0]

        def colocar(n, prof):
            hs = list(G.successors(n))
            if not hs:                                 # hoja: siguiente columna libre
                x = sig_x[0]; sig_x[0] += 1
            else:
                x = np.mean([colocar(h, prof + 1) for h in hs])
            pos[n] = (x, -prof)
            return x

        colocar(raiz, 0)
        return pos

    def plot_gen(self, ax=None):
        G = self.grafo()
        if len(G) == 0:
            raise "Your gen has no length"
        
        pos = self._posiciones_arbol(G)
        etiquetas = nx.get_node_attributes(G, "simbolo")
        es_func = [self.karva[n] < self.n_f for n in G.nodes]
        colores = ["lightcoral" if f else "lightblue" for f in es_func]
        nx.draw(G, pos, ax=ax, labels=etiquetas, node_color=colores,
                node_size=900, arrows=False, font_size=10)

    def save_as_json(self, name:str):
        """Guarda el gen en genetic_data/<name>.json (estructura + secuencia karva)."""
        os.makedirs(DATA_DIR, exist_ok=True)
        datos = {
            "n_vars": self.n_vars,
            "h": self.h,
            "funciones": list(self.funciones),
            "constantes": list(self.constantes),
            "karva": [int(s) for s in self.karva],
        }
        ruta = os.path.join(DATA_DIR, f"{name}.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2)
        return ruta

    def copy(self):
        c = gen(self.n_vars, self.h, self.funciones, self.constantes)
        c.karva = self.karva.copy()
        return c


class Cromosoma:
    r"""
    El cromosoma es una secuencia de genes. Los genes podrían ser de distintos
    tamaños. Pero para simplificar por ahora lo mantendremos en la mismas características.

    Cada cromosoma nos define un individuo.
    """
    def __init__(self, 
                 n_vars,
                 h,
                 n_gen = 2,
                 funciones=FUNCIONES_DEFECTO,
                 constantes=CONSTANTES_DEFECTO,
                 ):
        self.sec_genes = [
            gen(n_vars, h, funciones, constantes) for _ in range(n_gen)
        ]
        self.funciones = tuple(funciones)
        self.constantes = tuple(constantes)
        self.aridades = np.array([HPFUNCION[f] for f in self.funciones])
        self.n_f = len(self.funciones) # Numero de funciones
        self.n_term = n_vars + len(self.constantes) # Nodos terminales
        self.n_simb = self.n_f + self.n_term  # Numero de simbolos totales
        self.nombres = [f'X{i+1}' for i in range(self.n_vars)] # Nombres de las variables

        # cola: garantiza que todo gen se pueda leer completo
        self.t = self.h * (int(self.aridades.max()) - 1) + 1
        self.largo_gen = self.h + self.t

        # correspondencia entero -> funcion / variable / constante
        self.correspondencia = list(self.funciones) + self.nombres + list(self.constantes)

    def aleatorio(self, rng):
        for g in self.sec_genes:
            _ = g.aleatorio(rng)
        
    def evaluar(self, Z):
        return np.array([g.evaluar(Z) for g in self.sec_genes])