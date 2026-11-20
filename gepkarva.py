import json
import os
import numpy as np
import networkx as nx

# FIX: `from utils import *` aquí creaba un import circular (utils importa gepkarva), y
# `aplicar` no llegaba a definirse. Ahora se importa dentro de `gen.evaluar`.

HPFUNCION = {
            "+": 2,       
            "-": 2,      
            "*": 2,      
            "/": 2,      
            "Q": 1,      
            # "L": 1,      
            "ReLU": 1,   
            "H": 2,         
            "P": 2,         
            "Max": 2,       
            "Min": 2,
            "Gauss": 2,
            "R": 2,       
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
        self.peso_nl = 2
        # cola: garantiza que todo gen se pueda leer completo
        self.t = self.h * (int(self.aridades.max()) - 1) + 1
        self.largo_gen = self.h + self.t

        # correspondencia entero -> funcion / variable / constante
        self.correspondencia = list(self.funciones) + self.nombres + list(self.constantes)

        # secuencia karva
        self.karva = []

    def aleatorio(self, rng):
        r"""
        Gen aleatorio válido, shape (n_genes*largo_gen,): cabeza con cualquier
        símbolo, cola sólo con terminales.
        Parametros
        ----------
            - rng : numpy.random._generator.Generator
                Puedes definir esto cómo `numpy.random.default_rng(seed)`
                Esto nos permitirá repetir experimentos.
        
        NOTA
        ----
        Cómo nuestros genes van a ser utilizados para una regresión LINEAL vamos a 
        sesgar la aleatoriedad para que nuestros genes no sean lineales, ya que no nos
        aportarian ninguna información.
        """
        # Definimos un gen aleatorio con nodos terminales
        c = rng.integers(self.n_f, self.n_simb, size = self.largo_gen)
        # Añadimos posibles funciones a la cabeza de cada gen.
        # Definir una proporcion de funciones no lineale
        lineal = True
        while lineal:
            w = np.ones(self.n_simb)
            w[2:self.n_f] = self.peso_nl
            p = w / w.sum()
            c[np.arange(self.h)] = rng.choice(np.arange(self.n_simb), p=p, size=self.h)
            self.karva = c
            lineal = self.es_lineal()
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

        Parametros
        ----------
            Z : ndarray (N, n_vars)
                cada variable es una
        """ 
        from utils import aplicar  # FIX: import local para evitar el import circular
        hijos = self.adj()
        n_f, n_vars = self.n_f, self.n_vars

        def ev(i):
            s = self.karva[i]
            if s >= n_f: # terminal (se decide por el símbolo; `hijos` incluye {0: []} si la raíz es terminal)
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
        Revisa que el gen con no sea lineal. Devuelve True en caso de que
        lo sea.
        """
        if self.karva[0] >= self.n_f:
            return True
        
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
            raise ValueError("Your gen has no length")  # FIX: no se puede hacer raise de un str
        
        pos = self._posiciones_arbol(G)
        etiquetas = nx.get_node_attributes(G, "simbolo")
        es_func = [self.karva[n] < self.n_f for n in G.nodes]
        colores = ["lightcoral" if f else "lightblue" for f in es_func]
        nx.draw(G, pos, ax=ax, labels=etiquetas, node_color=colores,
                node_size=900, arrows=False, font_size=10)

    def save_as_json(self, name:str):
        """Guarda el gen en <name>.json (estructura + secuencia karva)."""
        datos = {
            "n_vars": self.n_vars,
            "h": self.h,
            "funciones": list(self.funciones),
            "constantes": list(self.constantes),
            "karva": [int(s) for s in self.karva],
        }
        ruta = f"{name}.json"
        os.makedirs(os.path.dirname(ruta), exist_ok=True)  # incluye subcarpetas dentro de name
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2)
        return ruta

    @classmethod
    def load_from_json(cls, name:str):
        """Lee un gen desde genetic_data/<name>.json (inverso de `save_as_json`)."""
        ruta = f"{name}.json"
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)
        g = cls(datos["n_vars"], datos["h"], datos["funciones"], datos["constantes"])
        g.karva = np.array(datos["karva"], dtype=int)
        return g

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
        self.n_vars, self.h = n_vars, h
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
        self.n_gen = n_gen
        # cola: garantiza que todo gen se pueda leer completo
        self.t = self.h * (int(self.aridades.max()) - 1) + 1
        self.largo_gen = self.h + self.t

        # correspondencia entero -> funcion / variable / constante
        self.correspondencia = list(self.funciones) + self.nombres + list(self.constantes)

    def aleatorio(self, rng):
        for g in self.sec_genes:
            _ = g.aleatorio(rng)
        
    def evaluar(self, Z):
        return np.column_stack([g.evaluar(Z) for g in self.sec_genes])

    def toarray(self):
        return np.concatenate([g.karva for g in self.sec_genes])  # FIX: concatenar los karva, no los objetos gen

    def update(self, sec):
        r"""
        Parametros
        ----------
            - Una secuencia
        """
        for i in range(self.n_gen):
            start = i * self.largo_gen
            self.sec_genes[i].karva = sec[start : start + self.largo_gen]

    def plot_cromosoma(self, num=None):
        r"""
        Plotea los `num` primeros genes en una figura en una tabla de cómo máximo 4 columnas.
        Es decir, si dibujamoos 5, dibujará 4 en una fila y 1 abajo. Y si dibujamos 8, en dos filas.
        """
        import matplotlib.pyplot as plt

        n = self.n_gen if num is None else min(num, self.n_gen)
        if n <= 0:
            raise ValueError("`num` debe ser al menos 1")

        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            if i < n:
                self.sec_genes[i].plot_gen(ax=ax)
                ax.set_title(f"Gen {i + 1}")
            else:
                ax.axis("off")  # huecos sobrantes de la última fila

        fig.tight_layout()
        return fig, axes[:n]

    def guardar_cromosoma(self, path):
        r"""
        Guarda el cromosmoa en una carpeta definida en path
        donde se guardan todos los archivos json de cada gen del 
        cromosoma.

        Parametros
        ----------
            - path : str
                Debe tener la forma '<destination>/<folder>' o '<folder>'
                si queremos que la carpeta se cree en el mismo nivel
        """

        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(self.n_gen):
            name = path + f'/gen{i}'
            self.sec_genes[i].save_as_json(name)

        return None

    @classmethod
    def cargar_cromosoma(cls, path):
        r"""
        Lee un cromosoma guardado con `guardar_cromosoma` y devuelve un objeto
        Cromosoma. Se leen los archivos gen0.json, gen1.json, ... de la carpeta.

        Parametros
        ----------
            - path : str
                Misma ruta que se usó en `guardar_cromosoma`.
        """
        carpeta = path
        if not os.path.isdir(carpeta):
            raise FileNotFoundError(f"No existe la carpeta {carpeta}")

        # Genes consecutivos gen0, gen1, ... hasta que falte uno
        genes, i = [], 0
        while os.path.exists(os.path.join(carpeta, f"gen{i}.json")):
            genes.append(gen.load_from_json(path + f'/gen{i}'))
            i += 1
        if not genes:
            raise FileNotFoundError(f"No hay archivos gen*.json en {carpeta}")

        g0 = genes[0]
        crom = cls(g0.n_vars, g0.h, len(genes), g0.funciones, g0.constantes)
        crom.sec_genes = genes
        return crom

    

