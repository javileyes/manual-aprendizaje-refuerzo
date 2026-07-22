"""
Dyna-Q en un laberinto: planificar "en la cabeza" para gastar menos datos.

Implementa a mano, con solo numpy, el clasico "Dyna maze" de Sutton & Barto
(rejilla de 6x9 con paredes) y entrena tres agentes que comparten EXACTAMENTE
el mismo bucle de Q-learning real, pero difieren en cuantos pasos de
planificacion imaginada n hacen tras cada paso real:

    n = 0   -> Q-learning puro (sin modelo, sin planificacion)
    n = 5   -> Dyna-Q con 5 pasos de planificacion por paso real
    n = 50  -> Dyna-Q con 50 pasos de planificacion por paso real

Cada agente aprende un modelo tabular del entorno (que recompensa y a que
estado lleva cada par estado-accion visitado) y, entre paso y paso real,
repite n transiciones muestreadas de ese modelo para propagar el valor mucho
mas rapido. El resultado: Dyna-Q necesita muchisimos menos EPISODIOS REALES
para resolver el laberinto.

Al terminar imprime, para cada n, los pasos del primer episodio y la media de
pasos de los ultimos episodios, y muestra una grafica con los pasos por
episodio (promediados sobre varias semillas). Se ve que n=50 cae al camino
casi optimo en apenas unos episodios, mientras que n=0 tarda decenas.

Como ejecutarlo:
    pip install -r requirements.txt
    python code/18-model-based/dyna_q.py
"""

import numpy as np
import matplotlib.pyplot as plt


class DynaMaze:
    """Laberinto de Sutton & Barto: rejilla de 6 filas x 9 columnas.

    La salida (S) esta en (2, 0) y la meta (G) en (0, 8). Hay siete paredes
    que bloquean el paso. Chocar contra una pared o contra el borde te deja
    donde estabas. Todas las transiciones dan recompensa 0 salvo llegar a la
    meta, que da +1 y termina el episodio. El camino optimo tiene 14 pasos.

    Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda.
    """

    def __init__(self):
        self.filas = 6
        self.columnas = 9
        self.n_estados = self.filas * self.columnas
        self.n_acciones = 4
        self.inicio = (2, 0)
        self.meta = (0, 8)
        self.paredes = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
        self.pos = self.inicio

    def id_estado(self, fila, col):
        return fila * self.columnas + col

    def reset(self):
        self.pos = self.inicio
        return self.id_estado(*self.pos)

    def step(self, accion):
        """Aplica una accion y devuelve (estado_siguiente, recompensa, fin)."""
        fila, col = self.pos
        nf, nc = fila, col
        if accion == 0:      # arriba
            nf = fila - 1
        elif accion == 1:    # derecha
            nc = col + 1
        elif accion == 2:    # abajo
            nf = fila + 1
        elif accion == 3:    # izquierda
            nc = col - 1

        # Si nos salimos del tablero o chocamos con una pared, no nos movemos.
        fuera = nf < 0 or nf >= self.filas or nc < 0 or nc >= self.columnas
        if fuera or (nf, nc) in self.paredes:
            nf, nc = fila, col

        self.pos = (nf, nc)
        if self.pos == self.meta:
            return self.id_estado(nf, nc), 1.0, True
        return self.id_estado(nf, nc), 0.0, False


def epsilon_greedy(Q, estado, epsilon, rng):
    """Elige una accion epsilon-greedy con desempate aleatorio."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    valores = Q[estado]
    optimas = np.flatnonzero(valores == valores.max())
    return int(rng.choice(optimas))


def dyna_q(env, n_planificacion, n_episodios, alpha=0.1, gamma=0.95,
           epsilon=0.1, max_pasos=5000, semilla=0):
    """Entrena Dyna-Q y devuelve los pasos usados en cada episodio.

    Con n_planificacion=0 esto es exactamente Q-learning tabular. Con n>0,
    tras cada transicion real se aprende el modelo y se hacen n actualizaciones
    de Q sobre transiciones imaginadas (muestreadas de pares ya visitados).
    """
    rng = np.random.default_rng(semilla)
    Q = np.zeros((env.n_estados, env.n_acciones))

    # Modelo aprendido: (estado, accion) -> (recompensa, estado_siguiente).
    modelo = {}
    visitados = []            # lista de pares (s, a) ya observados (sin repetir)

    pasos_por_episodio = np.zeros(n_episodios, dtype=int)

    for ep in range(n_episodios):
        s = env.reset()
        pasos = 0
        for _ in range(max_pasos):
            a = epsilon_greedy(Q, s, epsilon, rng)
            s2, r, fin = env.step(a)

            # --- 1) Aprendizaje directo (Q-learning con la experiencia real) ---
            # Q[meta] se queda en 0 (estado terminal), asi que el bootstrap
            # r + gamma * max Q[s2] vale r al llegar a la meta.
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])

            # --- 2) Aprendizaje del modelo (entorno determinista) ---
            if (s, a) not in modelo:
                visitados.append((s, a))
            modelo[(s, a)] = (r, s2)

            # --- 3) Planificacion: n repasos imaginados del modelo ---
            if n_planificacion > 0 and visitados:
                idxs = rng.integers(0, len(visitados), size=n_planificacion)
                for j in idxs:
                    sp, ap = visitados[j]
                    rp, sp2 = modelo[(sp, ap)]
                    Q[sp, ap] += alpha * (rp + gamma * Q[sp2].max() - Q[sp, ap])

            s = s2
            pasos += 1
            if fin:
                break
        pasos_por_episodio[ep] = pasos

    return pasos_por_episodio


def experimento(env, lista_n, n_episodios=50, n_semillas=15):
    """Promedia los pasos por episodio de cada n sobre varias semillas."""
    maestro = np.random.default_rng(0)
    semillas = maestro.integers(0, 2**31 - 1, size=n_semillas)

    curvas = {}
    for n in lista_n:
        acum = np.zeros(n_episodios)
        for semilla in semillas:
            # Todas las configuraciones comparten las mismas semillas: la
            # comparacion es justa (misma "suerte", distinto n).
            acum += dyna_q(env, n_planificacion=n, n_episodios=n_episodios,
                           semilla=int(semilla))
        curvas[n] = acum / n_semillas
    return curvas


def dibujar(curvas, optimo=14):
    plt.figure(figsize=(9, 5.2))
    colores = {0: "#dc2626", 5: "#0ea5e9", 50: "#4f46e5"}
    episodios = np.arange(1, len(next(iter(curvas.values()))) + 1)

    for n, curva in curvas.items():
        color = colores.get(n, None)
        etiqueta = "Q-learning (n = 0)" if n == 0 else f"Dyna-Q (n = {n})"
        # El primer episodio es enorme para todos (exploracion a ciegas);
        # se ve mejor la diferencia a partir del episodio 2.
        plt.plot(episodios[1:], curva[1:], color=color, lw=2, label=etiqueta)

    plt.axhline(optimo, color="#059669", ls="--", lw=1.2,
                label=f"optimo ({optimo} pasos)")
    plt.xlabel("Episodio")
    plt.ylabel("Pasos por episodio (promedio)")
    plt.title("Dyna-Q: mas planificacion, menos episodios reales")
    plt.ylim(0, 800)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = DynaMaze()
    lista_n = [0, 5, 50]
    curvas = experimento(env, lista_n, n_episodios=50, n_semillas=15)

    print("Pasos por episodio (promedio sobre 15 semillas):")
    print(f"{'n':>4} | {'episodio 1':>11} | {'episodio 2':>11} | "
          f"{'ultimos 10 (media)':>18}")
    print("-" * 54)
    for n in lista_n:
        c = curvas[n]
        print(f"{n:>4} | {c[0]:>11.0f} | {c[1]:>11.0f} | {c[-10:].mean():>18.1f}")

    dibujar(curvas)
