"""
SARSA on-policy sobre el entorno Cliff Walking (el acantilado).

Implementa a mano, con solo numpy, el clasico entorno "Cliff Walking" de
Sutton & Barto y entrena un agente con SARSA (control por diferencias
temporales on-policy). La actualizacion usa la quintupla (S, A, R, S', A'),
de donde viene el nombre del algoritmo.

Al terminar imprime la recompensa media de los ultimos episodios y la
politica aprendida, y muestra dos graficas: la recompensa por episodio
(suavizada con una media movil) y la politica greedy dibujada como flechas
sobre la rejilla. Se ve que SARSA aprende un camino "prudente" que rodea el
acantilado, alejandose del borde para no caerse durante la exploracion.

Como ejecutarlo:
    pip install -r requirements.txt
    python code/08-sarsa/sarsa_cliff.py
"""

import numpy as np
import matplotlib.pyplot as plt


class CliffWalking:
    """Entorno del acantilado: rejilla de 4 filas x 12 columnas.

    La salida esta abajo a la izquierda y la meta abajo a la derecha. Toda la
    fila inferior entre ambas es un acantilado: caer en el da -100 y te
    devuelve a la salida. Cada paso normal cuesta -1. El episodio termina
    solo al alcanzar la meta.

    Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda.
    """

    def __init__(self):
        self.filas = 4
        self.columnas = 12
        self.n_estados = self.filas * self.columnas
        self.n_acciones = 4
        self.inicio = (3, 0)
        self.meta = (3, 11)
        self.pos = self.inicio

    def es_acantilado(self, fila, col):
        return fila == 3 and 1 <= col <= 10

    def id_estado(self, fila, col):
        return fila * self.columnas + col

    def reset(self):
        self.pos = self.inicio
        return self.id_estado(*self.pos)

    def step(self, accion):
        """Aplica una accion y devuelve (estado_siguiente, recompensa, fin)."""
        fila, col = self.pos
        if accion == 0:      # arriba
            fila = max(fila - 1, 0)
        elif accion == 1:    # derecha
            col = min(col + 1, self.columnas - 1)
        elif accion == 2:    # abajo
            fila = min(fila + 1, self.filas - 1)
        elif accion == 3:    # izquierda
            col = max(col - 1, 0)

        if self.es_acantilado(fila, col):
            self.pos = self.inicio
            return self.id_estado(*self.pos), -100.0, False
        self.pos = (fila, col)
        if self.pos == self.meta:
            return self.id_estado(*self.pos), -1.0, True
        return self.id_estado(*self.pos), -1.0, False


def epsilon_greedy(Q, estado, epsilon, rng):
    """Elige una accion epsilon-greedy con desempate aleatorio."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    valores = Q[estado]
    optimas = np.flatnonzero(valores == valores.max())
    return int(rng.choice(optimas))


def entrenar_sarsa(env, n_episodios=500, alpha=0.5, gamma=1.0,
                   epsilon=0.1, max_pasos=1000, semilla=0):
    """Entrena SARSA y devuelve (Q, recompensas_por_episodio)."""
    rng = np.random.default_rng(semilla)
    Q = np.zeros((env.n_estados, env.n_acciones))
    recompensas = np.zeros(n_episodios)

    for ep in range(n_episodios):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        total = 0.0
        for _ in range(max_pasos):
            s_sig, r, fin = env.step(a)
            a_sig = epsilon_greedy(Q, s_sig, epsilon, rng)
            # Actualizacion SARSA con la quintupla (s, a, r, s', a').
            # Si el estado es terminal, no arrastramos valor futuro.
            objetivo = r + gamma * Q[s_sig, a_sig] * (0.0 if fin else 1.0)
            Q[s, a] += alpha * (objetivo - Q[s, a])
            s, a = s_sig, a_sig
            total += r
            if fin:
                break
        recompensas[ep] = total
    return Q, recompensas


def media_movil(x, k=10):
    """Media movil trailing (promedio de las ultimas k muestras)."""
    if len(x) < k:
        return x.copy()
    return np.convolve(x, np.ones(k) / k, mode="valid")


def dibujar(env, Q, recompensas, k=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # --- Recompensa por episodio (suavizada) ---
    suave = media_movil(recompensas, k=k)
    ax1.plot(recompensas, color="#c7d2fe", lw=1, label="por episodio")
    ax1.plot(np.arange(len(suave)) + (k - 1), suave, color="#4f46e5", lw=2,
             label=f"media movil ({k})")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa total")
    ax1.set_title("Aprendizaje de SARSA en Cliff Walking")
    ax1.set_ylim(-200, 0)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # --- Politica greedy aprendida ---
    flechas = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    ax2.set_xlim(-0.5, env.columnas - 0.5)
    ax2.set_ylim(-0.5, env.filas - 0.5)
    ax2.set_xticks(range(env.columnas))
    ax2.set_yticks(range(env.filas))
    ax2.invert_yaxis()
    ax2.set_title("Politica aprendida (greedy)")
    ax2.grid(True, color="#cbd5e0")

    for fila in range(env.filas):
        for col in range(env.columnas):
            centro = (col, fila)
            if env.es_acantilado(fila, col):
                ax2.add_patch(plt.Rectangle((col - 0.5, fila - 0.5), 1, 1,
                                            color="#fca5a5"))
            elif (fila, col) == env.inicio:
                ax2.add_patch(plt.Rectangle((col - 0.5, fila - 0.5), 1, 1,
                                            color="#86efac"))
                ax2.text(*centro, "S", ha="center", va="center", weight="bold")
            elif (fila, col) == env.meta:
                ax2.add_patch(plt.Rectangle((col - 0.5, fila - 0.5), 1, 1,
                                            color="#93c5fd"))
                ax2.text(*centro, "G", ha="center", va="center", weight="bold")
            else:
                a = int(np.argmax(Q[env.id_estado(fila, col)]))
                ax2.text(*centro, flechas[a], ha="center", va="center",
                         fontsize=16, color="#4f46e5")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = CliffWalking()
    Q, recompensas = entrenar_sarsa(env, n_episodios=500, semilla=0)

    print(f"Recompensa media (ultimos 50 episodios): "
          f"{recompensas[-50:].mean():.1f}")
    print("Politica greedy (0=arriba 1=derecha 2=abajo 3=izquierda):")
    politica = np.argmax(Q, axis=1).reshape(env.filas, env.columnas)
    print(politica)

    dibujar(env, Q, recompensas)
