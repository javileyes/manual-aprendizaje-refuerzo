"""
Q-Learning vs SARSA en el entorno Cliff Walking (el "paseo por el acantilado").
================================================================================

Este script compara los dos algoritmos de control por diferencias temporales
del Capitulo 9 del Manual de RL sobre el mismo problema:

  * Q-Learning: control TD *off-policy*. En el objetivo usa max_a Q(S', a),
    es decir, aprende asumiendo que a partir del siguiente estado actuara de
    forma OPTIMA, aunque durante el entrenamiento explore.
  * SARSA: control TD *on-policy*. En el objetivo usa Q(S', A'), donde A' es la
    accion que realmente tomara la politica exploratoria (epsilon-greedy).

La leccion clasica (Sutton & Barto, ejemplo 6.6): Q-Learning aprende el camino
OPTIMO pegado al acantilado (recompensa -13), pero durante el entrenamiento se
cae mas a menudo por culpa de la exploracion. SARSA aprende un camino mas SEGURO
y alejado del borde, con lo que su recompensa media *durante* el entrenamiento es
mayor, aunque su camino final sea un poco mas largo.

Al ejecutarlo veras:
  1) Un resumen numerico por pantalla (recompensa media final y longitud del
     camino greedy de cada algoritmo).
  2) Una grafica con la recompensa por episodio de ambos.
  3) Una grafica con las dos politicas greedy resultantes sobre la rejilla.

Como ejecutarlo:
    pip install -r requirements.txt
    python code/09-q-learning/q_learning_cliff.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- El entorno: Cliff Walking, una rejilla de 4 filas x 12 columnas ---
FILAS, COLS = 4, 12
INICIO = (FILAS - 1, 0)         # esquina inferior izquierda
META = (FILAS - 1, COLS - 1)    # esquina inferior derecha
N_ESTADOS = FILAS * COLS
N_ACCIONES = 4                  # 0=arriba, 1=derecha, 2=abajo, 3=izquierda
MOV = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def idx(fila, col):
    """Convierte coordenadas (fila, columna) en un indice de estado 0..47."""
    return fila * COLS + col


def es_acantilado(fila, col):
    """El acantilado ocupa la fila inferior entre el inicio y la meta."""
    return fila == FILAS - 1 and 0 < col < COLS - 1


def paso(estado, accion):
    """Dinamica del entorno: devuelve (nuevo_estado, recompensa, terminado)."""
    fila, col = divmod(estado, COLS)
    dfila, dcol = MOV[accion]
    fila = min(max(fila + dfila, 0), FILAS - 1)
    col = min(max(col + dcol, 0), COLS - 1)
    if es_acantilado(fila, col):
        return idx(*INICIO), -100.0, False   # te caes: -100 y vuelta al inicio
    if (fila, col) == META:
        return idx(fila, col), -1.0, True     # meta alcanzada
    return idx(fila, col), -1.0, False        # cada paso cuesta -1


def epsilon_greedy(Q, estado, epsilon, rng):
    """Elige accion epsilon-greedy con desempate aleatorio entre las mejores."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACCIONES))
    q = Q[estado]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def entrena(algoritmo, n_episodios, alpha, gamma, epsilon, semilla, max_pasos=1000):
    """Entrena 'q-learning' o 'sarsa'. Devuelve (Q, recompensa_por_episodio)."""
    rng = np.random.default_rng(semilla)
    Q = np.zeros((N_ESTADOS, N_ACCIONES))
    retornos = np.zeros(n_episodios)
    for ep in range(n_episodios):
        estado = idx(*INICIO)
        accion = epsilon_greedy(Q, estado, epsilon, rng)   # A inicial (SARSA)
        total = 0.0
        for _ in range(max_pasos):
            s2, r, fin = paso(estado, accion)
            # Misma politica de comportamiento en ambos: la accion que se tomara en s2.
            a2 = epsilon_greedy(Q, s2, epsilon, rng)
            # La UNICA diferencia entre SARSA y Q-Learning esta en el objetivo TD:
            if algoritmo == "sarsa":
                bootstrap = Q[s2, a2]        # on-policy: Q(S', A') con la accion real
            else:                            # q-learning
                bootstrap = Q[s2].max()      # off-policy: max_a Q(S', a), la MEJOR accion
            objetivo = r + gamma * bootstrap * (0.0 if fin else 1.0)
            Q[estado, accion] += alpha * (objetivo - Q[estado, accion])
            estado, accion = s2, a2
            total += r
            if fin:
                break
        retornos[ep] = total
    return Q, retornos


def promedia_semillas(algoritmo, n_runs, n_episodios, **kw):
    """Promedia las curvas de recompensa sobre varias semillas (menos ruido).

    Devuelve tambien la Q de la PRIMERA semilla (semilla 0), que usaremos como
    politica representativa: al ser una semilla fija, el resultado es
    reproducible en cualquier maquina.
    """
    curvas = np.zeros((n_runs, n_episodios))
    Q0 = None
    for k in range(n_runs):
        Q, curvas[k] = entrena(algoritmo, n_episodios, semilla=k, **kw)
        if k == 0:
            Q0 = Q
    return Q0, curvas.mean(axis=0)


def media_movil(x, w=10):
    """Suaviza una serie con una media movil de ventana w."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def camino_greedy(Q, max_pasos=200):
    """Recorre la politica greedy pura desde el inicio.

    Devuelve (pasos, llego). 'llego' es False si la politica entra en un ciclo
    y nunca alcanza la meta (algo que le puede pasar a la Q de SARSA en estados
    poco visitados, porque sus valores incluyen el coste de explorar).
    """
    estado, pasos, visitados = idx(*INICIO), 0, set()
    while pasos < max_pasos:
        if estado in visitados:
            return pasos, False       # ciclo: la politica no llega a la meta
        visitados.add(estado)
        s2, _, fin = paso(estado, int(np.argmax(Q[estado])))
        pasos += 1
        estado = s2
        if fin:
            return pasos, True
    return pasos, False


def dibuja_politica(ax, Q, titulo):
    """Dibuja en 'ax' la accion greedy de cada casilla como una flecha."""
    flechas = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    ax.set_title(titulo, fontsize=11)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(FILAS - 0.5, -0.5)   # fila 0 arriba
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for f in range(FILAS):
        for c in range(COLS):
            ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1,
                                       fill=False, edgecolor="#cbd5e0", lw=0.8))
            if es_acantilado(f, c):
                ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, color="#f6b6b6"))
                ax.text(c, f, "✕", ha="center", va="center", color="#7a1f1f")
            elif (f, c) == INICIO:
                ax.text(c, f, "S", ha="center", va="center", fontweight="bold")
            elif (f, c) == META:
                ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, color="#bfe3bf"))
                ax.text(c, f, "G", ha="center", va="center", fontweight="bold")
            else:
                a = int(np.argmax(Q[idx(f, c)]))
                ax.text(c, f, flechas[a], ha="center", va="center", fontsize=13)


def main():
    params = dict(alpha=0.5, gamma=1.0, epsilon=0.1)
    n_episodios, n_runs = 500, 6

    Q_q, rec_q = promedia_semillas("q-learning", n_runs, n_episodios, **params)
    Q_s, rec_s = promedia_semillas("sarsa", n_runs, n_episodios, **params)

    # --- Resumen numerico ---
    lq, _ = camino_greedy(Q_q)
    ls, _ = camino_greedy(Q_s)
    print("Cliff Walking  |  alpha=0.5, gamma=1.0, epsilon=0.1, "
          f"{n_episodios} episodios, media de {n_runs} semillas")
    print("-" * 66)
    print(f"{'':14s}{'recompensa media':>20s}{'camino greedy':>18s}")
    print(f"{'Q-Learning':14s}{rec_q[-100:].mean():>20.1f}{lq:>13d} pasos")
    print(f"{'SARSA':14s}{rec_s[-100:].mean():>20.1f}{ls:>13d} pasos")
    print("-" * 66)
    print("Q-Learning encuentra el camino mas corto (pegado al acantilado) pero,")
    print("al explorar, se cae mas durante el entrenamiento. SARSA aprende un")
    print("camino mas seguro y logra mejor recompensa mientras aprende.")

    # --- Grafica 1: recompensa por episodio ---
    plt.figure(figsize=(8, 4.2))
    ep = np.arange(len(media_movil(rec_q)))
    plt.plot(ep, media_movil(rec_q), label="Q-Learning (off-policy)", color="#4f46e5")
    plt.plot(ep, media_movil(rec_s), label="SARSA (on-policy)", color="#0ea5e9")
    plt.axhline(-13, ls="--", lw=1, color="#94a3b8")
    plt.text(len(ep) * 0.5, -13 + 1.5, "camino optimo (-13)", color="#64748b", fontsize=9)
    plt.ylim(-100, -5)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa por episodio (media movil)")
    plt.title("Recompensa durante el entrenamiento en Cliff Walking")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # --- Grafica 2: politicas greedy resultantes ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.6))
    dibuja_politica(ax1, Q_q, "Q-Learning: camino OPTIMO y arriesgado (borde del acantilado)")
    dibuja_politica(ax2, Q_s, "SARSA: camino SEGURO (se aleja del acantilado)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
