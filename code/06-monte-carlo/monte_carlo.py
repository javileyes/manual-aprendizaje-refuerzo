"""
Control de Monte Carlo (primera visita, epsilon-greedy) sobre un gridworld.

Aprende una politica SIN modelo: no conoce las probabilidades del entorno, solo
juega episodios completos y promedia los retornos observados. Es la idea central
del capitulo 6 del Manual de Aprendizaje por Refuerzo.

El agente vive en una cuadricula 4x4 con dos paredes. Sale de la esquina superior
izquierda y busca la meta (esquina inferior derecha). Cada paso cuesta -1 y llegar
a la meta da +10. Al final se muestra:
  - la recompensa media por episodio (media movil) subiendo a medida que aprende,
  - la politica greedy final como flechas sobre la cuadricula.

Como ejecutarlo:
    pip install -r requirements.txt
    python code/06-monte-carlo/monte_carlo.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- El entorno: gridworld 4x4 con dos paredes ---
FILAS, COLS = 4, 4
INICIO = (0, 0)
META = (3, 3)
PAREDES = {(1, 1), (2, 2)}
N_ESTADOS = FILAS * COLS
N_ACCIONES = 4                      # 0 arriba, 1 abajo, 2 izquierda, 3 derecha
MOV = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
FLECHAS = {0: "^", 1: "v", 2: "<", 3: ">"}
GAMMA = 0.95


def a_indice(fila, col):
    """Convierte una casilla (fila, col) en un indice de estado 0..15."""
    return fila * COLS + col


def paso(estado, accion):
    """Dinamica del entorno: devuelve (nuevo_estado, recompensa, terminado)."""
    fila, col = divmod(estado, COLS)
    df, dc = MOV[accion]
    nf, nc = fila + df, col + dc
    # Chocar contra el borde o una pared: te quedas donde estabas (pero el paso cuesta).
    if nf < 0 or nf >= FILAS or nc < 0 or nc >= COLS or (nf, nc) in PAREDES:
        nf, nc = fila, col
    nuevo = a_indice(nf, nc)
    if (nf, nc) == META:
        return nuevo, 10.0, True
    return nuevo, -1.0, False


def elegir_accion(estado, Q, epsilon, rng):
    """Politica epsilon-greedy: explora con prob. epsilon, si no explota Q."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACCIONES))
    q = Q[estado]
    # argmax con desempate aleatorio (clave al principio, cuando Q es todo ceros).
    candidatos = np.flatnonzero(q == q.max())
    return int(rng.choice(candidatos))


def generar_episodio(Q, epsilon, rng, max_pasos=50):
    """Juega un episodio completo siguiendo la politica epsilon-greedy."""
    estados, acciones, recompensas = [], [], []
    estado = a_indice(*INICIO)
    for _ in range(max_pasos):
        accion = elegir_accion(estado, Q, epsilon, rng)
        nuevo, r, terminado = paso(estado, accion)
        estados.append(estado)
        acciones.append(accion)
        recompensas.append(r)
        estado = nuevo
        if terminado:
            break
    return estados, acciones, recompensas


def mc_control(num_episodios=3000, semilla=0):
    """Control MC de primera visita con mejora epsilon-greedy (GPI)."""
    rng = np.random.default_rng(semilla)
    Q = np.zeros((N_ESTADOS, N_ACCIONES))     # valor accion-estado estimado
    N = np.zeros((N_ESTADOS, N_ACCIONES))     # nº de retornos promediados en cada (s, a)
    historial = np.zeros(num_episodios)       # recompensa total de cada episodio

    for ep in range(num_episodios):
        # Exploracion que decae: mucha al principio, poca (pero > 0) al final.
        epsilon = max(0.05, 1.0 - ep / (0.8 * num_episodios))
        estados, acciones, recompensas = generar_episodio(Q, epsilon, rng)
        historial[ep] = sum(recompensas)

        # Indice de la PRIMERA aparicion de cada par (estado, accion) del episodio.
        primera = {}
        for t, sa in enumerate(zip(estados, acciones)):
            if sa not in primera:
                primera[sa] = t

        # Recorremos el episodio hacia atras acumulando el retorno G.
        G = 0.0
        for t in range(len(estados) - 1, -1, -1):
            s, a = estados[t], acciones[t]
            G = GAMMA * G + recompensas[t]
            if primera[(s, a)] == t:          # solo la primera visita
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]   # media incremental

    return Q, historial


def evaluar_politica(Q, rng, episodios=500):
    """Recompensa media de la politica greedy (epsilon = 0)."""
    totales = [sum(generar_episodio(Q, 0.0, rng)[2]) for _ in range(episodios)]
    return float(np.mean(totales))


def media_movil(x, k=50):
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[k:] - c[:-k]) / k


def imprimir_politica(Q):
    print("Politica aprendida (^ v < > , # pared, G meta):")
    for fila in range(FILAS):
        linea = []
        for col in range(COLS):
            if (fila, col) in PAREDES:
                linea.append("#")
            elif (fila, col) == META:
                linea.append("G")
            else:
                linea.append(FLECHAS[int(np.argmax(Q[a_indice(fila, col)]))])
        print("   " + "  ".join(linea))


def dibujar(historial, Q):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Curva de aprendizaje ---
    ax1.plot(media_movil(historial, 50), color="#4f46e5")
    ax1.axhline(5, ls="--", color="#059669", lw=1, label="optimo (+5)")
    ax1.set_xlabel("episodio")
    ax1.set_ylabel("recompensa total (media movil 50)")
    ax1.set_title("Aprendizaje por Monte Carlo")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Politica greedy sobre la cuadricula ---
    ax2.set_title("Politica aprendida")
    ax2.set_xlim(-0.5, COLS - 0.5)
    ax2.set_ylim(-0.5, FILAS - 0.5)
    ax2.set_xticks(range(COLS))
    ax2.set_yticks(range(FILAS))
    ax2.set_aspect("equal")
    ax2.invert_yaxis()
    ax2.grid(True)
    for fila in range(FILAS):
        for col in range(COLS):
            if (fila, col) in PAREDES:
                ax2.add_patch(plt.Rectangle((col - 0.5, fila - 0.5), 1, 1, color="#334155"))
                continue
            if (fila, col) == META:
                ax2.add_patch(plt.Rectangle((col - 0.5, fila - 0.5), 1, 1, color="#bbf7d0"))
                ax2.text(col, fila, "META", ha="center", va="center",
                         color="#047857", fontsize=11, fontweight="bold")
                continue
            if (fila, col) == INICIO:
                ax2.add_patch(plt.Rectangle((col - 0.5, fila - 0.5), 1, 1, color="#e0e7ff"))
            a = int(np.argmax(Q[a_indice(fila, col)]))
            ax2.text(col, fila, FLECHAS[a], ha="center", va="center", fontsize=22)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Q, historial = mc_control(num_episodios=3000, semilla=0)
    media_greedy = evaluar_politica(Q, np.random.default_rng(123))
    print(f"Recompensa media de la politica greedy final: {media_greedy:.2f}")
    print(f"(el optimo desde el inicio es +5: 6 pasos = 5 x (-1) + 10)\n")
    imprimir_politica(Q)
    dibujar(historial, Q)
