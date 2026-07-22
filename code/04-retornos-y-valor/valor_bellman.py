"""
Retornos y funciones de valor: la ecuación de Bellman en un gridworld.

Este script calcula la función de valor V^pi de una política fija (moverse al
azar en las cuatro direcciones) sobre un gridworld de 4x4 con dos casillas
terminales (las dos esquinas opuestas). Lo hace de dos formas y comprueba que
coinciden:

  1. EXACTA, resolviendo el sistema lineal de las ecuaciones de Bellman
        V = (I - gamma * P_pi)^{-1} r_pi
     con numpy.linalg.solve.

  2. APROXIMADA, con una estimación de Monte Carlo: se simulan muchos
     episodios desde cada estado, se promedia el retorno descontado y se
     compara con la solución exacta.

Al final dibuja el mapa de calor de V^pi y un diagrama de dispersión que
confirma que Monte Carlo reproduce la solución de Bellman.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/04-retornos-y-valor/valor_bellman.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- El entorno: gridworld 4x4 -------------------------------------------
FILAS, COLUMNAS = 4, 4
N = FILAS * COLUMNAS          # 16 estados, numerados 0..15 (fila * 4 + columna)
TERMINALES = {0, N - 1}       # las dos esquinas opuestas son terminales
GAMMA = 0.9                   # factor de descuento

# Cuatro acciones: 0=arriba, 1=abajo, 2=izquierda, 3=derecha
MOVIMIENTOS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
N_ACCIONES = len(MOVIMIENTOS)


def siguiente_estado(s, a):
    """Estado al que se llega desde s aplicando la acción a.

    Si el movimiento sale del tablero, el agente se queda donde está.
    Las casillas terminales son absorbentes: no se sale de ellas.
    """
    if s in TERMINALES:
        return s
    fila, col = divmod(s, COLUMNAS)
    df, dc = MOVIMIENTOS[a]
    nueva_fila = min(max(fila + df, 0), FILAS - 1)
    nueva_col = min(max(col + dc, 0), COLUMNAS - 1)
    return nueva_fila * COLUMNAS + nueva_col


# Tabla de transiciones precalculada: sig[s, a] -> estado siguiente
SIG = np.array([[siguiente_estado(s, a) for a in range(N_ACCIONES)]
                for s in range(N)])

# Máscara de estados terminales (para vectorizar Monte Carlo)
ES_TERMINAL = np.zeros(N, dtype=bool)
for s in TERMINALES:
    ES_TERMINAL[s] = True


def construir_bellman():
    """Devuelve (P_pi, r_pi) de la política uniforme (moverse al azar).

    P_pi[s, s'] = probabilidad de pasar de s a s' bajo la política.
    r_pi[s]     = recompensa esperada inmediata al salir de s.
    Cada paso en una casilla no terminal cuesta -1; en las terminales, 0.
    """
    P = np.zeros((N, N))
    r = np.zeros(N)
    for s in range(N):
        if s in TERMINALES:
            P[s, s] = 1.0          # absorbente, sin recompensa
            continue
        r[s] = -1.0                # cada paso cuesta -1
        for a in range(N_ACCIONES):
            P[s, SIG[s, a]] += 1.0 / N_ACCIONES   # política uniforme
    return P, r


def valor_exacto(P, r, gamma):
    """Resuelve V = (I - gamma * P)^{-1} r  sin invertir explícitamente."""
    A = np.eye(N) - gamma * P
    return np.linalg.solve(A, r)


def valor_montecarlo(gamma, n_episodios=4000, max_pasos=400, semilla=0):
    """Estima V^pi con Monte Carlo: promedio del retorno descontado.

    Para cada estado inicial no terminal simula n_episodios en paralelo
    (vectorizado con numpy) siguiendo la política uniforme y promedia G_0.
    """
    rng = np.random.default_rng(semilla)
    V = np.zeros(N)
    for inicio in range(N):
        if ES_TERMINAL[inicio]:
            continue  # V de un estado terminal es 0 por definición
        actual = np.full(n_episodios, inicio)
        activo = np.ones(n_episodios, dtype=bool)   # aún no ha terminado
        G = np.zeros(n_episodios)
        descuento = np.ones(n_episodios)
        for _ in range(max_pasos):
            if not activo.any():
                break
            acciones = rng.integers(0, N_ACCIONES, size=n_episodios)
            # cada transición desde una casilla no terminal aporta -1
            G += descuento * (-1.0) * activo
            nuevo = SIG[actual, acciones]
            actual = np.where(activo, nuevo, actual)
            descuento *= gamma
            activo = activo & ~ES_TERMINAL[actual]
        V[inicio] = G.mean()
    return V


def main():
    P, r = construir_bellman()
    V_exacto = valor_exacto(P, r, GAMMA)
    V_mc = valor_montecarlo(GAMMA)

    error = np.abs(V_exacto - V_mc)
    print("V^pi por estado (gridworld 4x4, política uniforme, gamma = 0.9)")
    print("estado |   Bellman | Monte Carlo | |dif|")
    print("-------+-----------+-------------+------")
    for s in range(N):
        marca = " (terminal)" if ES_TERMINAL[s] else ""
        print(f"  {s:2d}   | {V_exacto[s]:8.3f}  |  {V_mc[s]:8.3f}   | {error[s]:.3f}{marca}")
    print(f"\nError máximo entre Bellman y Monte Carlo: {error.max():.3f}")

    # --- Visualización -----------------------------------------------------
    rejilla = V_exacto.reshape(FILAS, COLUMNAS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    im = ax1.imshow(rejilla, cmap="viridis")
    ax1.set_title("$V^\\pi(s)$ exacta (ecuación de Bellman)")
    ax1.set_xticks(range(COLUMNAS))
    ax1.set_yticks(range(FILAS))
    for s in range(N):
        f, c = divmod(s, COLUMNAS)
        etiqueta = "TERM" if ES_TERMINAL[s] else f"{V_exacto[s]:.2f}"
        ax1.text(c, f, etiqueta, ha="center", va="center",
                 color="white", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2.scatter(V_exacto, V_mc, s=60, edgecolor="black", zorder=3)
    lo, hi = V_exacto.min() - 0.5, V_exacto.max() + 0.5
    ax2.plot([lo, hi], [lo, hi], "--", color="crimson", label="y = x")
    ax2.set_xlabel("Bellman (exacta)")
    ax2.set_ylabel("Monte Carlo (estimada)")
    ax2.set_title("Monte Carlo reproduce a Bellman")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
