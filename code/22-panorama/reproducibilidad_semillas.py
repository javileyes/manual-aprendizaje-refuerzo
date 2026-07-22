"""
Reproducibilidad en RL: por que NUNCA debes fiarte de una sola semilla.
=======================================================================

Capitulo 22 del Manual de RL. Este script entrena el MISMO agente de
Q-Learning en el MISMO gridworld estocastico, pero con 20 semillas
aleatorias distintas, y dibuja dos paneles:

  * Izquierda: las 20 curvas de aprendizaje individuales (tenues) y UNA de
    ellas resaltada. Fijate en lo distintas que son entre si: si solo
    hubieras ejecutado la semilla resaltada, sacarias conclusiones
    equivocadas sobre "como de rapido/bien" aprende el algoritmo.
  * Derecha: la MEDIA de las 20 curvas con una banda de +/- una desviacion
    tipica. Ese es el resumen honesto: una tendencia central y su
    incertidumbre.

Moraleja practica del RL empirico: reporta media +/- dispersion sobre
varias semillas; nunca saques conclusiones de una unica ejecucion.

Como ejecutarlo en tu terminal:
    pip install -r requirements.txt
    python code/22-panorama/reproducibilidad_semillas.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) El entorno: un gridworld estocastico de 5x5
# ---------------------------------------------------------------------------
GRID = 5
N_ESTADOS = GRID * GRID
N_ACCIONES = 4                       # 0=arriba, 1=derecha, 2=abajo, 3=izquierda
MOV = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
INICIO = 0                           # esquina superior izquierda (0, 0)
META = N_ESTADOS - 1                 # esquina inferior derecha (4, 4)
P_RESBALON = 0.1                     # prob. de "resbalar" y ejecutar una accion al azar


def paso(estado, accion, rng):
    """Dinamica del entorno. Devuelve (nuevo_estado, recompensa, terminado).

    Es ESTOCASTICO: con probabilidad P_RESBALON el agente resbala y ejecuta
    una accion aleatoria en vez de la elegida. Esa aleatoriedad, junto con la
    exploracion epsilon-greedy, es la que hace que dos semillas distintas
    produzcan curvas de aprendizaje distintas.
    """
    if rng.random() < P_RESBALON:
        accion = int(rng.integers(N_ACCIONES))
    fila, col = divmod(estado, GRID)
    dfila, dcol = MOV[accion]
    fila = min(max(fila + dfila, 0), GRID - 1)
    col = min(max(col + dcol, 0), GRID - 1)
    nuevo = fila * GRID + col
    if nuevo == META:
        return nuevo, 0.0, True          # llegar a la meta termina el episodio
    return nuevo, -1.0, False            # cada paso cuesta -1 (queremos llegar rapido)


def epsilon_greedy(Q, estado, epsilon, rng):
    """Elige accion epsilon-greedy, con desempate aleatorio entre las mejores."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACCIONES))
    q = Q[estado]
    return int(rng.choice(np.flatnonzero(q == q.max())))


# ---------------------------------------------------------------------------
# 2) Q-Learning tabular: una ejecucion completa con una semilla dada
# ---------------------------------------------------------------------------
def entrena_una_semilla(semilla, n_episodios=200, alpha=0.5, gamma=0.95,
                        epsilon=0.1, max_pasos=100):
    """Entrena Q-Learning y devuelve el retorno de cada episodio (un vector)."""
    rng = np.random.default_rng(semilla)
    Q = np.zeros((N_ESTADOS, N_ACCIONES))
    retornos = np.zeros(n_episodios)
    for ep in range(n_episodios):
        estado, total = INICIO, 0.0
        for _ in range(max_pasos):
            accion = epsilon_greedy(Q, estado, epsilon, rng)
            s2, r, fin = paso(estado, accion, rng)
            # Objetivo off-policy: usa la MEJOR accion del estado siguiente.
            objetivo = r + gamma * Q[s2].max() * (0.0 if fin else 1.0)
            Q[estado, accion] += alpha * (objetivo - Q[estado, accion])
            estado, total = s2, total + r
            if fin:
                break
        retornos[ep] = total
    return retornos


def media_movil(x, w=15):
    """Suaviza una serie con una media movil de ventana w (para ver la tendencia)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


# ---------------------------------------------------------------------------
# 3) Entrenamos con MUCHAS semillas y resumimos con media +/- desviacion
# ---------------------------------------------------------------------------
def experimento(n_semillas=20, **kw):
    """Ejecuta el entrenamiento con n_semillas semillas distintas (0, 1, ...).

    Devuelve una matriz (n_semillas x n_episodios) con el retorno por episodio.
    """
    curvas = [entrena_una_semilla(semilla=s, **kw) for s in range(n_semillas)]
    return np.array(curvas)


def main():
    n_semillas = 20
    curvas = experimento(n_semillas=n_semillas)          # (n_semillas, n_episodios)

    # Suavizamos cada curva para ver la tendencia sin el ruido episodio a episodio.
    suaves = np.array([media_movil(c) for c in curvas])
    ejex = np.arange(suaves.shape[1])
    media = suaves.mean(axis=0)
    desv = suaves.std(axis=0, ddof=1)   # desviacion tipica MUESTRAL (divide por N-1)

    # Una semilla "cualquiera" (la que un incauto podria haber ejecutado una vez).
    elegida = int(np.random.default_rng(0).integers(n_semillas))

    # --- Resumen numerico por pantalla ---
    finales = curvas[:, -50:].mean(axis=1)   # retorno medio de los ultimos 50 episodios
    print(f"Q-Learning en gridworld estocastico 5x5 | {n_semillas} semillas")
    print("-" * 60)
    print("Retorno final (media de los ultimos 50 episodios) entre semillas:")
    print(f"  media                : {finales.mean():7.2f}")
    print(f"  desviacion tipica    : {finales.std(ddof=1):7.2f}")
    print(f"  mejor semilla        : {finales.max():7.2f}")
    print(f"  peor  semilla        : {finales.min():7.2f}")
    print("-" * 60)
    print(f"Entre la mejor y la peor semilla hay "
          f"{finales.max() - finales.min():.1f} puntos de retorno de diferencia.")
    print("Una sola ejecucion puede caer en cualquier punto de ese rango: por")
    print("eso se reporta la MEDIA +/- dispersion, no una unica semilla.")

    # --- Grafica de dos paneles ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)

    # (a) Las 20 curvas individuales + una resaltada.
    for c in suaves:
        ax1.plot(ejex, c, color="#94a3b8", lw=0.8, alpha=0.5)
    ax1.plot(ejex, suaves[elegida], color="#dc2626", lw=2.0,
             label=f"una sola semilla (#{elegida})")
    ax1.set_title(f"Las {n_semillas} ejecuciones individuales")
    ax1.set_xlabel("episodio")
    ax1.set_ylabel("retorno (media movil)")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.2)

    # (b) Media +/- una desviacion tipica.
    ax2.fill_between(ejex, media - desv, media + desv,
                     color="#4f46e5", alpha=0.22, label="media ± 1 desv. tipica")
    ax2.plot(ejex, media, color="#4f46e5", lw=2.2, label="media de las semillas")
    ax2.set_title("El resumen honesto: media y banda")
    ax2.set_xlabel("episodio")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.2)

    fig.suptitle("Por que una sola semilla engana en RL", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
