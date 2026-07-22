"""
Bandido multibrazo: el testbed de 10 brazos de Sutton & Barto.
==============================================================

Compara cuatro estrategias de seleccion de accion en el problema del
bandido multibrazo con k = 10 brazos:

  * greedy               (epsilon = 0): siempre explota, nunca explora.
  * epsilon-greedy 0.01  (explora un 1 % de las veces).
  * epsilon-greedy 0.1   (explora un 10 % de las veces).
  * UCB con c = 2        (exploracion dirigida por la incertidumbre).

Cada estrategia se promedia sobre 500 ejecuciones independientes de
1000 pasos. En cada ejecucion los valores reales q*(a) se muestrean de
una normal N(0, 1) y la recompensa de tirar del brazo a es
R = q*(a) + ruido, con ruido ~ N(0, 1).

Se dibujan dos graficas, como en el libro de Sutton & Barto:
  1) recompensa media por paso,
  2) porcentaje de veces que se elige la accion optima.

Todo esta vectorizado con numpy: las 500 ejecuciones avanzan en
paralelo (una fila por ejecucion), asi que corre en pocos segundos.

Como ejecutarlo:
    pip install -r requirements.txt
    python code/02-bandido-multibrazo/bandidos.py
"""

import numpy as np
import matplotlib.pyplot as plt


def simular(k=10, n_ejec=500, n_pasos=1000, epsilon=0.0, ucb_c=None, semilla=0):
    """Ejecuta n_ejec bandidos en paralelo y devuelve dos curvas promedio.

    Parametros
    ----------
    k        : numero de brazos.
    n_ejec   : numero de ejecuciones independientes que se promedian.
    n_pasos  : pasos (tiradas) por ejecucion.
    epsilon  : probabilidad de explorar (0 = greedy puro). Se ignora si ucb_c.
    ucb_c    : si no es None, usa UCB con esta constante c en vez de e-greedy.
    semilla  : semilla del generador para reproducibilidad.

    Devuelve (recompensa_media, prop_optima): dos arrays de longitud
    n_pasos con la recompensa promediada sobre ejecuciones en cada paso y
    la proporcion de ejecuciones que eligieron la accion optima.
    """
    rng = np.random.default_rng(semilla)

    # Valores reales q*(a): una fila por ejecucion, una columna por brazo.
    q_estrella = rng.standard_normal((n_ejec, k))
    accion_optima = np.argmax(q_estrella, axis=1)      # (n_ejec,)

    Q = np.zeros((n_ejec, k))       # estimaciones Q_t(a)
    N = np.zeros((n_ejec, k))       # veces que se ha elegido cada accion
    filas = np.arange(n_ejec)       # indices [0, 1, ..., n_ejec-1]

    recompensa_media = np.zeros(n_pasos)
    prop_optima = np.zeros(n_pasos)

    for t in range(n_pasos):
        if ucb_c is not None:
            # Cota superior de confianza (UCB). Un brazo nunca probado
            # (N = 0) recibe prioridad infinita para forzar que se pruebe.
            with np.errstate(divide="ignore", invalid="ignore"):
                bonus = ucb_c * np.sqrt(np.log(t + 1) / N)
            bonus[N == 0] = np.inf
            A = np.argmax(Q + bonus, axis=1)
        else:
            A = np.argmax(Q, axis=1)                    # accion greedy
            if epsilon > 0:
                explora = rng.random(n_ejec) < epsilon
                A_azar = rng.integers(0, k, size=n_ejec)
                A = np.where(explora, A_azar, A)

        # Recompensa de cada ejecucion: q*(A) mas ruido gaussiano.
        R = q_estrella[filas, A] + rng.standard_normal(n_ejec)

        # Media incremental: Q <- Q + (1/N) (R - Q).
        N[filas, A] += 1
        Q[filas, A] += (R - Q[filas, A]) / N[filas, A]

        recompensa_media[t] = R.mean()
        prop_optima[t] = np.mean(A == accion_optima)

    return recompensa_media, prop_optima


# --- Comparamos las cuatro estrategias ---
estrategias = {
    "greedy (e=0)":      dict(epsilon=0.0),
    "e-greedy (e=0.01)": dict(epsilon=0.01),
    "e-greedy (e=0.1)":  dict(epsilon=0.1),
    "UCB (c=2)":         dict(ucb_c=2.0),
}

resultados = {}
print(f"{'Estrategia':20s} {'recomp. media':>14s} {'% optima':>10s}")
print("-" * 46)
for nombre, kwargs in estrategias.items():
    rec, opt = simular(**kwargs)
    resultados[nombre] = (rec, opt)
    # Promedios sobre los ultimos 100 pasos (comportamiento ya asentado).
    print(f"{nombre:20s} {rec[-100:].mean():14.3f} {100 * opt[-100:].mean():9.1f}%")

# --- Dos graficas: recompensa media y % de accion optima ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
for nombre, (rec, opt) in resultados.items():
    ax1.plot(rec, label=nombre, linewidth=1.3)
    ax2.plot(100 * opt, label=nombre, linewidth=1.3)

ax1.set_xlabel("Pasos")
ax1.set_ylabel("Recompensa media")
ax1.set_title("Recompensa media por paso")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.set_xlabel("Pasos")
ax2.set_ylabel("% de acciones optimas")
ax2.set_title("Porcentaje de accion optima")
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
