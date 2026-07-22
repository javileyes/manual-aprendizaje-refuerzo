"""
sesgo_maximizacion.py - El sesgo de maximizacion y como lo corrige Double Q-learning.

Reproduce el experimento clasico (Sutton & Barto, Ejemplo 6.7). Un MDP diminuto:
  - Estado inicial A con dos acciones: "izquierda" lleva a B (recompensa 0),
    "derecha" termina el episodio (recompensa 0).
  - Estado B con N_B acciones; cada una termina con recompensa ~ Normal(-0.1, 1).
Como la recompensa media en B es NEGATIVA (-0.1), la accion optima desde A es
"derecha". Pero Q-learning usa max_a Q(B,a) sobre estimaciones ruidosas y las
sobreestima: cree que B es bueno y elige "izquierda" demasiado a menudo. Double
Q-learning separa la seleccion y la evaluacion de la accion, y corrige el sesgo.

Todo esta vectorizado sobre RUNS ejecuciones independientes con numpy, asi que
corre en pocos segundos (tambien en el navegador con Pyodide).

Ejecucion en tu terminal:
    python code/13-mejoras-dqn/sesgo_maximizacion.py
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Parametros del MDP y del aprendizaje ---
N_B = 10               # numero de acciones en el estado B
MU, SIGMA = -0.1, 1.0  # recompensa en B ~ Normal(-0.1, 1): en MEDIA es negativa
ALPHA = 0.1            # tasa de aprendizaje
EPS = 0.1              # exploracion epsilon-greedy
GAMMA = 1.0
RUNS = 2000            # ejecuciones independientes que promediamos
EPISODES = 300         # episodios por ejecucion

rng = np.random.default_rng(0)
idx = np.arange(RUNS)  # indice de filas para leer/escribir por-run


def eps_greedy(Q):
    """Accion epsilon-greedy por fila, vectorizada sobre las RUNS ejecuciones."""
    R, n = Q.shape
    # desempate aleatorio con ruido minusculo (no altera diferencias reales)
    greedy = np.argmax(Q + rng.uniform(0, 1e-9, size=(R, n)), axis=1)
    aleatoria = rng.integers(0, n, size=R)
    explora = rng.random(R) < EPS
    return np.where(explora, aleatoria, greedy)


def q_learning():
    """Q-learning tabular. Devuelve el % de veces que elige 'izquierda' por episodio."""
    QA = np.zeros((RUNS, 2))    # A: 0=izquierda->B, 1=derecha->fin
    QB = np.zeros((RUNS, N_B))  # B: N_B acciones -> fin (recompensa ruidosa)
    pct_izq = np.zeros(EPISODES)
    for ep in range(EPISODES):
        aA = eps_greedy(QA)
        izq = aA == 0
        der = ~izq
        pct_izq[ep] = izq.mean()
        # "derecha": termina con recompensa 0  ->  objetivo = 0
        QA[idx[der], 1] += ALPHA * (0.0 - QA[idx[der], 1])
        # "izquierda": pasa a B (recompensa 0) ->  objetivo = GAMMA * max_a Q(B,a)
        objetivo = GAMMA * QB.max(axis=1)
        QA[idx[izq], 0] += ALPHA * (objetivo[izq] - QA[idx[izq], 0])
        # en B: elegimos accion y terminamos con recompensa ruidosa
        aB = eps_greedy(QB)
        r = rng.normal(MU, SIGMA, size=RUNS)
        nuevo = QB[idx, aB] + ALPHA * (r - QB[idx, aB])
        QB[idx[izq], aB[izq]] = nuevo[izq]   # solo las runs que fueron a B
    return pct_izq


def bootstrap_doble(Q_sel, Q_eval):
    """Objetivo Double en B (no terminal): Q_sel ELIGE la accion, Q_eval la EVALUA."""
    a_star = np.argmax(Q_sel, axis=1)
    return GAMMA * Q_eval[idx, a_star]


def double_q_learning():
    """Double Q-learning tabular con dos tablas Q1 y Q2."""
    Q1A = np.zeros((RUNS, 2)); Q2A = np.zeros((RUNS, 2))
    Q1B = np.zeros((RUNS, N_B)); Q2B = np.zeros((RUNS, N_B))
    pct_izq = np.zeros(EPISODES)
    for ep in range(EPISODES):
        aA = eps_greedy(Q1A + Q2A)   # comportamiento: epsilon-greedy sobre Q1+Q2
        izq = aA == 0
        der = ~izq
        pct_izq[ep] = izq.mean()

        # --- transicion desde A (una moneda por run decide que tabla se actualiza) ---
        c = rng.random(RUNS) < 0.5       # True -> actualizar Q1 ; False -> Q2
        tA1 = bootstrap_doble(Q1B, Q2B)  # si toca Q1: selecciona con Q1, evalua con Q2
        tA2 = bootstrap_doble(Q2B, Q1B)
        m = c & izq;  Q1A[idx[m], 0] += ALPHA * (tA1[m] - Q1A[idx[m], 0])
        m = c & der;  Q1A[idx[m], 1] += ALPHA * (0.0    - Q1A[idx[m], 1])
        m = ~c & izq; Q2A[idx[m], 0] += ALPHA * (tA2[m] - Q2A[idx[m], 0])
        m = ~c & der; Q2A[idx[m], 1] += ALPHA * (0.0    - Q2A[idx[m], 1])

        # --- transicion desde B (solo runs "izquierda"); es terminal: objetivo = r ---
        aB = eps_greedy(Q1B + Q2B)
        r = rng.normal(MU, SIGMA, size=RUNS)
        c = rng.random(RUNS) < 0.5
        m = c & izq;  Q1B[idx[m], aB[m]] += ALPHA * (r[m] - Q1B[idx[m], aB[m]])
        m = ~c & izq; Q2B[idx[m], aB[m]] += ALPHA * (r[m] - Q2B[idx[m], aB[m]])
    return pct_izq


print(f"Entrenando: {RUNS} ejecuciones x {EPISODES} episodios (vectorizado)...")
ql = q_learning()
dq = double_q_learning()
optimo = EPS / 2  # con epsilon-greedy y 2 acciones, elegir la mala baja como mucho a eps/2

print(f"Q-learning        -> % 'izquierda': inicio={100*ql[:10].mean():5.1f}%  final={100*ql[-50:].mean():4.1f}%")
print(f"Double Q-learning -> % 'izquierda': inicio={100*dq[:10].mean():5.1f}%  final={100*dq[-50:].mean():4.1f}%")
print(f"Optimo (eps/2)    -> {100*optimo:.1f}%")

plt.figure(figsize=(8, 4.5))
plt.plot(100 * ql, label="Q-learning", color="#dc2626", lw=2)
plt.plot(100 * dq, label="Double Q-learning", color="#4f46e5", lw=2)
plt.axhline(100 * optimo, ls="--", color="#4a5568", lw=1, label="optimo (eps/2 = 5%)")
plt.xlabel("Episodio")
plt.ylabel("% de acciones 'izquierda' desde A")
plt.title("Sesgo de maximizacion: Q-learning vs Double Q-learning")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
