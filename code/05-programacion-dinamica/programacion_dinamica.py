"""
Programación dinámica en un gridworld: resolver el MDP EXACTAMENTE cuando
conocemos el modelo (las probabilidades de transición P y las recompensas R).

Implementa los tres algoritmos clásicos del capítulo:
  - evaluar_politica     : evaluación iterativa de política (Bellman de esperanza)
  - iteracion_de_politica: evaluar + mejorar en alternancia (GPI)
  - iteracion_de_valor   : un barrido con el máximo de Bellman de optimalidad

El entorno es un gridworld de 3x4 con una META (+10), una TRAMPA (-10) y un
MURO intransitable. Cada movimiento cuesta -1 (para llegar rápido). Al final se
comprueba que iteración de política e iteración de valor devuelven la MISMA
política óptima y se dibuja V* como mapa de calor con las flechas de la política.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/05-programacion-dinamica/programacion_dinamica.py

Solo usa numpy y matplotlib, así que también corre en el navegador (Pyodide).
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Definición del gridworld -------------------------------------------------
FILAS, COLS = 3, 4
META = (0, 3)         # esquina superior derecha: recompensa +10 (terminal)
TRAMPA = (1, 3)       # justo debajo de la meta: recompensa -10 (terminal)
MUROS = {(1, 1)}      # celdas intransitables
R_META = 10.0
R_TRAMPA = -10.0
R_PASO = -1.0         # coste de vivir: cada movimiento resta 1
GAMMA = 0.9           # factor de descuento

# Acciones como desplazamientos (fila, columna): arriba, derecha, abajo, izquierda
ACCIONES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
FLECHAS = ["arriba", "derecha", "abajo", "izquierda"]
SIMBOLOS = ["^", ">", "v", "<"]


def construir_modelo():
    """Construye el modelo del MDP: listas de estados, P[s,a,s'] y R[s,a].

    Como el gridworld es determinista, P[s,a] es un vector one-hot (todo el peso
    en el estado de destino). Devolvemos también qué estados son terminales.
    """
    estados = [(r, c) for r in range(FILAS) for c in range(COLS) if (r, c) not in MUROS]
    idx = {s: i for i, s in enumerate(estados)}
    nS, nA = len(estados), len(ACCIONES)
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    terminal = np.zeros(nS, dtype=bool)

    for s in estados:
        i = idx[s]
        if s == META or s == TRAMPA:
            # Estado absorbente: se queda en sí mismo con recompensa 0.
            terminal[i] = True
            P[i, :, i] = 1.0
            R[i, :] = 0.0
            continue
        for a, (dr, dc) in enumerate(ACCIONES):
            nr, nc = s[0] + dr, s[1] + dc
            destino = (nr, nc)
            # Si se sale del tablero o choca con un muro, se queda donde está.
            if not (0 <= nr < FILAS and 0 <= nc < COLS) or destino in MUROS:
                destino = s
            j = idx[destino]
            P[i, a, j] = 1.0
            # La recompensa depende de en qué celda se entra.
            if destino == META:
                R[i, a] = R_META
            elif destino == TRAMPA:
                R[i, a] = R_TRAMPA
            else:
                R[i, a] = R_PASO
    return estados, idx, P, R, terminal


def evaluar_politica(politica, P, R, gamma, theta=1e-8):
    """Evaluación iterativa de política: barre la ecuación de Bellman de
    esperanza hasta que V deja de cambiar. `politica` es un vector de índices
    de acción (uno por estado). Devuelve V^pi.
    """
    nS = len(politica)
    V = np.zeros(nS)
    while True:
        delta = 0.0
        for s in range(nS):
            a = politica[s]
            v_nuevo = R[s, a] + gamma * P[s, a] @ V   # actualización in situ
            delta = max(delta, abs(v_nuevo - V[s]))
            V[s] = v_nuevo
        if delta < theta:
            break
    return V


def valores_de_accion(V, P, R, gamma):
    """Calcula Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') V(s') para todo (s,a).

    P tiene forma (nS, nA, nS) y V forma (nS,); P @ V da forma (nS, nA).
    """
    return R + gamma * (P @ V)


def mejorar_politica(V, P, R, gamma):
    """Política avariciosa (greedy) respecto a V: en cada estado elige la acción
    con mayor Q. Devuelve la nueva política y la tabla Q.
    """
    Q = valores_de_accion(V, P, R, gamma)
    return np.argmax(Q, axis=1), Q


def iteracion_de_politica(P, R, gamma, politica0=None):
    """Iteración de política (GPI): evaluar V^pi y luego volverse greedy,
    repetido hasta que la política ya no cambia. Devuelve (politica, V, iteraciones).
    """
    nS, nA = R.shape
    politica = np.zeros(nS, dtype=int) if politica0 is None else politica0.copy()
    iteraciones = 0
    while True:
        iteraciones += 1
        V = evaluar_politica(politica, P, R, gamma)
        nueva, _ = mejorar_politica(V, P, R, gamma)
        if np.array_equal(nueva, politica):
            return politica, V, iteraciones
        politica = nueva


def iteracion_de_valor(P, R, gamma, theta=1e-8):
    """Iteración de valor: un único barrido por estado usando el MÁXIMO de la
    ecuación de Bellman de optimalidad, repetido hasta converger. Al final,
    extrae la política greedy. Devuelve (politica, V, barridos).
    """
    nS, nA = R.shape
    V = np.zeros(nS)
    barridos = 0
    while True:
        barridos += 1
        Q = valores_de_accion(V, P, R, gamma)
        V_nuevo = np.max(Q, axis=1)
        delta = np.max(np.abs(V_nuevo - V))
        V = V_nuevo
        if delta < theta:
            break
    politica = np.argmax(valores_de_accion(V, P, R, gamma), axis=1)
    return politica, V, barridos


def politica_a_texto(idx, politica, terminal):
    """Dibuja la política como un tablero ASCII: flechas, G (meta), X (trampa)
    y # (muro).
    """
    lineas = []
    for r in range(FILAS):
        fila = []
        for c in range(COLS):
            if (r, c) in MUROS:
                fila.append(" # ")
            elif (r, c) == META:
                fila.append(" G ")
            elif (r, c) == TRAMPA:
                fila.append(" X ")
            else:
                i = idx[(r, c)]
                fila.append(" " + SIMBOLOS[politica[i]] + " ")
        lineas.append("".join(fila))
    return "\n".join(lineas)


def dibujar(idx, V, politica, terminal):
    """Mapa de calor de V* con la política óptima como flechas."""
    Vgrid = np.full((FILAS, COLS), np.nan)
    for (r, c), i in idx.items():
        Vgrid[r, c] = V[i]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(Vgrid, cmap="RdYlGn", origin="upper")
    fig.colorbar(im, ax=ax, label="V*(s)")

    for (r, c), i in idx.items():
        # Número de valor en la parte alta de la celda.
        ax.text(c, r - 0.28, f"{V[i]:.1f}", ha="center", va="center",
                fontsize=9, color="black")
        if terminal[i]:
            etiqueta = "META\n+10" if (r, c) == META else "TRAMPA\n-10"
            ax.text(c, r + 0.18, etiqueta, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black")
        else:
            dr, dc = ACCIONES[politica[i]]
            ax.arrow(c, r + 0.12, dc * 0.28, dr * 0.28, head_width=0.13,
                     head_length=0.11, fc="black", ec="black", length_includes_head=True)

    # Dibuja los muros como celdas grises.
    for (r, c) in MUROS:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#555555"))
        ax.text(c, r, "MURO", ha="center", va="center", color="white", fontsize=8)

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(FILAS))
    ax.set_xlabel("columna")
    ax.set_ylabel("fila")
    ax.set_title("V* (color y número) y política óptima (flechas)")
    plt.tight_layout()
    plt.show()


# --- Programa principal: resolvemos el gridworld de las dos formas ------------
# (Sin bloque `if __name__ == "__main__"` a propósito, para que este mismo código
#  se ejecute tal cual en el navegador con Pyodide, que lo corre en un espacio de
#  nombres nuevo donde __name__ no vale "__main__".)
estados, idx, P, R, terminal = construir_modelo()
n_no_term = int((~terminal).sum())
print(f"Gridworld {FILAS}x{COLS}: {len(estados)} estados "
      f"({n_no_term} no terminales), {len(ACCIONES)} acciones, gamma = {GAMMA}")

# (1) EVALUACIÓN DE POLÍTICA sobre una política fija y mediocre: "ir siempre
# a la derecha". Se queda atascada contra la pared este y acumula costes.
pol_derecha = np.full(len(estados), 1, dtype=int)  # 1 = derecha
V_derecha = evaluar_politica(pol_derecha, P, R, GAMMA)
inicio = (2, 0)
print(f"\nEvaluación de 'siempre derecha': V del inicio {inicio} = "
      f"{V_derecha[idx[inicio]]:.2f}  (se atasca y suma -1 sin fin)")

# (2) ITERACIÓN DE POLÍTICA, arrancando desde una política ALEATORIA para
# ver que converge igual (semilla fija para reproducibilidad).
rng = np.random.default_rng(0)
pol_inicial = rng.integers(0, len(ACCIONES), size=len(estados))
pol_pi, V_pi, iters_pi = iteracion_de_politica(P, R, GAMMA, pol_inicial)
print(f"\nIteración de política: convergió en {iters_pi} iteraciones "
      f"(evaluar + mejorar).")
print(politica_a_texto(idx, pol_pi, terminal))

# (3) ITERACIÓN DE VALOR.
pol_vi, V_vi, barridos_vi = iteracion_de_valor(P, R, GAMMA)
print(f"\nIteración de valor: convergió en {barridos_vi} barridos.")
print(politica_a_texto(idx, pol_vi, terminal))

# Verificación: ¿misma política óptima? ¿mismo V*?
no_term = ~terminal
misma = np.array_equal(pol_pi[no_term], pol_vi[no_term])
print(f"\n¿Ambas dan la misma política? {'Sí' if misma else 'No'}")
print(f"Diferencia máxima entre las dos V*: {np.max(np.abs(V_pi - V_vi)):.2e}")

dibujar(idx, V_vi, pol_vi, terminal)
