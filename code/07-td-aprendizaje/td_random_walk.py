"""
TD(0) frente a Monte Carlo en el paseo aleatorio de 5 estados
=============================================================

Reproduce el ejemplo 6.2 de Sutton & Barto ("Reinforcement Learning: An
Introduction"). Un agente parte del estado central C de un pasillo

    [ 0 ] A   B   C   D   E [ 6 ]
      T                       T

con dos estados terminales (T) en los extremos. En cada paso se mueve a la
izquierda o a la derecha con probabilidad 1/2. Todas las recompensas son 0
salvo la de llegar al terminal derecho, que vale +1. Con gamma = 1, el valor
verdadero de cada estado es su probabilidad de acabar en el terminal derecho:

    V(A)=1/6, V(B)=2/6, V(C)=3/6, V(D)=4/6, V(E)=5/6.

Comparamos dos formas de ESTIMAR esos valores a partir de la experiencia:

  * TD(0): actualiza en cada paso usando su propia prediccion del estado
    siguiente (bootstrapping):
        V(S_t) <- V(S_t) + alpha * [ R_{t+1} + gamma * V(S_{t+1}) - V(S_t) ].
  * Monte Carlo de cada-visita: espera al final del episodio y actualiza cada
    estado visitado hacia el retorno observado:
        V(S_t) <- V(S_t) + alpha * [ G_t - V(S_t) ].

El script dibuja (1) los valores que estima TD tras 0, 1, 10 y 100 episodios
frente a los verdaderos y (2) la curva de error RMS frente al numero de
episodios para varios alpha, tanto para TD como para MC.

Ejecucion:
    pip install -r requirements.txt
    python code/07-td-aprendizaje/td_random_walk.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Definicion del entorno ---
N_STATES = 7                 # estados 0..6; 0 y 6 son terminales
START = 3                    # el estado central C
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6
GAMMA = 1.0                  # sin descuento

# Valores verdaderos de A, B, C, D, E  ->  i/6
TRUE_VALUES = np.arange(1, 6) / 6.0


def generar_episodio(rng):
    """Genera un episodio y devuelve la lista de (estado, recompensa, siguiente)."""
    estado = START
    transiciones = []
    while estado not in (LEFT_TERMINAL, RIGHT_TERMINAL):
        siguiente = estado + (1 if rng.random() < 0.5 else -1)
        recompensa = 1.0 if siguiente == RIGHT_TERMINAL else 0.0
        transiciones.append((estado, recompensa, siguiente))
        estado = siguiente
    return transiciones


def actualizar_td0(V, alpha, rng):
    """Un episodio de TD(0): actualiza V en cada paso con bootstrapping."""
    for s, r, s_sig in generar_episodio(rng):
        delta = r + GAMMA * V[s_sig] - V[s]      # error TD
        V[s] += alpha * delta


def actualizar_mc(V, alpha, rng):
    """Un episodio de Monte Carlo de cada-visita (alpha constante)."""
    transiciones = generar_episodio(rng)
    # Con gamma=1 y recompensas nulas salvo la final, G_t es la misma para todo
    # t del episodio: la recompensa terminal (0 o 1).
    G = transiciones[-1][1]
    for s, r, s_sig in transiciones:
        V[s] += alpha * (G - V[s])


def error_rms(V):
    """Raiz del error cuadratico medio sobre los 5 estados no terminales."""
    return np.sqrt(np.mean((V[1:6] - TRUE_VALUES) ** 2))


def valores_iniciales():
    """V=0 en los terminales y V=0.5 en A..E (estimacion neutra)."""
    V = np.zeros(N_STATES)
    V[1:6] = 0.5
    return V


def instantaneas_td(snapshots, alpha, seed):
    """Valores que estima TD(0) tras un numero dado de episodios (una tirada)."""
    rng = np.random.default_rng(seed)
    V = valores_iniciales()
    resultado = {}
    if 0 in snapshots:
        resultado[0] = V[1:6].copy()
    for ep in range(1, max(snapshots) + 1):
        actualizar_td0(V, alpha, rng)
        if ep in snapshots:
            resultado[ep] = V[1:6].copy()
    return resultado


def curva_rms(actualizar, alpha, n_runs, n_episodes, seed0):
    """Error RMS medio (sobre n_runs tiradas) tras cada episodio."""
    errores = np.zeros(n_episodes)
    for run in range(n_runs):
        rng = np.random.default_rng(seed0 + run)
        V = valores_iniciales()
        for ep in range(n_episodes):
            actualizar(V, alpha, rng)
            errores[ep] += error_rms(V)
    return errores / n_runs


def main():
    etiquetas = ["A", "B", "C", "D", "E"]

    # --- (1) Valores estimados por TD(0) en varias fotos ---
    snapshots = [0, 1, 10, 100]
    fotos = instantaneas_td(snapshots, alpha=0.1, seed=0)

    print("Valores estimados por TD(0) (alpha=0.1) frente a los verdaderos:")
    print("   estado   " + "".join(f"{e:>7s}" for e in etiquetas))
    print("   verdad   " + "".join(f"{v:7.3f}" for v in TRUE_VALUES))
    for ep in snapshots:
        print(f"   ep={ep:<5d} " + "".join(f"{v:7.3f}" for v in fotos[ep]))

    # --- (2) Curvas de error RMS ---
    n_runs, n_episodes = 100, 100
    td_alphas = [0.15, 0.10, 0.05]
    mc_alphas = [0.04, 0.02, 0.01]

    curvas_td = {a: curva_rms(actualizar_td0, a, n_runs, n_episodes, seed0=1000)
                 for a in td_alphas}
    curvas_mc = {a: curva_rms(actualizar_mc, a, n_runs, n_episodes, seed0=2000)
                 for a in mc_alphas}

    print(f"\nError RMS final (media de {n_runs} tiradas, {n_episodes} episodios):")
    for a in td_alphas:
        print(f"   TD(0)  alpha={a:.2f}  ->  RMS = {curvas_td[a][-1]:.4f}")
    for a in mc_alphas:
        print(f"   MC     alpha={a:.2f}  ->  RMS = {curvas_mc[a][-1]:.4f}")

    # --- Grafica ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    x = np.arange(1, 6)
    ax1.plot(x, TRUE_VALUES, "k--", marker="o", label="verdadero")
    for ep in snapshots:
        ax1.plot(x, fotos[ep], marker="o", label=f"{ep} episodios")
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas)
    ax1.set_xlabel("estado")
    ax1.set_ylabel("valor estimado")
    ax1.set_title("TD(0): los valores se acercan a la verdad")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    epis = np.arange(1, n_episodes + 1)
    for a in td_alphas:
        ax2.plot(epis, curvas_td[a], color="tab:blue",
                 alpha=0.5 + 0.4 * (a / max(td_alphas)),
                 label=f"TD  alpha={a:.2f}")
    for a in mc_alphas:
        ax2.plot(epis, curvas_mc[a], color="tab:red", linestyle="--",
                 alpha=0.5 + 0.4 * (a / max(mc_alphas)),
                 label=f"MC  alpha={a:.2f}")
    ax2.set_xlabel("episodios")
    ax2.set_ylabel("error RMS (medio)")
    ax2.set_title("TD (azul) converge mas rapido y estable que MC (rojo)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
