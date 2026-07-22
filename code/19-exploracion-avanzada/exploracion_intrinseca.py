"""
Exploración avanzada: motivación intrínseca por conteo en un gridworld difícil.

Este script implementa DESDE CERO, usando solo numpy, un problema de exploración
"dura": una rejilla larga y estrecha donde la ÚNICA recompensa está en la esquina
más lejana. Con recompensas tan escasas, actuar al azar (epsilon-greedy) casi
nunca tropieza con la meta.

Comparamos dos agentes de Q-learning que solo se diferencian en UNA cosa:

  1) epsilon-greedy "a secas": explora dando pasos aleatorios de vez en cuando.
  2) epsilon-greedy + BONO POR CONTEO: a la recompensa real le suma un premio
     intrínseco por novedad,
         r+  =  r  +  beta / sqrt(N(s, a))
     donde N(s, a) es cuántas veces se ha tomado la acción a en el estado s.
     Los pares (estado, acción) poco visitados valen más -> el agente es empujado
     sistemáticamente hacia lo desconocido ("ve donde no has ido").

Al terminar imprime cuándo encuentra la meta cada agente y dibuja:
  - La curva de éxitos ACUMULADOS por episodio (el bono despega mucho antes).
  - Dos mapas de calor de visitas: el bono cubre toda la rejilla; epsilon-greedy
    se queda dando vueltas cerca de la salida.

Cómo ejecutarlo en tu terminal:
    pip install -r requirements.txt
    python code/19-exploracion-avanzada/exploracion_intrinseca.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) El entorno: una rejilla larga con recompensa SOLO en la esquina lejana
# ---------------------------------------------------------------------------
# Acciones: 0=arriba, 1=abajo, 2=izquierda, 3=derecha (desplazamiento fila,col).
MOVIMIENTOS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
N_ACCIONES = 4


class GridWorldDisperso:
    """Salida en la esquina (0,0); meta en la esquina opuesta. Recompensa escasa."""

    def __init__(self, filas=2, columnas=30, recompensa_meta=20.0):
        self.filas = filas
        self.columnas = columnas
        self.recompensa_meta = recompensa_meta
        self.salida = (0, 0)
        self.meta = (filas - 1, columnas - 1)

    def paso(self, fila, col, accion):
        """Devuelve (fila2, col2, recompensa, terminado)."""
        df, dc = MOVIMIENTOS[accion]
        fila2 = min(max(fila + df, 0), self.filas - 1)   # las paredes frenan
        col2 = min(max(col + dc, 0), self.columnas - 1)
        if (fila2, col2) == self.meta:
            return fila2, col2, self.recompensa_meta, True
        return fila2, col2, 0.0, False                   # recompensa dispersa: 0


# ---------------------------------------------------------------------------
# 2) Política epsilon-greedy con desempate aleatorio
# ---------------------------------------------------------------------------
def eps_greedy(qs, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACCIONES))
    maximo = qs.max()
    candidatas = np.flatnonzero(qs == maximo)   # empates -> elección al azar
    return int(rng.choice(candidatas))


# ---------------------------------------------------------------------------
# 3) Q-learning, con o sin bono de exploración por conteo
# ---------------------------------------------------------------------------
def entrenar(usar_bono, n_episodios=250, max_pasos=250,
             alpha=0.5, gamma=0.95, epsilon=0.1, beta=0.1, semilla=0):
    rng = np.random.default_rng(semilla)
    env = GridWorldDisperso()
    F, C = env.filas, env.columnas

    Q = np.zeros((F, C, N_ACCIONES))     # valores acción-estado
    N = np.zeros((F, C, N_ACCIONES))     # conteos N(s, a) para el bono
    visitas = np.zeros((F, C))           # mapa de estados visitados
    exitos = np.zeros(n_episodios, dtype=int)   # 1 si el episodio alcanzó la meta

    for ep in range(n_episodios):
        f, c = env.salida
        for _ in range(max_pasos):
            visitas[f, c] += 1
            a = eps_greedy(Q[f, c], epsilon, rng)
            f2, c2, r, terminado = env.paso(f, c, a)

            if usar_bono:
                N[f, c, a] += 1
                bono = beta / np.sqrt(N[f, c, a])   # 1/sqrt(N): decae con la visita
            else:
                bono = 0.0

            r_aug = r + bono                        # recompensa aumentada r+
            futuro = 0.0 if terminado else gamma * Q[f2, c2].max()
            objetivo = r_aug + futuro
            Q[f, c, a] += alpha * (objetivo - Q[f, c, a])   # actualización Q-learning

            f, c = f2, c2
            if terminado:
                exitos[ep] = 1
                break

    return Q, N, visitas, exitos


# ---------------------------------------------------------------------------
# 4) Utilidad: primer episodio en el que se alcanza la meta
# ---------------------------------------------------------------------------
def primer_exito(exitos):
    idx = np.flatnonzero(exitos)
    return int(idx[0]) + 1 if idx.size else None


# ---------------------------------------------------------------------------
# 5) Gráficas: éxitos acumulados y mapas de visitas
# ---------------------------------------------------------------------------
def graficar(env, exitos_eps, visitas_eps, exitos_bono, visitas_bono):
    fig, axd = plt.subplot_mosaic(
        [["curva", "curva"], ["heat_eps", "heat_bono"]],
        figsize=(11, 6.6), gridspec_kw={"height_ratios": [1.15, 1]})

    # (a) Éxitos acumulados por episodio.
    ep = np.arange(1, len(exitos_eps) + 1)
    axd["curva"].plot(ep, np.cumsum(exitos_eps), color="#e11d48", lw=2.2,
                      label="ε-greedy (sin bono)")
    axd["curva"].plot(ep, np.cumsum(exitos_bono), color="#4f46e5", lw=2.2,
                      label="ε-greedy + bono por conteo")
    axd["curva"].set_xlabel("episodio")
    axd["curva"].set_ylabel("episodios con éxito (acumulado)")
    axd["curva"].set_title("¿Cuántas veces se ha encontrado ya la meta?")
    axd["curva"].legend(loc="upper left")
    axd["curva"].grid(alpha=0.25)

    # (b) y (c) Mapas de calor de visitas (raíz para comprimir el rango).
    vmax = np.sqrt(max(visitas_eps.max(), visitas_bono.max()))
    for clave, visitas, titulo in [
            ("heat_eps", visitas_eps, "Visitas · ε-greedy"),
            ("heat_bono", visitas_bono, "Visitas · con bono")]:
        ax = axd[clave]
        im = ax.imshow(np.sqrt(visitas), origin="upper", aspect="auto",
                       cmap="magma", vmin=0, vmax=vmax)
        ax.scatter([env.salida[1]], [env.salida[0]], marker="o",
                   s=60, edgecolor="white", facecolor="#22c55e", label="salida")
        ax.scatter([env.meta[1]], [env.meta[0]], marker="*",
                   s=160, edgecolor="white", facecolor="#facc15", label="meta")
        ax.set_title(titulo)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85, label="√(nº de visitas)")

    axd["heat_eps"].legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N_EP = 250
    Q_eps, _, vis_eps, ex_eps = entrenar(usar_bono=False, n_episodios=N_EP, semilla=0)
    Q_bono, _, vis_bono, ex_bono = entrenar(usar_bono=True, n_episodios=N_EP, semilla=0)

    env = GridWorldDisperso()
    print(f"Rejilla {env.filas}x{env.columnas}, meta en {env.meta}, "
          f"recompensa solo en la meta.\n")

    pe_eps = primer_exito(ex_eps)
    pe_bono = primer_exito(ex_bono)
    print("Q-learning ε-greedy (sin bono):")
    print(f"  primera vez que alcanza la meta : "
          f"{pe_eps if pe_eps else 'nunca'} de {N_EP} episodios")
    print(f"  episodios con éxito             : {ex_eps.sum()} / {N_EP}")
    print("Q-learning ε-greedy + bono por conteo (1/sqrt(N)):")
    print(f"  primera vez que alcanza la meta : "
          f"{pe_bono if pe_bono else 'nunca'} de {N_EP} episodios")
    print(f"  episodios con éxito             : {ex_bono.sum()} / {N_EP}")

    graficar(env, ex_eps, vis_eps, ex_bono, vis_bono)
