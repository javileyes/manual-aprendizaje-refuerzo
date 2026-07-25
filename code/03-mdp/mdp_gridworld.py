"""
mdp_gridworld.py — Un Proceso de Decisión de Markov (MDP) explícito: GridWorld.

Este script define un mundo de rejilla 5x5 con muros y una meta, formalizado
como un MDP con la tupla (S, A, P, R, gamma). Muestra tres cosas:
  1) una trayectoria (s, a, r, s') muestreada bajo una política aleatoria,
  2) una estimación por Monte Carlo del retorno medio desde el estado inicial,
  3) una visualización del mapa con matplotlib (imshow), marcando inicio y meta.

La clase GridWorld es reutilizable: expone reset() y step(accion) al estilo de
los entornos clásicos de RL, y la reutilizaremos conceptualmente en capítulos
posteriores (funciones de valor, programación dinámica, Monte Carlo, TD...).

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/03-mdp/mdp_gridworld.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Acciones: desplazamiento (dfila, dcol) para N, S, E, O.
ACCIONES = {
    0: (-1, 0),   # N (Norte): subir una fila
    1: (1, 0),    # S (Sur):   bajar una fila
    2: (0, 1),    # E (Este):  avanzar una columna
    3: (0, -1),   # O (Oeste): retroceder una columna
}
NOMBRES = {0: "N", 1: "S", 2: "E", 3: "O"}

# Para cada acción, sus dos direcciones perpendiculares (por donde "resbala").
PERPENDICULARES = {
    0: (2, 3), 1: (2, 3),   # N y S resbalan hacia E / O
    2: (0, 1), 3: (0, 1),   # E y O resbalan hacia N / S
}


class GridWorld:
    """Mundo de rejilla como MDP episódico.

    Estados (S): enteros 0..(filas*cols - 1), con estado = fila*cols + columna.
    Acciones (A): 0=N, 1=S, 2=E, 3=O.
    Transición (P): con probabilidad (1 - slip) el agente va en la dirección
        elegida; con probabilidad slip resbala a una de las dos perpendiculares.
        Si el destino es un muro o cae fuera del tablero, se queda donde está.
    Recompensa (R): -1 por cada paso (para incentivar llegar pronto) y +10 al
        alcanzar la meta, que es un estado terminal (absorbente).
    """

    def __init__(self, slip=0.1, seed=0):
        # Mapa 5x5: 0 = casilla libre, 1 = muro. La meta se marca aparte.
        self.mapa = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        self.filas, self.cols = self.mapa.shape
        self.inicio = (0, 0)
        self.meta = (4, 4)
        self.slip = float(slip)
        self.n_estados = self.filas * self.cols
        self.n_acciones = 4
        self.rng = np.random.default_rng(seed)
        self.pos = self.inicio

    # --- conversión coordenadas <-> índice de estado ---
    def _a_estado(self, pos):
        fila, col = pos
        return fila * self.cols + col

    def es_muro(self, pos):
        fila, col = pos
        return self.mapa[fila, col] == 1

    def _dentro(self, pos):
        fila, col = pos
        return 0 <= fila < self.filas and 0 <= col < self.cols

    def _mover(self, pos, accion):
        """Aplica un desplazamiento; si choca con muro o borde, se queda."""
        dfila, dcol = ACCIONES[accion]
        destino = (pos[0] + dfila, pos[1] + dcol)
        if not self._dentro(destino) or self.es_muro(destino):
            return pos
        return destino

    def reset(self):
        """Coloca al agente en el inicio y devuelve el estado inicial."""
        self.pos = self.inicio
        return self._a_estado(self.pos)

    def step(self, accion):
        """Ejecuta una acción. Devuelve (nuevo_estado, recompensa, terminado)."""
        # ¿Resbala? Elegimos la dirección REAL del movimiento.
        if self.rng.random() < self.slip:
            accion = int(self.rng.choice(PERPENDICULARES[accion]))
        self.pos = self._mover(self.pos, accion)

        if self.pos == self.meta:
            return self._a_estado(self.pos), 10.0, True
        return self._a_estado(self.pos), -1.0, False


def dinamica(env, s, a):
    """La función P del MDP: devuelve p(s', r | s, a) como {(s', r): prob}.

    step() TIRA LOS DADOS; esta función DESCRIBE los dados. Recorre las
    direcciones reales posibles (la elegida con probabilidad 1-slip, cada
    perpendicular con slip/2), calcula a dónde llevan y con qué recompensa,
    y acumula la probabilidad de cada par (s', r).
    """
    pos = divmod(s, env.cols)
    if pos == env.meta:
        # Estado terminal absorbente: se queda donde está y no cobra nada más.
        return {(s, 0.0): 1.0}

    # Probabilidad de que la dirección REAL sea cada una.
    reales = {a: 1.0 - env.slip}
    for perp in PERPENDICULARES[a]:
        reales[perp] = reales.get(perp, 0.0) + env.slip / 2.0

    p = {}
    for accion_real, prob in reales.items():
        if prob == 0.0:      # con slip=0 las perpendiculares no llegan a ocurrir
            continue
        destino = env._mover(pos, accion_real)
        s2 = destino[0] * env.cols + destino[1]
        r = 10.0 if destino == env.meta else -1.0
        p[(s2, r)] = p.get((s2, r), 0.0) + prob
    return p


def mostrar_dinamica(env, s, a):
    """Imprime la tabla p(s', r | s, a) de un par (s, a) y comprueba que suma 1."""
    p = dinamica(env, s, a)
    for (s2, r), prob in p.items():
        print(f"  p(s'={s2:>2}, r={r:>3.0f} | s={s}, a={NOMBRES[a]}) = {prob:.2f}")
    print(f"  suma = {sum(p.values()):.2f}")


def politica_aleatoria(rng):
    """Política pi(a|s) uniforme: elige una de las 4 acciones al azar."""
    return int(rng.integers(4))


def muestrear_trayectoria(env, rng, max_pasos=30):
    """Genera una trayectoria e imprime cada transición (s, a, r, s')."""
    s = env.reset()
    for t in range(max_pasos):
        a = politica_aleatoria(rng)
        s2, r, done = env.step(a)
        print(f"t={t:<2} s={s:<2} a={NOMBRES[a]} r={r:>3.0f}  ->  s'={s2:<2}")
        s = s2
        if done:
            print(f"  ¡Meta alcanzada en {t + 1} pasos!")
            break
    else:
        print(f"  (sin llegar a la meta en {max_pasos} pasos)")


def estimar_retorno(env, rng, gamma=0.95, episodios=300, max_pasos=200):
    """Estima E[G_0] promediando el retorno descontado de muchos episodios.

    El retorno de un episodio es G_0 = sum_t gamma^t * r_{t+1}. Al promediar
    muchos episodios obtenemos una estimación Monte Carlo del valor esperado.
    """
    retornos = np.empty(episodios)
    for e in range(episodios):
        env.reset()
        G = 0.0
        descuento = 1.0
        for _ in range(max_pasos):
            a = politica_aleatoria(rng)
            _, r, done = env.step(a)
            G += descuento * r
            descuento *= gamma
            if done:
                break
        retornos[e] = G
    return retornos


def dibujar_mapa(env, retorno_medio=None):
    """Dibuja el tablero con imshow, marcando muros, inicio (S) y meta (G)."""
    disp = env.mapa.astype(float).copy()
    disp[env.inicio] = 2.0   # inicio
    disp[env.meta] = 3.0     # meta
    cmap = ListedColormap(["#eef1f5", "#334155", "#0ea5e9", "#10b981"])

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=3)

    # índice de estado en cada casilla transitable
    for f in range(env.filas):
        for c in range(env.cols):
            if env.mapa[f, c] == 1:
                continue
            s = f * env.cols + c
            ax.text(c, f - 0.26, str(s), ha="center", va="center",
                    fontsize=8, color="#64748b")
    ax.text(env.inicio[1], env.inicio[0] + 0.12, "S", ha="center",
            va="center", fontsize=16, color="white", fontweight="bold")
    ax.text(env.meta[1], env.meta[0] + 0.12, "G", ha="center",
            va="center", fontsize=16, color="white", fontweight="bold")

    # rejilla blanca entre casillas
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.filas, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])

    titulo = "GridWorld 5x5 como MDP"
    if retorno_medio is not None:
        titulo += f"  —  retorno medio ≈ {retorno_medio:.1f}"
    ax.set_title(titulo, fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = GridWorld(slip=0.1, seed=0)
    rng = np.random.default_rng(0)   # semilla fija: resultados reproducibles

    print("=== La dinámica p(s', r | s, a), con números ===")
    print("Desde la casilla 0, hacia el Este:")
    mostrar_dinamica(env, 0, 2)
    print("Desde la casilla 23, hacia el Este (la meta está al lado):")
    mostrar_dinamica(env, 23, 2)

    print("\n=== Una trayectoria bajo política aleatoria pi(a|s) uniforme ===")
    muestrear_trayectoria(env, rng, max_pasos=30)

    print("\n=== Retorno medio por Monte Carlo (política aleatoria) ===")
    gamma = 0.95
    retornos = estimar_retorno(env, rng, gamma=gamma, episodios=300, max_pasos=200)
    print(f"gamma = {gamma}   episodios = {len(retornos)}")
    print(f"Retorno medio G_0 ≈ {retornos.mean():.2f}"
          f"   (desviación {retornos.std():.2f})")

    dibujar_mapa(env, retorno_medio=retornos.mean())
