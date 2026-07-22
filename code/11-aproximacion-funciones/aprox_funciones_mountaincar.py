"""
Aproximación de funciones: SARSA semi-gradiente con tile coding en Mountain Car.

Este script implementa DESDE CERO, usando solo numpy:
  1) La física del entorno clásico "Mountain Car" (posición y velocidad).
  2) Un tile coder (codificación por mosaicos) que convierte el estado
     continuo en un vector de características binarias disperso.
  3) Control SARSA SEMI-GRADIENTE con aproximación lineal de la función Q:
         q_hat(s, a) = w · x(s, a)
     y la actualización
         w <- w + (alpha / n_mosaicos) * [R + gamma * q_hat(s',a') - q_hat(s,a)] * grad_w q_hat
     donde grad_w q_hat es el propio vector de características (features binarias).

Al terminar dibuja dos gráficas:
  - La curva de aprendizaje (pasos por episodio), que debe DESCENDER.
  - La superficie de "coste por ir" -max_a q_hat(s,a) como mapa de calor.

Cómo ejecutarlo en tu terminal:
    pip install -r requirements.txt
    python code/11-aproximacion-funciones/aprox_funciones_mountaincar.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) El entorno Mountain Car, implementado a mano con numpy
# ---------------------------------------------------------------------------
class MountainCar:
    """Coche en un valle: hay que coger impulso para subir la colina derecha."""

    POS_MIN, POS_MAX = -1.2, 0.6
    VEL_MIN, VEL_MAX = -0.07, 0.07
    META = 0.5                      # posición objetivo (cima derecha)
    ACCIONES = np.array([-1, 0, 1])  # empujar izquierda / soltar / empujar derecha

    def __init__(self, rng):
        self.rng = rng
        self.pos = 0.0
        self.vel = 0.0

    def reset(self):
        # Arranca en reposo, cerca del fondo del valle.
        self.pos = self.rng.uniform(-0.6, -0.4)
        self.vel = 0.0
        return np.array([self.pos, self.vel])

    def step(self, accion):
        empuje = self.ACCIONES[accion]
        self.vel += 0.001 * empuje - 0.0025 * np.cos(3.0 * self.pos)
        self.vel = np.clip(self.vel, self.VEL_MIN, self.VEL_MAX)
        self.pos += self.vel
        self.pos = np.clip(self.pos, self.POS_MIN, self.POS_MAX)
        if self.pos == self.POS_MIN:      # choca con la pared izquierda
            self.vel = 0.0
        terminado = self.pos >= self.META
        recompensa = -1.0                 # cada paso cuesta: hay que llegar rápido
        return np.array([self.pos, self.vel]), recompensa, terminado


# ---------------------------------------------------------------------------
# 2) Tile coder: estado continuo -> índices de mosaicos activos
# ---------------------------------------------------------------------------
class TileCoder:
    """Codificación por mosaicos con desplazamientos asimétricos (1, 3, ...)."""

    def __init__(self, low, high, n_mosaicos=8, n_bins=8, n_acciones=3):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.n_mosaicos = n_mosaicos
        self.n_bins = n_bins
        self.n_acciones = n_acciones
        self.n_dims = len(self.low)
        # Ancho de un mosaico en cada dimensión.
        self.ancho = (self.high - self.low) / n_bins
        # Vector de desplazamiento asimétrico: (1, 3, 5, ...).
        self.disp = (2 * np.arange(self.n_dims) + 1).astype(float)
        # Mosaicos por rejilla: un bin extra para absorber los desplazamientos.
        self.por_mosaico = (n_bins + 1) ** self.n_dims
        self.bloque = n_mosaicos * self.por_mosaico   # índices por acción
        self.n_features = n_acciones * self.bloque    # tamaño del vector w

    def activos(self, estado):
        """Índices (sin acción) de los n_mosaicos mosaicos activos."""
        escala = (np.asarray(estado) - self.low) / self.ancho
        offs = np.outer(np.arange(self.n_mosaicos), self.disp) / self.n_mosaicos
        coords = np.floor(escala[None, :] + offs).astype(np.int64)
        coords = np.clip(coords, 0, self.n_bins)
        plano = np.zeros(self.n_mosaicos, dtype=np.int64)
        for d in range(self.n_dims):
            plano = plano * (self.n_bins + 1) + coords[:, d]
        return np.arange(self.n_mosaicos) * self.por_mosaico + plano

    def indices(self, estado, accion):
        """Índices en w de las features activas para el par (estado, acción)."""
        return accion * self.bloque + self.activos(estado)

    def q_todas(self, estado, w):
        """Valor q_hat(estado, a) para todas las acciones (vector)."""
        base = self.activos(estado)
        return np.array([w[a * self.bloque + base].sum()
                         for a in range(self.n_acciones)])


# ---------------------------------------------------------------------------
# 3) Política epsilon-greedy con desempate aleatorio
# ---------------------------------------------------------------------------
def eps_greedy(qs, epsilon, rng, n_acciones):
    if rng.random() < epsilon:
        return int(rng.integers(n_acciones))
    maximo = qs.max()
    candidatas = np.flatnonzero(qs == maximo)  # empates -> elección al azar
    return int(rng.choice(candidatas))


# ---------------------------------------------------------------------------
# 4) SARSA semi-gradiente
# ---------------------------------------------------------------------------
def entrenar(n_episodios=50, alpha=0.5, gamma=1.0, epsilon=0.0,
             max_pasos=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    env = MountainCar(rng)
    tc = TileCoder(low=[MountainCar.POS_MIN, MountainCar.VEL_MIN],
                   high=[MountainCar.POS_MAX, MountainCar.VEL_MAX],
                   n_mosaicos=8, n_bins=8, n_acciones=3)
    w = np.zeros(tc.n_features)           # inicio optimista: q_hat = 0 > retornos reales
    paso_alpha = alpha / tc.n_mosaicos
    pasos_por_ep = np.zeros(n_episodios, dtype=int)

    for ep in range(n_episodios):
        s = env.reset()
        a = eps_greedy(tc.q_todas(s, w), epsilon, rng, tc.n_acciones)
        for t in range(max_pasos):
            s2, r, terminado = env.step(a)
            idx = tc.indices(s, a)                 # features activas de (s, a)
            q_sa = w[idx].sum()
            if terminado:
                delta = r - q_sa
                w[idx] += paso_alpha * delta
                pasos_por_ep[ep] = t + 1
                break
            a2 = eps_greedy(tc.q_todas(s2, w), epsilon, rng, tc.n_acciones)
            q_s2a2 = tc.q_todas(s2, w)[a2]
            delta = r + gamma * q_s2a2 - q_sa      # error TD (bootstrapping)
            w[idx] += paso_alpha * delta
            s, a = s2, a2
        else:
            pasos_por_ep[ep] = max_pasos           # episodio truncado

    return w, tc, pasos_por_ep


# ---------------------------------------------------------------------------
# 5) Evaluación de la política greedy aprendida
# ---------------------------------------------------------------------------
def evaluar(w, tc, n=5, max_pasos=2000, semilla=100):
    rng = np.random.default_rng(semilla)
    env = MountainCar(rng)
    pasos = []
    for _ in range(n):
        s = env.reset()
        for t in range(max_pasos):
            a = int(np.argmax(tc.q_todas(s, w)))
            s, r, terminado = env.step(a)
            if terminado:
                pasos.append(t + 1)
                break
        else:
            pasos.append(max_pasos)
    return float(np.mean(pasos))


# ---------------------------------------------------------------------------
# 6) Gráficas: curva de aprendizaje y superficie de coste por ir
# ---------------------------------------------------------------------------
def graficar(pasos_por_ep, w, tc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) Curva de aprendizaje.
    episodios = np.arange(1, len(pasos_por_ep) + 1)
    ax1.plot(episodios, pasos_por_ep, color="#94a3b8", lw=1, label="pasos por episodio")
    if len(pasos_por_ep) >= 5:
        k = 5
        suave = np.convolve(pasos_por_ep, np.ones(k) / k, mode="valid")
        ax1.plot(episodios[k - 1:], suave, color="#4f46e5", lw=2.2, label="media móvil (5)")
    ax1.set_xlabel("episodio")
    ax1.set_ylabel("pasos hasta la meta")
    ax1.set_title("Curva de aprendizaje")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # (b) Coste por ir: -max_a q_hat(s, a) sobre una rejilla de estados.
    n = 60
    pos = np.linspace(MountainCar.POS_MIN, MountainCar.POS_MAX, n)
    vel = np.linspace(MountainCar.VEL_MIN, MountainCar.VEL_MAX, n)
    coste = np.zeros((n, n))
    for i, v in enumerate(vel):
        for j, p in enumerate(pos):
            coste[i, j] = -tc.q_todas(np.array([p, v]), w).max()
    im = ax2.imshow(coste, origin="lower", aspect="auto", cmap="viridis",
                    extent=[MountainCar.POS_MIN, MountainCar.POS_MAX,
                            MountainCar.VEL_MIN, MountainCar.VEL_MAX])
    ax2.set_xlabel("posición")
    ax2.set_ylabel("velocidad")
    ax2.set_title("Coste por ir  -max_a q̂(s, a)")
    fig.colorbar(im, ax=ax2, shrink=0.85)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    w, tc, pasos = entrenar()
    print(f"Episodio 1  : {pasos[0]:5d} pasos (aún explorando)")
    print(f"Último ep.  : {pasos[-1]:5d} pasos")
    print(f"Mejor ep.   : {pasos.min():5d} pasos")
    media = evaluar(w, tc)
    print(f"Política greedy final: {media:.1f} pasos de media en 5 arranques")
    graficar(pasos, w, tc)
