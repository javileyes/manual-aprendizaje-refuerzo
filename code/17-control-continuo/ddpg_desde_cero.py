"""
DDPG (Deep Deterministic Policy Gradient) DESDE CERO con numpy.

Implementa un DDPG mínimo, sin librerías de deep learning, sobre un entorno
continuo diminuto codificado a mano: un punto que debe "alcanzar" el objetivo
(el origen) en un plano 2D. La acción es continua: un vector de velocidad en
[-1, 1]^2. Como las acciones son infinitas, no podemos hacer max_a sobre ellas:
en su lugar, un ACTOR determinista mu(s) propone la acción y un CRÍTICO Q(s, a)
la evalúa.

Todo está hecho a mano con numpy:
  1) El entorno continuo (física trivial: p <- p + dt * a).
  2) Dos redes MLP (actor y crítico) con paso hacia delante y RETROPROPAGACIÓN
     manual, más un optimizador Adam artesanal.
  3) El bucle DDPG completo: buffer de repetición, redes objetivo con
     actualización suave (Polyak) y ruido gaussiano para explorar.

El gradiente de política determinista (teorema DPG) dice:
    grad_theta J = E[ grad_a Q(s, a)|_{a=mu(s)} * grad_theta mu(s) ]
Es decir: propagamos hacia atrás a través del crítico para saber "en qué
dirección mover la acción" y luego a través del actor para saber "cómo mover
sus pesos". Eso es exactamente lo que hace la función `actualiza`.

Al terminar dibuja:
  - El retorno por episodio (debe SUBIR: de vagar sin rumbo a ir al objetivo).
  - El campo vectorial del actor mu(s), que debería apuntar hacia el origen.

Cómo ejecutarlo en tu terminal:
    pip install -r requirements.txt
    python code/17-control-continuo/ddpg_desde_cero.py

Solo necesita numpy y matplotlib, así que también corre en el navegador (Pyodide).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) El entorno: alcanzar el origen en un plano 2D con acción continua
# ---------------------------------------------------------------------------
class Reacher2D:
    """Un punto en [-1,1]^2 debe llegar al origen. Acción = velocidad continua."""

    DT = 0.1            # cuánto avanza por paso a velocidad máxima
    MAX_PASOS = 30      # duración fija del episodio

    def __init__(self, rng):
        self.rng = rng
        self.p = np.zeros(2)
        self.t = 0

    def reset(self):
        # Arranca en un punto aleatorio de un anillo alrededor del objetivo.
        ang = self.rng.uniform(0.0, 2.0 * np.pi)
        rad = self.rng.uniform(0.5, 0.95)
        self.p = np.array([rad * np.cos(ang), rad * np.sin(ang)])
        self.t = 0
        return self.p.copy()

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)                 # la acción vive en [-1, 1]^2
        self.p = np.clip(self.p + self.DT * a, -1.0, 1.0)
        self.t += 1
        recompensa = -float(np.linalg.norm(self.p))   # premio = -distancia al origen
        terminado = self.t >= self.MAX_PASOS          # fin por límite de tiempo
        return self.p.copy(), recompensa, terminado


# ---------------------------------------------------------------------------
# 2) Un MLP con retropropagación y Adam, hechos a mano
# ---------------------------------------------------------------------------
class MLP:
    """Perceptrón multicapa con ocultas ReLU y salida 'tanh' o 'lineal'.

    Guarda las activaciones en el paso hacia delante para poder retropropagar.
    `backward` devuelve además el gradiente respecto de la ENTRADA, que es lo
    que DDPG necesita para encadenar crítico -> actor.
    """

    def __init__(self, tamanos, salida, rng):
        self.salida = salida               # 'tanh' (actor) o 'lineal' (crítico)
        self.W, self.b = [], []
        for i in range(len(tamanos) - 1):
            limite = 1.0 / np.sqrt(tamanos[i])
            self.W.append(rng.uniform(-limite, limite, (tamanos[i], tamanos[i + 1])))
            self.b.append(np.zeros(tamanos[i + 1]))
        # Estado de Adam (un momento por parámetro).
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.paso_t = 0

    def forward(self, x):
        self.a_cache = [x]                 # activaciones (a_0 = entrada)
        self.z_cache = []                  # preactivaciones
        h = x
        L = len(self.W)
        for i in range(L):
            z = h @ self.W[i] + self.b[i]
            self.z_cache.append(z)
            if i < L - 1:
                h = np.maximum(0.0, z)     # ReLU en las capas ocultas
            elif self.salida == "tanh":
                h = np.tanh(z)             # salida acotada en (-1, 1)
            else:
                h = z                      # salida lineal (valor Q)
            self.a_cache.append(h)
        return h

    def backward(self, grad_salida):
        """Retropropaga `grad_salida` (dL/d_salida). Rellena gW, gb y devuelve dL/d_entrada."""
        L = len(self.W)
        self.gW = [None] * L
        self.gb = [None] * L
        g = grad_salida
        for i in reversed(range(L)):
            z = self.z_cache[i]
            if i == L - 1:
                if self.salida == "tanh":
                    g = g * (1.0 - np.tanh(z) ** 2)   # derivada de tanh
                # salida lineal: derivada 1, g no cambia
            else:
                g = g * (z > 0.0)                      # derivada de ReLU
            self.gW[i] = self.a_cache[i].T @ g
            self.gb[i] = g.sum(axis=0)
            g = g @ self.W[i].T                        # gradiente hacia la capa previa
        return g                                       # dL/d_entrada

    def adam(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        """Un paso de descenso de Adam usando los gradientes de `backward`."""
        self.paso_t += 1
        corr1 = 1.0 - b1 ** self.paso_t
        corr2 = 1.0 - b2 ** self.paso_t
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * self.gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * self.gW[i] ** 2
            self.W[i] -= lr * (self.mW[i] / corr1) / (np.sqrt(self.vW[i] / corr2) + eps)
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * self.gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * self.gb[i] ** 2
            self.b[i] -= lr * (self.mb[i] / corr1) / (np.sqrt(self.vb[i] / corr2) + eps)


def copia_dura(destino, origen):
    """Copia todos los pesos de `origen` en `destino` (para inicializar objetivos)."""
    for i in range(len(destino.W)):
        destino.W[i] = origen.W[i].copy()
        destino.b[i] = origen.b[i].copy()


def actualiza_suave(destino, origen, tau):
    """Polyak: destino <- tau*origen + (1-tau)*destino (redes objetivo lentas)."""
    for i in range(len(destino.W)):
        destino.W[i] = tau * origen.W[i] + (1 - tau) * destino.W[i]
        destino.b[i] = tau * origen.b[i] + (1 - tau) * destino.b[i]


# ---------------------------------------------------------------------------
# 3) Buffer de repetición (experiencias pasadas para aprender off-policy)
# ---------------------------------------------------------------------------
class Buffer:
    def __init__(self, cap, dim_s, dim_a):
        self.s = np.zeros((cap, dim_s))
        self.a = np.zeros((cap, dim_a))
        self.r = np.zeros((cap, 1))
        self.s2 = np.zeros((cap, dim_s))
        self.cap, self.i, self.n = cap, 0, 0

    def add(self, s, a, r, s2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i] = s, a, r, s2
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx]


# ---------------------------------------------------------------------------
# 4) El corazón de DDPG: una actualización de crítico + actor
# ---------------------------------------------------------------------------
def actualiza(actor, critico, actor_obj, critico_obj, lote, gamma, lr_a, lr_c):
    s, a, r, s2 = lote
    dim_s = s.shape[1]
    batch = s.shape[0]

    # --- (a) Crítico: regresión hacia el objetivo TD y = r + gamma * Q'(s', mu'(s')) ---
    a2 = actor_obj.forward(s2)                         # acción objetivo
    q_obj = critico_obj.forward(np.hstack([s2, a2]))   # Q'(s', a2)
    y = r + gamma * q_obj                              # sin término terminal: tarea continua
    q = critico.forward(np.hstack([s, a]))             # Q(s, a) actual
    grad_q = 2.0 * (q - y) / batch                     # dL/dQ del ECM
    critico.backward(grad_q)
    critico.adam(lr_c)

    # --- (b) Actor: ascenso del gradiente de política determinista (DPG) ---
    # Loss_actor = -mean Q(s, mu(s)); minimizarla = maximizar Q.
    a_pred = actor.forward(s)                          # mu(s)
    critico.forward(np.hstack([s, a_pred]))            # Q(s, mu(s)) (nuevo forward)
    grad_in = critico.backward(-np.ones((batch, 1)) / batch)  # dLoss/d[s, a]
    grad_a = grad_in[:, dim_s:]                        # nos quedamos con dLoss/da
    actor.backward(grad_a)                             # ...y lo encadenamos al actor
    actor.adam(lr_a)


# ---------------------------------------------------------------------------
# 5) Bucle de entrenamiento
# ---------------------------------------------------------------------------
def entrena(n_episodios=60, gamma=0.95, tau=0.01, lr_a=1e-3, lr_c=2e-3,
            batch=64, oculta=32, inicio_aleatorio=300, semilla=0):
    rng = np.random.default_rng(semilla)
    env = Reacher2D(rng)
    dim_s, dim_a = 2, 2

    actor = MLP([dim_s, oculta, oculta, dim_a], "tanh", rng)
    critico = MLP([dim_s + dim_a, oculta, oculta, 1], "lineal", rng)
    actor_obj = MLP([dim_s, oculta, oculta, dim_a], "tanh", rng)
    critico_obj = MLP([dim_s + dim_a, oculta, oculta, 1], "lineal", rng)
    copia_dura(actor_obj, actor)
    copia_dura(critico_obj, critico)

    buffer = Buffer(20000, dim_s, dim_a)
    retornos = np.zeros(n_episodios)
    pasos_totales = 0

    for ep in range(n_episodios):
        s = env.reset()
        # El ruido de exploración decae: mucho al principio, poco al final.
        sigma = 0.30 - 0.25 * ep / max(1, n_episodios - 1)
        total = 0.0
        terminado = False
        while not terminado:
            if pasos_totales < inicio_aleatorio:
                a = rng.uniform(-1.0, 1.0, size=dim_a)     # arranque puramente aleatorio
            else:
                a = actor.forward(s[None, :])[0]           # mu(s) determinista
                a = a + rng.normal(0.0, sigma, size=dim_a)  # + ruido gaussiano
                a = np.clip(a, -1.0, 1.0)

            s2, r, terminado = env.step(a)
            buffer.add(s, a, r, s2)
            s = s2
            total += r
            pasos_totales += 1

            if buffer.n >= max(batch, 256):
                lote = buffer.sample(batch, rng)
                actualiza(actor, critico, actor_obj, critico_obj,
                          lote, gamma, lr_a, lr_c)
                actualiza_suave(actor_obj, actor, tau)
                actualiza_suave(critico_obj, critico, tau)

        retornos[ep] = total

    return retornos, actor


# ---------------------------------------------------------------------------
# 6) Evaluación de la política determinista (sin ruido)
# ---------------------------------------------------------------------------
def evaluar(actor, n=20, semilla=123):
    rng = np.random.default_rng(semilla)
    env = Reacher2D(rng)
    totales = []
    for _ in range(n):
        s = env.reset()
        total, terminado = 0.0, False
        while not terminado:
            a = actor.forward(s[None, :])[0]      # sin ruido: pura explotación
            s, r, terminado = env.step(a)
            total += r
        totales.append(total)
    return float(np.mean(totales))


# ---------------------------------------------------------------------------
# 7) Gráficas: retorno por episodio y campo vectorial del actor
# ---------------------------------------------------------------------------
def graficar(retornos, actor):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) Curva de aprendizaje.
    ep = np.arange(1, len(retornos) + 1)
    ax1.plot(ep, retornos, color="#94a3b8", lw=1, label="retorno por episodio")
    if len(retornos) >= 5:
        k = 5
        suave = np.convolve(retornos, np.ones(k) / k, mode="valid")
        ax1.plot(ep[k - 1:], suave, color="#4f46e5", lw=2.2, label="media móvil (5)")
    ax1.set_xlabel("episodio")
    ax1.set_ylabel("retorno (suma de -distancias)")
    ax1.set_title("Curva de aprendizaje de DDPG")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.25)

    # (b) Campo vectorial del actor: mu(s) debería apuntar al origen.
    g = np.linspace(-0.95, 0.95, 11)
    X, Y = np.meshgrid(g, g)
    estados = np.column_stack([X.ravel(), Y.ravel()])
    A = actor.forward(estados)
    U = A[:, 0].reshape(X.shape)
    V = A[:, 1].reshape(Y.shape)
    ax2.quiver(X, Y, U, V, color="#0ea5e9", pivot="mid")
    ax2.plot(0, 0, "o", color="#dc2626", ms=10, label="objetivo")
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Política aprendida  mu(s)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    retornos, actor = entrena()
    print(f"Retorno episodio 1   : {retornos[0]:7.2f}  (aún explorando)")
    print(f"Retorno último ep.   : {retornos[-1]:7.2f}")
    media_ini = retornos[:5].mean()
    media_fin = retornos[-5:].mean()
    print(f"Media primeros 5 ep. : {media_ini:7.2f}")
    print(f"Media últimos 5 ep.  : {media_fin:7.2f}")
    print(f"Política determinista: {evaluar(actor):7.2f}  (retorno medio, 20 arranques)")
    graficar(retornos, actor)
