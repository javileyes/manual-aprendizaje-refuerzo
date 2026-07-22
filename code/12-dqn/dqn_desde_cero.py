"""
DQN desde cero, solo con NumPy (version didactica).
=====================================================

Este script implementa un Deep Q-Network COMPLETO sin usar ninguna libreria de
aprendizaje profundo: la red neuronal (un MLP de una capa oculta), la
retropropagacion (backprop manual), el buffer de repeticion de experiencia
(replay) y la red objetivo (target network) estan escritos a mano con NumPy.

El objetivo es que veas, sin cajas negras, las DOS piezas que hacen estable a
DQN frente a un simple Q-learning con red neuronal:
  1) el replay buffer, que rompe la correlacion entre transiciones consecutivas;
  2) la red objetivo congelada, que evita "perseguir nuestra propia cola".

El entorno es un mundo continuo 2D barato de simular: un agente que parte de la
zona inferior-izquierda de un cuadrado [0,1]x[0,1] y debe llegar a una meta en la
esquina superior-derecha. Recibe -0.05 por paso y +1 al alcanzar la meta, asi que
aprender = llegar en los menos pasos posibles.

Ejecutar:
    pip install -r requirements.txt
    python code/12-dqn/dqn_desde_cero.py

Es ligero: unos 200 episodios que terminan en 1-2 segundos y dibujan la
recompensa por episodio. Con la semilla fijada el resultado es reproducible.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Semilla unica: reproducibilidad de entorno, exploracion e inicializacion.
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# 1) EL ENTORNO: un mundo continuo 2D (barato de simular)
# ----------------------------------------------------------------------------
class MundoContinuo:
    """Agente en [0,1]x[0,1] que debe alcanzar una meta moviendose en 4 direcciones.

    step() devuelve (estado, recompensa, terminado, truncado):
      - terminado = True solo cuando se alcanza la meta (fin "real" del episodio);
      - truncado  = True cuando se agota el limite de pasos (fin por tiempo).
    La distincion importa para el objetivo de DQN: al bootstrapear, solo ponemos
    a cero el valor futuro en un final REAL, no cuando cortamos por tiempo.
    """

    def __init__(self):
        self.meta = np.array([0.9, 0.9])
        self.radio = 0.12                 # se considera "meta" si te acercas tanto
        self.dmov = 0.05                  # cuanto avanzas por paso
        self.max_pasos = 100
        # 4 acciones: +x, -x, +y, -y
        self.moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float) * self.dmov
        self.n_acciones = 4
        self.dim_estado = 2

    def reset(self):
        self.pos = rng.uniform(0.0, 0.5, size=2)   # arranca abajo-izquierda (aleatorio)
        self.t = 0
        return self.pos.copy()

    def step(self, a):
        self.pos = np.clip(self.pos + self.moves[a], 0.0, 1.0)
        self.t += 1
        dist = np.hypot(*(self.pos - self.meta))
        if dist < self.radio:
            return self.pos.copy(), 1.0, True, False     # meta: fin real
        if self.t >= self.max_pasos:
            return self.pos.copy(), -0.05, False, True    # se acabo el tiempo
        return self.pos.copy(), -0.05, False, False       # un paso mas cuesta -0.05


# ----------------------------------------------------------------------------
# 2) LA RED Q: un MLP de 2 capas (entrada -> oculta ReLU -> salida lineal)
#    Estado (2) -> oculta (H) -> Q de las 4 acciones (4). Todo a mano.
# ----------------------------------------------------------------------------
def crear_red(din, h, dout):
    """Inicializacion He para las capas ReLU; sesgos a cero."""
    return {
        "W1": rng.normal(0, np.sqrt(2 / din), (din, h)),
        "b1": np.zeros(h),
        "W2": rng.normal(0, np.sqrt(2 / h), (h, dout)),
        "b2": np.zeros(dout),
    }


def forward(red, S):
    """Paso hacia delante para un lote S de forma (B, din).

    Devuelve Q de forma (B, dout) y la cache (Z1, A1) que necesita el backprop.
    """
    Z1 = S @ red["W1"] + red["b1"]     # pre-activacion de la capa oculta
    A1 = np.maximum(Z1, 0.0)           # ReLU
    Q = A1 @ red["W2"] + red["b2"]     # salida lineal: un valor Q por accion
    return Q, (Z1, A1)


class Adam:
    """Optimizador Adam minimo (el DQN original usaba RMSProp; Adam va igual de bien)."""

    def __init__(self, red, lr):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, 0.9, 0.999, 1e-8, 0
        self.m = {k: np.zeros_like(v) for k, v in red.items()}
        self.v = {k: np.zeros_like(v) for k, v in red.items()}

    def paso(self, red, grads):
        self.t += 1
        for k in red:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            red[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ----------------------------------------------------------------------------
# 3) HIPERPARAMETROS
# ----------------------------------------------------------------------------
GAMMA = 0.95        # factor de descuento
H = 64              # neuronas de la capa oculta
LR = 1e-3           # tasa de aprendizaje
BATCH = 32          # tamano del minibatch que sacamos del replay
CAP = 5000          # capacidad del replay buffer
WARMUP = 300        # no entrenamos hasta tener estas transiciones guardadas
C = 300             # cada C pasos copiamos la red online en la red objetivo
EPS_INI, EPS_FIN = 1.0, 0.05    # epsilon-greedy: de explorar todo a casi explotar
EPISODIOS = 200

env = MundoContinuo()
q = crear_red(env.dim_estado, H, env.n_acciones)     # red ONLINE (la que aprende)
q_obj = {k: v.copy() for k, v in q.items()}          # red OBJETIVO (congelada)
opt = Adam(q, LR)
buffer = deque(maxlen=CAP)                            # replay: (s, a, r, s', done)


def epsilon(ep):
    """Decrece linealmente durante el 70% de los episodios y luego se mantiene."""
    frac = min(1.0, ep / (0.7 * EPISODIOS))
    return EPS_INI + frac * (EPS_FIN - EPS_INI)


# ----------------------------------------------------------------------------
# 4) BUCLE DE ENTRENAMIENTO
# ----------------------------------------------------------------------------
recompensas = []
paso_global = 0
for ep in range(EPISODIOS):
    s = env.reset()
    total, fin, eps = 0.0, False, epsilon(ep)
    while not fin:
        # --- politica epsilon-greedy sobre la red online ---
        if rng.random() < eps:
            a = int(rng.integers(env.n_acciones))
        else:
            qv, _ = forward(q, s[None, :])
            a = int(np.argmax(qv[0]))

        s2, r, done, trunc = env.step(a)
        buffer.append((s, a, r, s2, done))     # guardamos la experiencia
        s, total, fin = s2, total + r, (done or trunc)
        paso_global += 1

        # --- una actualizacion de gradiente por paso (cuando ya hay datos) ---
        if len(buffer) >= WARMUP:
            # 1. Muestreamos un minibatch ALEATORIO: rompe la correlacion temporal.
            idx = rng.integers(0, len(buffer), BATCH)
            S = np.array([buffer[i][0] for i in idx])
            A = np.array([buffer[i][1] for i in idx])
            R = np.array([buffer[i][2] for i in idx])
            S2 = np.array([buffer[i][3] for i in idx])
            D = np.array([buffer[i][4] for i in idx], dtype=float)

            # 2. Objetivo con la red OBJETIVO congelada: y = r + gamma * max_a' Q-(s',a').
            Qobj, _ = forward(q_obj, S2)
            y = R + GAMMA * np.max(Qobj, axis=1) * (1.0 - D)

            # 3. Prediccion de la red online y perdida MSE solo en la accion tomada.
            Q, (Z1, A1) = forward(q, S)
            qsa = Q[np.arange(BATCH), A]

            # 4. Backprop manual de L = mean((qsa - y)^2).
            dQ = np.zeros_like(Q)
            dQ[np.arange(BATCH), A] = (2.0 / BATCH) * (qsa - y)
            grads = {"W2": A1.T @ dQ, "b2": dQ.sum(0)}
            dA1 = dQ @ q["W2"].T
            dZ1 = dA1 * (Z1 > 0)                # derivada de la ReLU
            grads["W1"] = S.T @ dZ1
            grads["b1"] = dZ1.sum(0)
            opt.paso(q, grads)

            # 5. Cada C pasos, sincronizamos la red objetivo (theta- <- theta).
            if paso_global % C == 0:
                q_obj = {k: v.copy() for k, v in q.items()}

    recompensas.append(total)

recompensas = np.array(recompensas)


# ----------------------------------------------------------------------------
# 5) RESULTADOS: evaluacion greedy y grafica de la recompensa por episodio
# ----------------------------------------------------------------------------
exitos, pasos_tot = 0, 0
for _ in range(50):
    s, fin = env.reset(), False
    while not fin:
        qv, _ = forward(q, s[None, :])
        s, r, done, trunc = env.step(int(np.argmax(qv[0])))
        fin = done or trunc
    exitos += int(done)
    pasos_tot += env.t

print(f"Recompensa media primeros 10 episodios: {recompensas[:10].mean():+.2f}")
print(f"Recompensa media ultimos  10 episodios: {recompensas[-10:].mean():+.2f}")
print(f"Politica final (greedy): llega a la meta en {exitos}/50 episodios "
      f"({pasos_tot / 50:.0f} pasos de media).")

ventana = 10
media_movil = np.convolve(recompensas, np.ones(ventana) / ventana, mode="valid")

plt.figure(figsize=(8, 4.5))
plt.plot(recompensas, color="#b8c0e0", lw=1, label="recompensa por episodio")
plt.plot(np.arange(ventana - 1, len(recompensas)), media_movil,
         color="#4f46e5", lw=2.5, label=f"media movil ({ventana})")
plt.axhline(0.0, color="#94a3b8", lw=0.8, ls="--")
plt.xlabel("episodio")
plt.ylabel("recompensa acumulada")
plt.title("DQN desde cero (NumPy): aprende a alcanzar la meta")
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
