"""
DQN idiomatico con PyTorch + Gymnasium sobre CartPole-v1 (para TERMINAL).
========================================================================

Este es el "mismo" DQN que la version desde cero en NumPy, pero escrito como se
hace en la practica: la red es un `nn.Module`, el optimizador es `optim.Adam`, el
autograd de PyTorch calcula la retropropagacion por nosotros y el entorno lo pone
Gymnasium. Reconoceras las mismas dos piezas clave que estabilizan el metodo:

  * ReplayBuffer  -> rompe la correlacion entre transiciones consecutivas.
  * red objetivo  -> un juego de pesos congelado que da el objetivo y = r + gamma*max Q-.

CartPole-v1: un carro con un poste articulado. Estado de 4 numeros (posicion y
velocidad del carro, angulo y velocidad angular del poste), 2 acciones (empujar a
izquierda o derecha) y +1 de recompensa por cada paso que el poste siga en pie
(maximo 500). Se considera "resuelto" con una recompensa media >= 475.

NO corre en el navegador (torch y gymnasium son demasiado pesados). Ejecutalo en tu
terminal:

    pip install -r requirements.txt
    python code/12-dqn/dqn_torch.py

En una CPU normal aprende a mantener el poste en pie en uno o dos minutos.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# --------------------------------------------------------------------------- #
# Hiperparametros
# --------------------------------------------------------------------------- #
SEMILLA = 0
GAMMA = 0.99             # factor de descuento
LR = 1e-3               # tasa de aprendizaje de Adam
BATCH = 128             # tamano del minibatch
CAP = 10_000           # capacidad del replay buffer
WARMUP = 1_000         # transiciones minimas antes de empezar a entrenar
SYNC_OBJETIVO = 500     # cada cuantos PASOS copiamos theta -> theta-
EPS_INI, EPS_FIN = 1.0, 0.05
EPS_DECAY = 6_000       # constante de la caida exponencial de epsilon (en pasos)
MAX_EPISODIOS = 800
OBJETIVO_RESUELTO = 475.0    # media de las ultimas 100 recompensas para "resolver"

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# La red Q: un MLP que mapea estado (4) -> valor Q de cada accion (2)
# --------------------------------------------------------------------------- #
class RedQ(nn.Module):
    def __init__(self, dim_estado, n_acciones, oculta=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_estado, oculta), nn.ReLU(),
            nn.Linear(oculta, oculta), nn.ReLU(),
            nn.Linear(oculta, n_acciones),
        )

    def forward(self, x):
        return self.red(x)


# --------------------------------------------------------------------------- #
# Replay buffer: guarda (s, a, r, s', terminado) y devuelve minibatches al azar
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacidad):
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, s, a, r, s2, terminado):
        self.memoria.append((s, a, r, s2, terminado))

    def muestrear(self, batch):
        lote = random.sample(self.memoria, batch)
        s, a, r, s2, term = zip(*lote)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32, device=DISPOSITIVO),
            torch.as_tensor(a, dtype=torch.int64, device=DISPOSITIVO).unsqueeze(1),
            torch.as_tensor(r, dtype=torch.float32, device=DISPOSITIVO).unsqueeze(1),
            torch.as_tensor(np.array(s2), dtype=torch.float32, device=DISPOSITIVO),
            torch.as_tensor(term, dtype=torch.float32, device=DISPOSITIVO).unsqueeze(1),
        )

    def __len__(self):
        return len(self.memoria)


def epsilon(paso):
    """Caida exponencial de epsilon: mucho explorar al principio, casi nada al final."""
    return EPS_FIN + (EPS_INI - EPS_FIN) * np.exp(-paso / EPS_DECAY)


def entrenar():
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    env = gym.make("CartPole-v1")
    env.action_space.seed(SEMILLA)          # que las acciones al azar tambien sean reproducibles
    dim_estado = env.observation_space.shape[0]
    n_acciones = env.action_space.n

    q = RedQ(dim_estado, n_acciones).to(DISPOSITIVO)          # red ONLINE (aprende)
    q_obj = RedQ(dim_estado, n_acciones).to(DISPOSITIVO)      # red OBJETIVO (congelada)
    q_obj.load_state_dict(q.state_dict())
    q_obj.eval()

    opt = optim.Adam(q.parameters(), lr=LR)
    buffer = ReplayBuffer(CAP)

    paso_global = 0
    mejor_media = float("-inf")
    historico = deque(maxlen=100)

    for ep in range(1, MAX_EPISODIOS + 1):
        estado, _ = env.reset(seed=SEMILLA + ep)
        recompensa_ep = 0.0
        terminado = truncado = False

        while not (terminado or truncado):
            # --- politica epsilon-greedy sobre la red online ---
            eps = epsilon(paso_global)
            if random.random() < eps:
                accion = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.as_tensor(estado, dtype=torch.float32, device=DISPOSITIVO)
                    accion = int(q(s).argmax().item())

            sig, recompensa, terminado, truncado, _ = env.step(accion)
            # Guardamos SOLO 'terminado' (el poste cae): un corte por tiempo
            # (truncado) no debe poner a cero el valor futuro al bootstrapear.
            buffer.guardar(estado, accion, recompensa, sig, terminado)
            estado = sig
            recompensa_ep += recompensa
            paso_global += 1

            # --- una actualizacion de gradiente por paso ---
            if len(buffer) >= WARMUP:
                s, a, r, s2, term = buffer.muestrear(BATCH)

                # Q(s,a) de la red online, seleccionando la accion tomada.
                q_sa = q(s).gather(1, a)

                # Objetivo con la red objetivo congelada (sin gradiente).
                with torch.no_grad():
                    max_q2 = q_obj(s2).max(dim=1, keepdim=True).values
                    objetivo = r + GAMMA * max_q2 * (1.0 - term)

                # La misma perdida MSE del capitulo. Para mas estabilidad puedes
                # cambiarla por la de Huber: nn.functional.smooth_l1_loss(q_sa, objetivo).
                perdida = nn.functional.mse_loss(q_sa, objetivo)

                opt.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)   # evita pasos gigantes
                opt.step()

                # Sincronizacion periodica de la red objetivo: theta- <- theta.
                if paso_global % SYNC_OBJETIVO == 0:
                    q_obj.load_state_dict(q.state_dict())

        historico.append(recompensa_ep)
        media100 = np.mean(historico)
        mejor_media = max(mejor_media, media100)
        if ep % 10 == 0:
            print(f"Episodio {ep:4d} | recompensa {recompensa_ep:6.1f} | "
                  f"media100 {media100:6.1f} | epsilon {epsilon(paso_global):.3f}")

        if len(historico) == 100 and media100 >= OBJETIVO_RESUELTO:
            print(f"\nResuelto en el episodio {ep}: media de las ultimas 100 "
                  f"recompensas = {media100:.1f} (>= {OBJETIVO_RESUELTO:.0f}).")
            break
    else:
        # Si llegamos aqui, no se cruzo el umbral de "resuelto" en MAX_EPISODIOS.
        print(f"\nFin del entrenamiento ({MAX_EPISODIOS} episodios). "
              f"Mejor media de 100 episodios = {mejor_media:.1f}.")
        print("La politica aprende a sostener el poste cientos de pasos y alcanza a "
              "menudo el maximo de 500, pero el DQN 'a secas' sufre colapsos ocasionales "
              "por sobreestimacion de Q. Esa inestabilidad es justo lo que atacan "
              "Double DQN, Dueling y PER en el capitulo 13.")

    env.close()
    return q


if __name__ == "__main__":
    entrenar()
