"""
double_dueling_torch.py - DQN con Double Q-learning y arquitectura Dueling sobre
CartPole-v1 (PyTorch + Gymnasium).

SOLO TERMINAL: este ejemplo usa torch y gymnasium, demasiado pesados para el
navegador. Reune dos mejoras de DQN del capitulo 13:

  * Double DQN: la red 'online' ELIGE la mejor accion en s' (argmax) y la red
    'target' la EVALUA. Asi se rompe el sesgo de sobreestimacion del max.
        y = r + gamma * Q_target(s', argmax_a Q_online(s', a))

  * Dueling: la red se separa en dos ramas, V(s) (como de bueno es el estado) y
    A(s,a) (la ventaja de cada accion sobre la media), y las recombina como
        Q(s,a) = V(s) + (A(s,a) - media_a A(s,a))

Anadir Prioritized Experience Replay (PER) es una extension natural: bastaria
sustituir el ReplayBuffer uniforme por uno que muestree segun |error TD|.

Requisitos:
    pip install torch gymnasium
Ejecucion:
    python code/13-mejoras-dqn/double_dueling_torch.py
"""
import random
from collections import deque, namedtuple

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Reproducibilidad ---
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

Transicion = namedtuple("Transicion", ["s", "a", "r", "s2", "fin"])


class ReplayBuffer:
    """Memoria de repeticion uniforme (muestreo al azar sin prioridad)."""

    def __init__(self, cap=50_000):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(Transicion(*args))

    def sample(self, n):
        lote = random.sample(self.buf, n)
        s = torch.as_tensor(np.array([t.s for t in lote]), dtype=torch.float32)
        a = torch.as_tensor([t.a for t in lote], dtype=torch.int64).unsqueeze(1)
        r = torch.as_tensor([t.r for t in lote], dtype=torch.float32).unsqueeze(1)
        s2 = torch.as_tensor(np.array([t.s2 for t in lote]), dtype=torch.float32)
        fin = torch.as_tensor([t.fin for t in lote], dtype=torch.float32).unsqueeze(1)
        return s, a, r, s2, fin

    def __len__(self):
        return len(self.buf)


class DuelingQNet(nn.Module):
    """Red Dueling: Q(s,a) = V(s) + (A(s,a) - media_a A(s,a))."""

    def __init__(self, n_obs, n_acc, oculto=128):
        super().__init__()
        self.cuerpo = nn.Sequential(
            nn.Linear(n_obs, oculto), nn.ReLU(),
            nn.Linear(oculto, oculto), nn.ReLU(),
        )
        self.valor = nn.Linear(oculto, 1)        # rama V(s)   -> escalar
        self.ventaja = nn.Linear(oculto, n_acc)  # rama A(s,a) -> una por accion

    def forward(self, x):
        h = self.cuerpo(x)
        v = self.valor(h)                        # (batch, 1)
        a = self.ventaja(h)                      # (batch, n_acc)
        # restar la media de la ventaja fija la ambiguedad V/A y estabiliza
        return v + (a - a.mean(dim=1, keepdim=True))


def entrenar(episodios=400):
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_acc = env.action_space.n

    online = DuelingQNet(n_obs, n_acc)
    target = DuelingQNet(n_obs, n_acc)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)

    buffer = ReplayBuffer()
    GAMMA = 0.99
    BATCH = 64
    EPS_INI, EPS_FIN, EPS_DECAY = 1.0, 0.02, 3000.0
    SYNC_CADA = 500          # pasos entre sincronizaciones de la red target
    INICIO_APRENDER = 1000   # transiciones minimas antes de entrenar
    pasos = 0
    historial = deque(maxlen=20)

    for ep in range(1, episodios + 1):
        s, _ = env.reset(seed=SEED + ep)
        total, done = 0.0, False
        while not done:
            eps = EPS_FIN + (EPS_INI - EPS_FIN) * np.exp(-pasos / EPS_DECAY)
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = online(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
                    a = int(q.argmax(dim=1).item())

            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            # 'term' (no 'trunc') marca el fin real del episodio para el bootstrap
            buffer.push(s, a, r, s2, float(term))
            s = s2
            total += r
            pasos += 1

            # --- paso de aprendizaje ---
            if len(buffer) >= INICIO_APRENDER:
                bs, ba, br, bs2, bfin = buffer.sample(BATCH)
                q_sa = online(bs).gather(1, ba)                 # Q(s,a) actual
                with torch.no_grad():
                    # DOUBLE DQN: online elige la accion, target la evalua
                    a_star = online(bs2).argmax(dim=1, keepdim=True)
                    q_next = target(bs2).gather(1, a_star)
                    y = br + GAMMA * (1.0 - bfin) * q_next
                perdida = F.smooth_l1_loss(q_sa, y)
                opt.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()

            if pasos % SYNC_CADA == 0:
                target.load_state_dict(online.state_dict())

        historial.append(total)
        if ep % 20 == 0:
            media = np.mean(historial)
            print(f"Episodio {ep:3d} | recompensa media (ult. 20) = {media:6.1f} | eps = {eps:.3f}")

    env.close()
    print("Listo. CartPole-v1 se considera resuelto con recompensa media >= 475.")


if __name__ == "__main__":
    entrenar()
