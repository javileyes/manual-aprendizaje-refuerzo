"""
SAC (Soft Actor-Critic) con PyTorch sobre Pendulum-v1  (SOLO TERMINAL)
=====================================================================

Versión "de verdad" de SAC, el método off-policy actor-crítico de MÁXIMA
ENTROPÍA, sobre el clásico de control continuo Pendulum-v1 de Gymnasium. El
objetivo no es solo maximizar la recompensa, sino la recompensa MÁS la entropía
de la política:
        J(pi) = E[ sum_t r_t + alpha * H(pi(.|s_t)) ]
El término de entropía empuja a la política a mantenerse lo más aleatoria
posible mientras resuelve la tarea: explora mejor y es más robusta.

Ingredientes clave (todos presentes abajo):
  * Política ESTOCÁSTICA gaussiana con squashing tanh y TRUCO DE
    REPARAMETRIZACIÓN (rsample): a = tanh(mu + sigma * epsilon), con
    epsilon ~ N(0, I). Así el gradiente fluye a través del muestreo.
  * DOBLE CRÍTICO (dos redes Q) y se toma el MÍNIMO para el objetivo, lo que
    combate la sobreestimación (idea heredada de TD3).
  * Redes objetivo con actualización suave (Polyak).
  * Ajuste AUTOMÁTICO de la temperatura alpha hacia una entropía objetivo
    (por convención, -dim_accion).

Pendulum-v1: estado de 3 números (cos, sen, velocidad angular), acción continua
= par aplicado en [-2, 2]. La recompensa es negativa; un agente resuelto ronda
un retorno de -200 o mejor (más cercano a 0). Es demasiado pesado para el
navegador: se ejecuta en tu terminal.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/17-control-continuo/sac_torch.py
"""

import random
import collections

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


# ---------------------------------------------------------------------------
# Buffer de repetición
# ---------------------------------------------------------------------------
class Buffer:
    def __init__(self, cap):
        self.mem = collections.deque(maxlen=cap)

    def add(self, s, a, r, s2, d):
        self.mem.append((s, a, r, s2, d))

    def sample(self, n):
        lote = random.sample(self.mem, n)
        s, a, r, s2, d = zip(*lote)
        t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32)
        return (t(s), t(a), t(r).unsqueeze(1), t(s2), t(d).unsqueeze(1))

    def __len__(self):
        return len(self.mem)


# ---------------------------------------------------------------------------
# Actor gaussiano con squashing tanh (política estocástica)
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, dim_s, dim_a, a_high, oculta=256):
        super().__init__()
        self.cuerpo = nn.Sequential(
            nn.Linear(dim_s, oculta), nn.ReLU(),
            nn.Linear(oculta, oculta), nn.ReLU(),
        )
        self.mu = nn.Linear(oculta, dim_a)
        self.log_std = nn.Linear(oculta, dim_a)
        # Escala para llevar tanh(.) in (-1,1) al rango real de la acción.
        self.register_buffer("escala", torch.as_tensor(a_high, dtype=torch.float32))

    def forward(self, s):
        h = self.cuerpo(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def muestrea(self, s):
        """Devuelve (accion, log_prob) con el truco de reparametrización."""
        mu, log_std = self(s)
        std = log_std.exp()
        dist = Normal(mu, std)
        u = dist.rsample()                       # reparametrización: mu + std * eps
        y = torch.tanh(u)                        # squashing a (-1, 1)
        accion = y * self.escala
        # Corrección del cambio de variable tanh en la densidad (regla del jacobiano).
        log_prob = dist.log_prob(u) - torch.log(self.escala * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return accion, log_prob


# ---------------------------------------------------------------------------
# Crítico Q(s, a)  (usaremos dos copias independientes)
# ---------------------------------------------------------------------------
class Critico(nn.Module):
    def __init__(self, dim_s, dim_a, oculta=256):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_s + dim_a, oculta), nn.ReLU(),
            nn.Linear(oculta, oculta), nn.ReLU(),
            nn.Linear(oculta, 1),
        )

    def forward(self, s, a):
        return self.red(torch.cat([s, a], dim=1))


def actualiza_suave(destino, origen, tau):
    for p_obj, p in zip(destino.parameters(), origen.parameters()):
        p_obj.data.mul_(1 - tau).add_(tau * p.data)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrena(pasos_totales=20000, inicio_aprendizaje=1000, batch=256,
            gamma=0.99, tau=0.005, lr=3e-4, semilla=0):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

    env = gym.make("Pendulum-v1")
    dim_s = env.observation_space.shape[0]
    dim_a = env.action_space.shape[0]
    a_high = env.action_space.high              # [2.0] en Pendulum

    actor = Actor(dim_s, dim_a, a_high)
    q1 = Critico(dim_s, dim_a)
    q2 = Critico(dim_s, dim_a)
    q1_obj = Critico(dim_s, dim_a)
    q2_obj = Critico(dim_s, dim_a)
    q1_obj.load_state_dict(q1.state_dict())
    q2_obj.load_state_dict(q2.state_dict())

    opt_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

    # Temperatura alpha ajustada automáticamente hacia una entropía objetivo.
    entropia_objetivo = -float(dim_a)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)

    buffer = Buffer(100_000)
    s, _ = env.reset(seed=semilla)
    retorno_ep, retornos = 0.0, []

    for paso in range(1, pasos_totales + 1):
        # --- Recoger experiencia ---
        if paso < inicio_aprendizaje:
            a = env.action_space.sample()                 # arranque aleatorio
        else:
            with torch.no_grad():
                a, _ = actor.muestrea(torch.as_tensor(s, dtype=torch.float32)[None])
            a = a[0].numpy()

        s2, r, term, trunc, _ = env.step(a)
        # Para bootstrapping distinguimos fin real (term) de corte por tiempo (trunc):
        # Pendulum nunca "termina" de verdad, así que d=0 y siempre se hace bootstrap.
        buffer.add(s, a, r, s2, float(term))
        s = s2
        retorno_ep += r

        if term or trunc:
            retornos.append(retorno_ep)
            s, _ = env.reset()
            retorno_ep = 0.0

        # --- Aprender ---
        if len(buffer) >= max(batch, inicio_aprendizaje):
            bs, ba, br, bs2, bd = buffer.sample(batch)
            alpha = log_alpha.exp()

            # (1) Crítico: objetivo con doble Q (mínimo) y bonus de entropía.
            with torch.no_grad():
                a2, logp2 = actor.muestrea(bs2)
                q_obj = torch.min(q1_obj(bs2, a2), q2_obj(bs2, a2)) - alpha * logp2
                y = br + gamma * (1 - bd) * q_obj
            loss_q = F.mse_loss(q1(bs, ba), y) + F.mse_loss(q2(bs, ba), y)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()

            # (2) Actor: maximizar Q - alpha*logp (reparametrización).
            a_pi, logp = actor.muestrea(bs)
            q_pi = torch.min(q1(bs, a_pi), q2(bs, a_pi))
            loss_actor = (alpha.detach() * logp - q_pi).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()

            # (3) Temperatura: acerca la entropía de la política a la objetivo.
            loss_alpha = -(log_alpha * (logp + entropia_objetivo).detach()).mean()
            opt_alpha.zero_grad()
            loss_alpha.backward()
            opt_alpha.step()

            # (4) Redes objetivo, actualización suave.
            actualiza_suave(q1_obj, q1, tau)
            actualiza_suave(q2_obj, q2, tau)

        if paso % 2000 == 0 and retornos:
            media = np.mean(retornos[-10:])
            print(f"paso {paso:6d} | retorno medio (últimos 10 ep.) = {media:8.1f} "
                  f"| alpha = {log_alpha.exp().item():.3f}")

    env.close()
    return actor


if __name__ == "__main__":
    entrena()
