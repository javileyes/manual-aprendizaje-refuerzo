"""
REINFORCE con PyTorch sobre CartPole-v1  (SOLO TERMINAL)
=======================================================

Version "de verdad" del algoritmo REINFORCE (Monte Carlo policy gradient) sobre
el entorno clasico CartPole-v1 de Gymnasium. La politica es una red neuronal
pequena (un MLP) que, dado el estado (4 numeros), produce logits para las 2
acciones (empujar el carro a izquierda o derecha). Muestreamos la accion de esa
distribucion categorica, jugamos el episodio completo y actualizamos la red por
ascenso de gradiente de la funcion objetivo J(theta) = E[G_0].

Como baseline para reducir la varianza usamos la normalizacion de los retornos
del episodio (les restamos su media y dividimos por su desviacion). Restar una
media es un baseline valido: no cambia la direccion esperada del gradiente
(no introduce sesgo) pero si reduce mucho su varianza.

Este ejemplo necesita PyTorch y Gymnasium, demasiado pesados para el navegador:
se ejecuta en tu terminal.

Como ejecutarlo:
  pip install -r requirements.txt
  python code/14-policy-gradient-reinforce/reinforce_torch.py

CartPole se considera "resuelto" cuando la recompensa media (= numero de pasos
que el poste aguanta en pie) supera ~475 sobre 500 posibles.
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PoliticaMLP(nn.Module):
    """Red que transforma el estado en logits sobre las acciones."""

    def __init__(self, dim_estado, n_acciones, oculta=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_estado, oculta),
            nn.ReLU(),
            nn.Linear(oculta, n_acciones),
        )

    def forward(self, estado):
        return self.red(estado)          # logits (sin softmax; lo hace Categorical)


def elige_accion(politica, estado):
    """Muestrea una accion ~ pi_theta(.|s) y devuelve (accion, log pi(a|s))."""
    estado_t = torch.as_tensor(estado, dtype=torch.float32)
    logits = politica(estado_t)
    dist = Categorical(logits=logits)    # softmax interno sobre los logits
    accion = dist.sample()
    return int(accion.item()), dist.log_prob(accion)


def retornos_descontados(recompensas, gamma):
    """G_t = r_t + gamma*r_{t+1} + ... (calculado hacia atras)."""
    G, acc = [], 0.0
    for r in reversed(recompensas):
        acc = r + gamma * acc
        G.append(acc)
    G.reverse()
    return torch.tensor(G, dtype=torch.float32)


def entrena(n_episodios=600, gamma=0.99, lr=1e-2, semilla=0):
    rng_env_seed = semilla
    torch.manual_seed(semilla)
    np.random.seed(semilla)

    env = gym.make("CartPole-v1")
    politica = PoliticaMLP(env.observation_space.shape[0], env.action_space.n)
    opt = optim.Adam(politica.parameters(), lr=lr)

    ventana = []                         # ultimas recompensas para la media movil
    for ep in range(n_episodios):
        estado, _ = env.reset(seed=rng_env_seed + ep)
        log_probs, recompensas = [], []
        terminado = False
        while not terminado:
            accion, logp = elige_accion(politica, estado)
            estado, r, term, trunc, _ = env.step(accion)
            log_probs.append(logp)
            recompensas.append(r)
            terminado = term or trunc

        # --- Retornos y baseline (normalizacion) ---
        G = retornos_descontados(recompensas, gamma)
        G = (G - G.mean()) / (G.std() + 1e-8)     # baseline que reduce varianza

        # --- Perdida de policy gradient: -sum_t log pi(a_t|s_t) * ventaja_t ---
        log_probs = torch.stack(log_probs)
        perdida = -(log_probs * G).sum()

        opt.zero_grad()
        perdida.backward()
        opt.step()

        total = sum(recompensas)
        ventana.append(total)
        if len(ventana) > 50:
            ventana.pop(0)
        media = np.mean(ventana)

        if (ep + 1) % 20 == 0:
            print(f"Episodio {ep + 1:4d} | recompensa = {total:6.1f} "
                  f"| media (ult. 50) = {media:6.1f}")

        if media >= 475.0 and len(ventana) >= 50:
            print(f"\n¡Resuelto en el episodio {ep + 1}! "
                  f"Media (ult. 50) = {media:.1f}")
            break

    env.close()
    return politica


if __name__ == "__main__":
    entrena()
