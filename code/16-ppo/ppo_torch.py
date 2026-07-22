"""
PPO con PyTorch sobre CartPole-v1  (SOLO TERMINAL)
==================================================

Version "de verdad" de PPO (Proximal Policy Optimization) en un unico fichero,
al estilo CleanRL, sobre el entorno clasico CartPole-v1 de Gymnasium. Reune
todas las piezas del capitulo:

  * Recogida de un ROLLOUT de longitud fija (num_pasos) siguiendo la politica.
  * Ventajas con GAE(gamma, lambda) y objetivo del critico = ventaja + valor.
  * Objetivo RECORTADO (clipped surrogate) sobre el ratio r_t(theta).
  * VARIAS EPOCAS de SGD por lote, dividiendo el lote en MINIBATCHES.
  * Perdida total = perdida_politica(clip) + c1*perdida_valor - c2*entropia,
    con recorte de la norma del gradiente para mayor estabilidad.

Este ejemplo necesita PyTorch y Gymnasium, demasiado pesados para el navegador:
se ejecuta en tu terminal.

Como ejecutarlo:
    pip install -r requirements.txt
    python code/16-ppo/ppo_torch.py

CartPole se considera "resuelto" cuando la recompensa media (= numero de pasos
que el poste aguanta en pie) supera ~475 sobre 500 posibles. PPO suele resolverlo
en unas decenas de miles de pasos.
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# --- Hiperparametros (valores tipicos de PPO para CartPole) ---------------
SEMILLA          = 1
TOTAL_PASOS      = 200_000     # tope de pasos de entorno (paramos antes si se resuelve)
NUM_PASOS        = 512         # longitud del rollout entre actualizaciones
LR               = 2.5e-4      # tasa de aprendizaje de Adam
GAMMA            = 0.99        # factor de descuento
GAE_LAMBDA       = 0.95        # lambda de GAE (sesgo/varianza)
NUM_MINIBATCHES  = 4           # minibatches en los que dividir cada lote
EPOCAS           = 4           # epocas de SGD por lote (reutilizamos los datos)
CLIP_EPS         = 0.2         # epsilon del recorte del ratio
C1_VALOR         = 0.5         # peso de la perdida del critico
C2_ENTROPIA      = 0.01        # peso del bonus de entropia
MAX_GRAD_NORM    = 0.5         # recorte de la norma del gradiente
ANNEAL_LR        = True        # decaimiento lineal de la tasa de aprendizaje
OBJETIVO_RESUELTO = 475.0      # recompensa media (ult. 20 episodios) para parar

TAM_LOTE      = NUM_PASOS                       # 1 entorno -> lote = num_pasos
TAM_MINIBATCH = TAM_LOTE // NUM_MINIBATCHES
NUM_ITERS     = TOTAL_PASOS // TAM_LOTE


def init_capa(capa, std=np.sqrt(2), sesgo=0.0):
    """Inicializacion ortogonal (recomendada para PPO)."""
    nn.init.orthogonal_(capa.weight, std)
    nn.init.constant_(capa.bias, sesgo)
    return capa


class Agente(nn.Module):
    """Actor y critico como dos MLP independientes."""

    def __init__(self, dim_obs, n_acciones):
        super().__init__()
        self.critico = nn.Sequential(
            init_capa(nn.Linear(dim_obs, 64)), nn.Tanh(),
            init_capa(nn.Linear(64, 64)), nn.Tanh(),
            init_capa(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            init_capa(nn.Linear(dim_obs, 64)), nn.Tanh(),
            init_capa(nn.Linear(64, 64)), nn.Tanh(),
            init_capa(nn.Linear(64, n_acciones), std=0.01),   # logits pequenos al inicio
        )

    def valor(self, x):
        return self.critico(x).squeeze(-1)                    # (B,)

    def accion_y_valor(self, x, accion=None):
        """Devuelve (accion, log pi(a|s), entropia, V(s)) para x de forma (B, dim_obs)."""
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if accion is None:
            accion = dist.sample()
        return accion, dist.log_prob(accion), dist.entropy(), self.critico(x).squeeze(-1)


def entrena():
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)

    # RecordEpisodeStatistics rellena info["episode"] al terminar cada episodio.
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    dim_obs = env.observation_space.shape[0]
    n_acciones = env.action_space.n

    agente = Agente(dim_obs, n_acciones)
    opt = optim.Adam(agente.parameters(), lr=LR, eps=1e-5)

    # --- Almacenamiento del rollout (1 entorno) ---
    obs      = torch.zeros((NUM_PASOS, dim_obs))
    acciones = torch.zeros(NUM_PASOS, dtype=torch.long)
    logprobs = torch.zeros(NUM_PASOS)
    recomps  = torch.zeros(NUM_PASOS)
    dones    = torch.zeros(NUM_PASOS)
    valores  = torch.zeros(NUM_PASOS)

    paso_global = 0
    obs_np, _ = env.reset(seed=SEMILLA)
    next_obs = torch.tensor(obs_np, dtype=torch.float32)
    next_done = torch.zeros(())
    ventana = []                                  # ultimas recompensas de episodio

    for it in range(1, NUM_ITERS + 1):
        # Decaimiento lineal de la tasa de aprendizaje.
        if ANNEAL_LR:
            frac = 1.0 - (it - 1.0) / NUM_ITERS
            opt.param_groups[0]["lr"] = frac * LR

        # ---------- 1) Recogida del rollout ----------
        for t in range(NUM_PASOS):
            paso_global += 1
            obs[t] = next_obs
            dones[t] = next_done
            with torch.no_grad():
                a, logp, _, v = agente.accion_y_valor(next_obs.unsqueeze(0))
            valores[t] = v.squeeze(0)
            acciones[t] = a.squeeze(0)
            logprobs[t] = logp.squeeze(0)

            obs_np, r, term, trunc, info = env.step(int(a.item()))
            recomps[t] = float(r)
            done = term or trunc
            next_done = torch.tensor(float(done))

            if done:
                # Un entorno simple de Gymnasium NO se reinicia solo: al terminar
                # el episodio leemos su retorno y llamamos a reset() a mano.
                if "episode" in info:
                    ventana.append(float(info["episode"]["r"]))
                    if len(ventana) > 20:
                        ventana.pop(0)
                obs_np, _ = env.reset()
            next_obs = torch.tensor(obs_np, dtype=torch.float32)

        # ---------- 2) Ventajas con GAE ----------
        with torch.no_grad():
            next_valor = agente.valor(next_obs.unsqueeze(0)).squeeze(0)
            ventajas = torch.zeros(NUM_PASOS)
            ultimo_gae = 0.0
            for t in reversed(range(NUM_PASOS)):
                if t == NUM_PASOS - 1:
                    no_terminal = 1.0 - next_done
                    v_siguiente = next_valor
                else:
                    no_terminal = 1.0 - dones[t + 1]
                    v_siguiente = valores[t + 1]
                delta = recomps[t] + GAMMA * v_siguiente * no_terminal - valores[t]
                ultimo_gae = delta + GAMMA * GAE_LAMBDA * no_terminal * ultimo_gae
                ventajas[t] = ultimo_gae
            retornos = ventajas + valores          # objetivo de regresion del critico

        # ---------- 3) Optimizacion: EPOCAS x MINIBATCHES ----------
        indices = np.arange(TAM_LOTE)
        for _ in range(EPOCAS):
            np.random.shuffle(indices)
            for arranque in range(0, TAM_LOTE, TAM_MINIBATCH):
                mb = indices[arranque:arranque + TAM_MINIBATCH]

                _, nueva_logp, entropia, nuevo_v = agente.accion_y_valor(
                    obs[mb], acciones[mb])
                log_ratio = nueva_logp - logprobs[mb]
                ratio = log_ratio.exp()            # r_t(theta)

                # Ventajas normalizadas por minibatch (media 0, desviacion 1).
                mb_adv = ventajas[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # --- Perdida de politica: objetivo recortado (con signo -) ---
                perdida_pi_1 = -mb_adv * ratio
                perdida_pi_2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                perdida_pi = torch.max(perdida_pi_1, perdida_pi_2).mean()

                # --- Perdida del critico: error cuadratico contra el retorno ---
                perdida_v = 0.5 * ((nuevo_v - retornos[mb]) ** 2).mean()

                # --- Bonus de entropia (fomenta explorar) ---
                perdida_ent = entropia.mean()

                perdida = perdida_pi + C1_VALOR * perdida_v - C2_ENTROPIA * perdida_ent

                opt.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(agente.parameters(), MAX_GRAD_NORM)
                opt.step()

        media = np.mean(ventana) if ventana else float("nan")
        print(f"Iter {it:3d} | pasos {paso_global:6d} | "
              f"recompensa media (ult. 20) = {media:6.1f}")

        if len(ventana) >= 20 and media >= OBJETIVO_RESUELTO:
            print(f"\n¡Resuelto! Recompensa media (ult. 20) = {media:.1f} "
                  f"tras {paso_global} pasos.")
            break

    env.close()
    return agente


if __name__ == "__main__":
    entrena()
