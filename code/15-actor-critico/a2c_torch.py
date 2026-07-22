"""
A2C (Advantage Actor-Critic) con PyTorch sobre CartPole-v1.

Este ejemplo NO corre en el navegador: torch y gymnasium son demasiado pesados.
Ejecútalo en tu terminal.

Idea:
  * Una sola red con un tronco compartido y DOS cabezas:
      - la cabeza del ACTOR produce los logits de la política pi(a|s);
      - la cabeza del CRÍTICO estima el valor V(s).
  * En cada iteración recogemos un LOTE de episodios completos, estimamos la
    VENTAJA con GAE (Generalized Advantage Estimation) y hacemos UNA actualización
    on-policy que mejora actor y crítico a la vez.
  * GAE interpola entre el error TD de un paso (lambda=0: poca varianza pero
    algo de sesgo) y el retorno Monte Carlo (lambda=1: sin sesgo pero mucha
    varianza). lambda=0.95 suele ser un buen compromiso.
  * Añadimos un bonus de ENTROPÍA para mantener la exploración.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/15-actor-critico/a2c_torch.py

El máximo por episodio es 500 pasos. Verás que A2C aprende en menos de un minuto
a mantener el poste durante cientos de pasos, pero su curva OSCILA y no se clava
en el máximo: esa inestabilidad de los pasos grandes de actor-crítico es justo lo
que PPO (capítulo 16) domestica. Clavar un 500 estable con este código básico
cuesta, a propósito.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym


class ActorCritico(nn.Module):
    """Tronco compartido + dos cabezas: política (actor) y valor (crítico)."""

    def __init__(self, dim_obs, n_acciones, oculto=128):
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Linear(dim_obs, oculto), nn.Tanh(),
            nn.Linear(oculto, oculto), nn.Tanh(),
        )
        self.cabeza_pi = nn.Linear(oculto, n_acciones)  # logits de la política
        self.cabeza_v = nn.Linear(oculto, 1)            # V(s)

    def forward(self, x):
        h = self.tronco(x)
        return self.cabeza_pi(h), self.cabeza_v(h).squeeze(-1)


def calcula_gae(recompensas, valores, v_ultimo, gamma, lam):
    """
    Ventaja generalizada (GAE) para UN episodio.
      delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
      A_t     = sum_l (gamma*lambda)^l * delta_{t+l}
    'v_ultimo' es V del estado que sigue al último paso: 0 si el episodio TERMINÓ
    de verdad (el poste cayó) o V(s_final) si solo se truncó por límite de tiempo.
    Devuelve (ventajas, retornos) con retornos = ventajas + valores.
    """
    T = len(recompensas)
    ventajas = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        v_siguiente = v_ultimo if t == T - 1 else valores[t + 1]
        delta = recompensas[t] + gamma * v_siguiente - valores[t]
        gae = delta + gamma * lam * gae
        ventajas[t] = gae
    retornos = ventajas + valores
    return ventajas, retornos


def main():
    semilla = 0
    torch.manual_seed(semilla)
    np.random.seed(semilla)

    env = gym.make("CartPole-v1")
    dim_obs = env.observation_space.shape[0]
    n_acciones = env.action_space.n

    modelo = ActorCritico(dim_obs, n_acciones)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=3e-3)

    gamma, lam = 0.99, 0.95     # descuento y lambda de GAE
    coef_valor = 0.5            # peso de la pérdida del crítico
    coef_entropia = 0.01        # peso del bonus de entropía
    pasos_por_lote = 2000       # transiciones (en episodios completos) por update
    max_actualizaciones = 500

    historial = []                 # retorno de cada episodio terminado
    ventana_100 = deque(maxlen=100)
    mejor_media = 0.0

    for it in range(max_actualizaciones):
        obs_b, act_b, adv_b, ret_b = [], [], [], []
        pasos = 0

        # --- 1) Recolectar un lote de episodios completos con la política actual ---
        while pasos < pasos_por_lote:
            obs, _ = env.reset()
            e_obs, e_act, e_rew, e_val = [], [], [], []
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    logits, valor = modelo(obs_t)
                accion = Categorical(logits=logits).sample()
                obs2, recompensa, terminado, truncado, _ = env.step(accion.item())

                e_obs.append(obs)
                e_act.append(accion.item())
                e_rew.append(recompensa)
                e_val.append(valor.item())
                obs = obs2
                if terminado or truncado:
                    break

            # Bootstrap: 0 si el poste cayó; V(s_final) si solo se truncó.
            if truncado and not terminado:
                with torch.no_grad():
                    _, v_fin = modelo(torch.as_tensor(obs2, dtype=torch.float32))
                v_ultimo = v_fin.item()
            else:
                v_ultimo = 0.0

            ventajas, retornos = calcula_gae(
                np.array(e_rew, dtype=np.float32),
                np.array(e_val, dtype=np.float32),
                v_ultimo, gamma, lam,
            )
            obs_b += e_obs
            act_b += e_act
            adv_b += list(ventajas)
            ret_b += list(retornos)

            retorno_ep = float(sum(e_rew))
            historial.append(retorno_ep)
            ventana_100.append(retorno_ep)
            pasos += len(e_rew)

        # --- 2) Una actualización on-policy (actor + crítico + entropía) ---
        obs_t = torch.as_tensor(np.array(obs_b), dtype=torch.float32)
        act_t = torch.as_tensor(np.array(act_b), dtype=torch.int64)
        adv_t = torch.as_tensor(np.array(adv_b), dtype=torch.float32)
        ret_t = torch.as_tensor(np.array(ret_b), dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)  # normalizar

        logits, valores = modelo(obs_t)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_t)

        perdida_actor = -(logp * adv_t).mean()           # gradiente de política
        perdida_critico = F.mse_loss(valores, ret_t)     # ajuste del crítico
        entropia = dist.entropy().mean()                 # exploración
        perdida = perdida_actor + coef_valor * perdida_critico - coef_entropia * entropia

        optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 0.5)
        optimizador.step()

        # --- 3) Registro del progreso ---
        media = float(np.mean(ventana_100))
        mejor_media = max(mejor_media, media)
        if it % 10 == 0:
            print(f"act {it:4d} | episodios {len(historial):5d} | "
                  f"media(100) = {media:6.1f}")

    env.close()
    print(f"\nEntrenamiento terminado. Mejor media(100) = {mejor_media:.0f} pasos "
          f"(de un máximo de 500).")
    print("A2C aprende deprisa, pero su rendimiento oscila: esa inestabilidad de "
          "los pasos grandes es justo lo que PPO (capítulo 16) domestica.")

    # --- Gráfica opcional de la curva de aprendizaje ---
    try:
        import matplotlib.pyplot as plt
        h = np.array(historial, dtype=np.float32)
        if len(h) >= 10:
            suave = np.convolve(h, np.ones(10) / 10, mode="valid")
            plt.figure(figsize=(8, 4.5))
            plt.plot(h, color="#c7d2fe", lw=1, label="retorno por episodio")
            plt.plot(np.arange(len(suave)) + 10, suave, color="#4f46e5",
                     lw=2, label="media móvil (10)")
            plt.axhline(475, color="#059669", ls="--", lw=1, label="referencia 475")
            plt.xlabel("Episodio")
            plt.ylabel("Retorno (pasos que el poste aguanta)")
            plt.title("A2C sobre CartPole-v1")
            plt.legend()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("(sin gráfica:", e, ")")


if __name__ == "__main__":
    main()
