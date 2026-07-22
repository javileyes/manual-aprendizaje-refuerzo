"""
PPO desde cero (solo NumPy + Matplotlib)
========================================

Implementa PPO (Proximal Policy Optimization) con el objetivo RECORTADO
(clipped surrogate), ventajas por GAE y VARIAS EPOCAS de ascenso de gradiente
por cada lote de datos, sobre un pequeno gridworld 5x5. Todo con actor y critico
LINEALES para que las derivadas se puedan escribir a mano y el ejemplo corra en
segundos (incluso en el navegador con Pyodide).

Piezas del algoritmo:
  * Actor:   pi_theta(a|s) = softmax_a( phi(s) . theta[:, a] ),  phi(s) one-hot.
  * Critico: V_w(s) = phi(s) . w  (lineal).
  * Ratio:   r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s).
  * Objetivo recortado:
        L^CLIP = E[ min( r_t * A_t , clip(r_t, 1-eps, 1+eps) * A_t ) ].
  * Ventaja: GAE(gamma, lambda) calculada por episodio.
  * Perdida total = -L^CLIP + c1 * perdida_valor - c2 * bonus_entropia,
    optimizada durante K epocas sobre el MISMO lote (reutilizamos los datos).

Como el actor y el critico son lineales y no comparten parametros, actualizamos
theta (subiendo L^CLIP + c2*H) y w (bajando el error del critico) por separado;
es equivalente a minimizar la perdida total conjunta con el peso c1 absorbido en
el paso de aprendizaje del critico.

Que veras al ejecutarlo:
  * Por consola, el retorno medio por iteracion sube (se hace menos negativo)
    hasta acercarse al optimo (el camino mas corto = 8 pasos, retorno -8).
  * Una grafica con el retorno por iteracion y su media movil, claramente
    creciente hacia la linea del optimo.

Como ejecutarlo en tu terminal:
    pip install -r requirements.txt
    python code/16-ppo/ppo_desde_cero.py

Solo necesita numpy y matplotlib, asi que tambien corre en el navegador (Pyodide).
"""

import numpy as np
import matplotlib.pyplot as plt


# --- El entorno: gridworld 5x5 -------------------------------------------
# Estados 0..24 (fila = s // 5, columna = s % 5). Inicio: 0 (arriba-izq).
# Meta: 24 (abajo-der). Acciones: 0=arriba 1=abajo 2=izquierda 3=derecha.
# Recompensa: -1 por paso (buscamos el camino mas corto). La meta es terminal.
LADO = 5
N_ESTADOS = LADO * LADO
N_ACCIONES = 4
INICIO = 0
META = N_ESTADOS - 1
OPTIMO = -2 * (LADO - 1)          # camino mas corto = 8 pasos -> retorno -8


def paso(estado, accion):
    """Dinamica del entorno: devuelve (nuevo_estado, recompensa, terminado)."""
    fila, col = divmod(estado, LADO)
    if accion == 0:                  # arriba
        fila = max(fila - 1, 0)
    elif accion == 1:                # abajo
        fila = min(fila + 1, LADO - 1)
    elif accion == 2:                # izquierda
        col = max(col - 1, 0)
    else:                            # derecha
        col = min(col + 1, LADO - 1)
    nuevo = fila * LADO + col
    return nuevo, -1.0, (nuevo == META)


def softmax_filas(z):
    """Softmax numericamente estable aplicado por filas (z de forma (B, A))."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def genera_episodio(theta, w, rng, max_pasos=80):
    """Juega un episodio con pi_theta. Devuelve arrays alineados por paso:
    estados, acciones, recompensas, logp_old (log-prob en el momento de actuar),
    ademas del estado final y si termino de verdad (llego a la meta)."""
    s = INICIO
    estados, acciones, recompensas, logp_old = [], [], [], []
    terminado = False
    for _ in range(max_pasos):
        logits = theta[s]                        # phi(s) one-hot -> fila s
        p = softmax_filas(logits[None, :])[0]
        a = rng.choice(N_ACCIONES, p=p)
        s2, r, terminado = paso(s, a)
        estados.append(s)
        acciones.append(a)
        recompensas.append(r)
        logp_old.append(np.log(p[a] + 1e-12))
        s = s2
        if terminado:
            break
    return (np.array(estados), np.array(acciones), np.array(recompensas),
            np.array(logp_old), s, terminado)


def ventajas_gae(estados, recompensas, w, s_final, terminado, gamma, lam):
    """GAE por episodio. Devuelve (ventajas, retornos_objetivo_del_critico).

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma*lambda * A_{t+1}
    Si el episodio termina de verdad, V(s_final)=0; si se corto por el tope de
    pasos, arrancamos con el bootstrap V(s_final) del critico.
    """
    T = len(recompensas)
    valores = w[estados]                         # V(s_t) para cada paso
    adv = np.zeros(T)
    siguiente_v = 0.0 if terminado else w[s_final]
    acc = 0.0
    for t in reversed(range(T)):
        delta = recompensas[t] + gamma * siguiente_v - valores[t]
        acc = delta + gamma * lam * acc
        adv[t] = acc
        siguiente_v = valores[t]
    retornos = adv + valores                     # objetivo de regresion del critico
    return adv, retornos


def entropia_por_fila(p, logp):
    """H(pi(.|s)) = -sum_a p_a log p_a para cada fila (forma (B,))."""
    return -(p * logp).sum(axis=1)


def entrena(n_iteraciones=160, episodios_por_iter=8, epocas=6,
            lr_pi=0.20, lr_v=0.30, gamma=0.99, lam=0.95,
            eps=0.2, c2=0.01, semilla=0):
    """Bucle de PPO. Devuelve (historial_de_retornos, theta)."""
    rng = np.random.default_rng(semilla)
    theta = np.zeros((N_ESTADOS, N_ACCIONES))    # actor (preferencias)
    w = np.zeros(N_ESTADOS)                       # critico V_w(s)
    identidad = np.eye(N_ACCIONES)                # filas = one-hot de cada accion
    historial = np.zeros(n_iteraciones)

    for it in range(n_iteraciones):
        # ---------- 1) Recogida: varios episodios completos ----------
        b_s, b_a, b_logp, b_adv, b_ret = [], [], [], [], []
        retornos_ep = []
        for _ in range(episodios_por_iter):
            s, a, r, logp, s_fin, term = genera_episodio(theta, w, rng)
            adv, ret = ventajas_gae(s, r, w, s_fin, term, gamma, lam)
            b_s.append(s); b_a.append(a); b_logp.append(logp)
            b_adv.append(adv); b_ret.append(ret)
            retornos_ep.append(r.sum())
        b_s = np.concatenate(b_s)
        b_a = np.concatenate(b_a)
        b_logp = np.concatenate(b_logp)
        b_adv = np.concatenate(b_adv)
        b_ret = np.concatenate(b_ret)
        historial[it] = np.mean(retornos_ep)

        # Normalizamos las ventajas (media 0, desviacion 1): estabiliza el paso.
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        onehot_a = identidad[b_a]                 # (B, A)

        # ---------- 2) Optimizacion: K epocas sobre el MISMO lote ----------
        for _ in range(epocas):
            logits = theta[b_s]                   # (B, A)
            p = softmax_filas(logits)             # pi_theta(.|s) actual
            logp = np.log(p + 1e-12)
            logp_a = logp[np.arange(len(b_a)), b_a]
            ratio = np.exp(logp_a - b_logp)       # r_t(theta)

            # --- Gradiente del objetivo recortado ---
            # d/dtheta [r_t A_t] = A_t * r_t * grad_log_pi.  El recorte anula el
            # gradiente solo cuando empujar mas ya no aporta (region plana):
            #   A>=0 y r>1+eps   ->  gradiente 0
            #   A<0  y r<1-eps   ->  gradiente 0
            plano = ((b_adv >= 0) & (ratio > 1 + eps)) | \
                    ((b_adv < 0) & (ratio < 1 - eps))
            coef = np.where(plano, 0.0, b_adv * ratio)     # (B,)
            # grad_log pi(a|s) = phi(s) x (e_a - p);  con phi one-hot, la fila s.
            g_pi_filas = coef[:, None] * (onehot_a - p)    # (B, A)

            # --- Gradiente del bonus de entropia (fomenta explorar) ---
            H = entropia_por_fila(p, logp)                 # (B,)
            g_ent_filas = -p * (logp + H[:, None])         # dH/dlogits

            # Acumulamos por estado (varios pasos pueden caer en el mismo estado).
            grad_actor = np.zeros_like(theta)
            np.add.at(grad_actor, b_s, g_pi_filas + c2 * g_ent_filas)
            theta += lr_pi * grad_actor / len(b_s)         # ASCENSO de gradiente

            # --- Critico: descenso sobre 0.5 (V_w(s) - retorno)^2 ---
            v = w[b_s]
            err = v - b_ret                                # (B,)
            grad_critico = np.zeros_like(w)
            np.add.at(grad_critico, b_s, err)
            w -= lr_v * grad_critico / len(b_s)            # DESCENSO de gradiente

        if (it + 1) % 20 == 0:
            print(f"Iteracion {it + 1:3d} | retorno medio = {historial[it]:7.2f}"
                  f"   (optimo = {OPTIMO})")

    return historial, theta


def camino_voraz(theta, max_pasos=80):
    """Sigue la politica de forma voraz (argmax) y cuenta pasos hasta la meta."""
    s, pasos = INICIO, 0
    while s != META and pasos < max_pasos:
        a = int(np.argmax(theta[s]))
        s, _, _ = paso(s, a)
        pasos += 1
    return pasos


def media_movil(x, ventana=10):
    nucleo = np.ones(ventana) / ventana
    return np.convolve(x, nucleo, mode="valid")


if __name__ == "__main__":
    historial, theta = entrena()

    print(f"\nCamino mas corto posible: {-OPTIMO} pasos (retorno {OPTIMO}).")
    print(f"Pasos de la politica aprendida (voraz): {camino_voraz(theta)}.")

    # --- Grafica del retorno por iteracion ---
    plt.figure(figsize=(8, 4.6))
    plt.plot(historial, color="#c7d2fe", linewidth=1.2,
             label="Retorno por iteracion")
    suave = media_movil(historial, 10)
    plt.plot(np.arange(len(suave)) + 9, suave, color="#4f46e5",
             linewidth=2.4, label="Media movil (10 iteraciones)")
    plt.axhline(OPTIMO, color="#059669", linestyle="--", linewidth=1.3,
                label=f"Optimo ({OPTIMO})")
    plt.xlabel("Iteracion de PPO")
    plt.ylabel("Retorno medio por episodio")
    plt.title("PPO desde cero: aprendizaje en un gridworld 5x5")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
