"""
REINFORCE desde cero (solo NumPy + Matplotlib)
==============================================

Implementa el algoritmo REINFORCE con una politica softmax lineal y una
linea base (baseline) de valor aprendida, sobre un pequeno gridworld 4x4.

La politica es pi_theta(a|s) = softmax_a( phi(s) . theta[:, a] ), donde phi(s)
son las "features" del estado (aqui, codificacion one-hot). El baseline es una
funcion de valor lineal V_w(s) = phi(s) . w que se aprende por regresion contra
el retorno muestreado. Al restar V_w(s) al retorno reducimos la varianza del
gradiente sin introducir sesgo.

Que veras al ejecutarlo:
  - Por consola, el retorno medio va creciendo (se hace menos negativo) segun
    el agente aprende el camino corto hasta la meta.
  - Una grafica con el retorno por episodio y su media movil, claramente
    creciente.

Como ejecutarlo en tu terminal:
  pip install -r requirements.txt
  python code/14-policy-gradient-reinforce/reinforce_desde_cero.py
"""

import numpy as np
import matplotlib.pyplot as plt


# --- El entorno: gridworld 4x4 (sin obstaculos) ---------------------------
# Estados: 0..15 (fila = s // 4, columna = s % 4). Inicio: 0 (arriba-izq).
# Meta: 15 (abajo-der). Acciones: 0=arriba 1=abajo 2=izquierda 3=derecha.
# Recompensa: -1 por cada paso (queremos el camino mas corto).
LADO = 4
N_ESTADOS = LADO * LADO
N_ACCIONES = 4
INICIO = 0
META = N_ESTADOS - 1


def paso(estado, accion):
    """Devuelve (nuevo_estado, recompensa, terminado)."""
    fila, col = divmod(estado, LADO)
    if accion == 0:                      # arriba
        fila = max(fila - 1, 0)
    elif accion == 1:                    # abajo
        fila = min(fila + 1, LADO - 1)
    elif accion == 2:                    # izquierda
        col = max(col - 1, 0)
    else:                                # derecha
        col = min(col + 1, LADO - 1)
    nuevo = fila * LADO + col
    terminado = (nuevo == META)
    return nuevo, -1.0, terminado


def features(estado):
    """phi(s): codificacion one-hot del estado (vector de longitud N_ESTADOS)."""
    phi = np.zeros(N_ESTADOS)
    phi[estado] = 1.0
    return phi


def softmax(z):
    z = z - np.max(z)                    # estabilidad numerica
    e = np.exp(z)
    return e / np.sum(e)


def probs_politica(theta, phi):
    """pi_theta(.|s): distribucion sobre acciones dado phi(s)."""
    logits = phi @ theta                 # forma (N_ACCIONES,)
    return softmax(logits)


def genera_episodio(theta, rng, max_pasos=60):
    """Ejecuta un episodio siguiendo pi_theta. Devuelve listas s, a, r."""
    estado = INICIO
    estados, acciones, recompensas = [], [], []
    for _ in range(max_pasos):
        phi = features(estado)
        p = probs_politica(theta, phi)
        accion = rng.choice(N_ACCIONES, p=p)
        nuevo, r, terminado = paso(estado, accion)
        estados.append(estado)
        acciones.append(accion)
        recompensas.append(r)
        estado = nuevo
        if terminado:
            break
    return estados, acciones, recompensas


def retornos_descontados(recompensas, gamma):
    """G_t = r_t + gamma*r_{t+1} + ... calculado hacia atras."""
    G = np.zeros(len(recompensas))
    acc = 0.0
    for t in reversed(range(len(recompensas))):
        acc = recompensas[t] + gamma * acc
        G[t] = acc
    return G


def entrena(n_episodios=600, alpha=0.02, beta=0.10, gamma=0.99, semilla=0):
    """REINFORCE con baseline. Devuelve el retorno total por episodio."""
    rng = np.random.default_rng(semilla)
    theta = np.zeros((N_ESTADOS, N_ACCIONES))   # preferencias de la politica
    w = np.zeros(N_ESTADOS)                      # pesos del baseline V_w(s)
    historial = np.zeros(n_episodios)

    for ep in range(n_episodios):
        estados, acciones, recompensas = genera_episodio(theta, rng)
        G = retornos_descontados(recompensas, gamma)
        historial[ep] = np.sum(recompensas)     # retorno sin descuento = -pasos

        for t, (s, a) in enumerate(zip(estados, acciones)):
            phi = features(s)
            p = probs_politica(theta, phi)
            v = phi @ w                          # prediccion del baseline
            delta = G[t] - v                     # ventaja estimada (advantage)

            # 1) Ajusta el baseline hacia el retorno observado
            w += beta * delta * phi

            # 2) Ascenso de gradiente de la politica (truco del log-derivada):
            #    grad log pi(a|s) = phi(s) x (one_hot(a) - pi(.|s))
            one_hot_a = np.zeros(N_ACCIONES)
            one_hot_a[a] = 1.0
            grad_log = np.outer(phi, one_hot_a - p)
            theta += alpha * (gamma ** t) * delta * grad_log

        if (ep + 1) % 50 == 0:
            media = np.mean(historial[max(0, ep - 49):ep + 1])
            print(f"Episodio {ep + 1:4d} | retorno medio (ult. 50) = {media:6.2f}")

    return historial, theta


def camino_voraz(theta, max_pasos=60):
    """Sigue la politica de forma voraz (argmax) y cuenta los pasos hasta la meta."""
    estado, pasos = INICIO, 0
    while estado != META and pasos < max_pasos:
        accion = int(np.argmax(probs_politica(theta, features(estado))))
        estado, _, _ = paso(estado, accion)
        pasos += 1
    return pasos


def media_movil(x, ventana=25):
    nucleo = np.ones(ventana) / ventana
    return np.convolve(x, nucleo, mode="valid")


if __name__ == "__main__":
    historial, theta = entrena()

    optimo = 2 * (LADO - 1)              # camino mas corto en el grid 4x4 = 6
    print(f"\nCamino mas corto posible: {optimo} pasos.")
    print(f"Pasos de la politica aprendida (voraz): {camino_voraz(theta)}.")

    # --- Grafica del retorno por episodio ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(historial, color="#c7d2fe", linewidth=1, label="Retorno por episodio")
    suave = media_movil(historial, 25)
    plt.plot(np.arange(len(suave)) + 24, suave, color="#4f46e5",
             linewidth=2.2, label="Media movil (25 episodios)")
    plt.axhline(-optimo, color="#059669", linestyle="--", linewidth=1.2,
                label=f"Optimo ({-optimo})")
    plt.xlabel("Episodio")
    plt.ylabel("Retorno total (= -pasos)")
    plt.title("REINFORCE con baseline en un gridworld 4x4")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
