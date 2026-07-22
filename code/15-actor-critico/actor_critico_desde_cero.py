"""
Actor-Crítico de un paso (A2C) DESDE CERO con numpy.

Entrenamos dos agentes sobre el mismo entorno (un pasillo con dos metas y una
recompensa RUIDOSA) para comparar la VARIANZA de sus curvas de aprendizaje:

  * A2C  : actor softmax lineal + crítico lineal. Usa el error TD
           delta = r + gamma*V(s') - V(s) como estimador de la ventaja.
           Solo arrastra el ruido de UN paso por actualización.
  * REINFORCE : el mismo actor, pero SIN crítico; usa el retorno Monte Carlo
                G_t completo (sin baseline), que acumula el ruido de TODA la
                trayectoria.

Para que la comparación sea justa, el entrenamiento usa la recompensa ruidosa,
pero la métrica que dibujamos es el retorno LIMPIO (sin ruido): así la curva
refleja la calidad de la política, no el azar de las recompensas. El crítico de
A2C aprende a predecir ese ruido y lo cancela vía la ventaja, de modo que su
curva sube antes y con bandas mucho más estrechas.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/15-actor-critico/actor_critico_desde_cero.py

Solo necesita numpy y matplotlib, así que también corre en el navegador (Pyodide).
"""
import numpy as np
import matplotlib.pyplot as plt

# --- El entorno: un pasillo de 11 casillas [0..10] con dos metas ---
# La casilla 0 (izquierda) da +0.2 y la casilla 10 (derecha) da +1.0.
# Empezamos en el centro (5): ambas metas están a 5 pasos, pero la derecha
# premia 5 veces más, así que la política óptima es ir siempre a la derecha.
# A cada paso le sumamos ruido gaussiano: imita un entorno con recompensa
# estocástica, el escenario donde un crítico (baseline) marca la diferencia.
N = 11           # número de casillas
INICIO = 5       # casilla de partida (centro)
IZQ, DER = 0, 1  # acciones: 0 = izquierda, 1 = derecha
R_IZQ, R_DER = 0.2, 1.0
GAMMA = 0.99
SIGMA = 0.5      # desviación del ruido de recompensa (media cero)
MAX_PASOS = 150  # tope de seguridad por episodio


def paso(s, a):
    """Dinámica del entorno: devuelve (s_siguiente, recompensa_limpia, terminado)."""
    s2 = s + 1 if a == DER else s - 1
    if s2 == 0:
        return s2, R_IZQ, True    # meta izquierda (premio pequeño)
    if s2 == N - 1:
        return s2, R_DER, True    # meta derecha (premio grande)
    return s2, 0.0, False         # casilla intermedia, sin premio


def phi(s):
    """Codificación one-hot del estado: un vector de features lineal."""
    v = np.zeros(N)
    v[s] = 1.0
    return v


def softmax(x):
    """Softmax numéricamente estable."""
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def entrena_a2c(semilla, n_epis, alpha_pi, alpha_v):
    """A2C de un paso. W: pesos del actor (N x 2). wv: pesos del crítico (N)."""
    rng = np.random.default_rng(semilla)
    W = np.zeros((N, 2))
    wv = np.zeros(N)
    retornos = np.zeros(n_epis)
    ident = np.eye(2)  # filas = one-hot de cada acción

    for ep in range(n_epis):
        s = INICIO
        g_limpio, descuento, terminado, pasos = 0.0, 1.0, False, 0
        while not terminado and pasos < MAX_PASOS:
            f = phi(s)
            p = softmax(f @ W)               # política pi(.|s)
            a = rng.choice(2, p=p)
            s2, r_limpia, terminado = paso(s, a)
            r = r_limpia + SIGMA * rng.standard_normal()  # recompensa ruidosa

            # Crítico: estima V(s) y V(s'); el error TD es la ventaja.
            v_s = wv @ f
            v_s2 = 0.0 if terminado else wv @ phi(s2)
            delta = r + GAMMA * v_s2 - v_s

            # Actualización del crítico (descenso sobre el error TD al cuadrado).
            wv += alpha_v * delta * f

            # Actualización del actor (ascenso de gradiente de política).
            # grad log pi(a|s) = phi(s) ⊗ (e_a - p)
            grad_log = np.outer(f, ident[a] - p)
            W += alpha_pi * delta * grad_log

            g_limpio += descuento * r_limpia  # métrica: retorno SIN ruido
            descuento *= GAMMA
            s = s2
            pasos += 1
        retornos[ep] = g_limpio
    return retornos


def entrena_reinforce(semilla, n_epis, alpha_pi):
    """REINFORCE (Monte Carlo, sin baseline): el mismo actor, sin crítico."""
    rng = np.random.default_rng(semilla)
    W = np.zeros((N, 2))
    retornos = np.zeros(n_epis)
    ident = np.eye(2)

    for ep in range(n_epis):
        # 1) Generamos un episodio completo, guardando recompensa ruidosa y limpia.
        estados, acciones, r_ruido, r_limpia = [], [], [], []
        s, terminado, pasos = INICIO, False, 0
        while not terminado and pasos < MAX_PASOS:
            p = softmax(phi(s) @ W)
            a = rng.choice(2, p=p)
            s2, rl, terminado = paso(s, a)
            estados.append(s)
            acciones.append(a)
            r_ruido.append(rl + SIGMA * rng.standard_normal())
            r_limpia.append(rl)
            s = s2
            pasos += 1

        # 2) Retorno Monte Carlo G_t (con ruido) hacia atrás: el actor lo usa entero.
        T = len(r_ruido)
        Gs = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = r_ruido[t] + GAMMA * G
            Gs[t] = G

        # 3) Ascenso de gradiente con el retorno COMPLETO (sin línea base).
        for t in range(T):
            f = phi(estados[t])
            p = softmax(f @ W)
            grad_log = np.outer(f, ident[acciones[t]] - p)
            W += alpha_pi * Gs[t] * grad_log

        # 4) Retorno LIMPIO para la métrica (no interviene en el aprendizaje).
        g_limpio = 0.0
        for t in reversed(range(T)):
            g_limpio = r_limpia[t] + GAMMA * g_limpio
        retornos[ep] = g_limpio
    return retornos


def media_movil(x, w):
    """Media móvil de ventana w (modo 'valid')."""
    nucleo = np.ones(w) / w
    return np.convolve(x, nucleo, mode="valid")


def main():
    n_semillas = 8
    n_epis = 400
    ventana = 15

    # Entrenamos ambos métodos con varias semillas para medir la varianza.
    ac = np.array([entrena_a2c(s, n_epis, alpha_pi=0.15, alpha_v=0.30)
                   for s in range(n_semillas)])
    rf = np.array([entrena_reinforce(100 + s, n_epis, alpha_pi=0.15)
                   for s in range(n_semillas)])

    # Suavizamos cada corrida y agregamos media y desviación entre semillas.
    ac_s = np.array([media_movil(fila, ventana) for fila in ac])
    rf_s = np.array([media_movil(fila, ventana) for fila in rf])
    eje = np.arange(ac_s.shape[1]) + ventana

    ac_mu, ac_sd = ac_s.mean(0), ac_s.std(0)
    rf_mu, rf_sd = rf_s.mean(0), rf_s.std(0)

    # Resumen numérico por consola (últimos 50 episodios).
    # Óptimo = ir siempre a la derecha: la meta está a (N-1-INICIO)=5 pasos y la
    # recompensa R_DER llega en ese último paso, descontada gamma^(pasos-1)=gamma^4.
    optimo = GAMMA ** (N - 2 - INICIO) * R_DER
    print(f"Retorno óptimo aprox. (ir siempre a la derecha): {optimo:.3f}")
    print(f"A2C        -> retorno final medio = {ac[:, -50:].mean():.3f}"
          f"   (banda media entre semillas = {ac_sd.mean():.3f})")
    print(f"REINFORCE  -> retorno final medio = {rf[:, -50:].mean():.3f}"
          f"   (banda media entre semillas = {rf_sd.mean():.3f})")

    # Gráfica comparativa: línea = media entre semillas, banda = ±1 desviación.
    plt.figure(figsize=(8, 4.6))
    plt.plot(eje, rf_mu, color="#d97706", label="REINFORCE (sin baseline)")
    plt.fill_between(eje, rf_mu - rf_sd, rf_mu + rf_sd, color="#d97706", alpha=0.20)
    plt.plot(eje, ac_mu, color="#4f46e5", label="A2C (crítico como baseline)")
    plt.fill_between(eje, ac_mu - ac_sd, ac_mu + ac_sd, color="#4f46e5", alpha=0.20)
    plt.axhline(optimo, color="#059669", ls="--", lw=1, label="óptimo")
    plt.xlabel("Episodio")
    plt.ylabel("Retorno limpio (sin ruido) por episodio")
    plt.title("A2C reduce la varianza frente a REINFORCE")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
