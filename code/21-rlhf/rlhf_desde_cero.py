"""
RLHF de juguete DESDE CERO con numpy.

Reproduce, en miniatura, el mismo esqueleto que alinea modelos como ChatGPT,
pero sobre un "bandido" de K respuestas candidatas a un mismo prompt:

  1) PREFERENCIAS SINTÉTICAS. Existe una recompensa "verdadera" r*(y) oculta
     (la utilidad real para el humano). No la conocemos: solo observamos
     COMPARACIONES ruidosas y_i > y_j muestreadas con el modelo de
     Bradley-Terry  P(y_i > y_j) = sigmoid(r*(y_i) - r*(y_j)).

  2) MODELO DE RECOMPENSA. Ajustamos r_theta(y) por máxima verosimilitud sobre
     esas comparaciones (descenso de gradiente de la pérdida de Bradley-Terry).
     Recuperamos r* salvo una constante aditiva... pero con ERROR de estimación.

  3) OPTIMIZACIÓN DE POLÍTICA CON PENALIZACIÓN KL. Buscamos una política
     pi(y) que maximice   E_pi[r_theta(y)] - beta * KL(pi || pi_ref)
     respecto a una política de referencia pi_ref (el "modelo base"). En este
     bandido la política se optimiza por ascenso de gradiente (el análogo de
     PPO) y coincide con la solución cerrada  pi_beta ∝ pi_ref * exp(r_theta/beta),
     la misma que explota DPO.

La moraleja (y la gráfica): si quitamos la penalización KL (beta -> 0) la
política COLAPSA sobre el argmax de la recompensa APRENDIDA. Como ese máximo
suele estar sobrestimado (sesgo de maximización, igual que en Q-learning), a
partir de cierta KL seguir optimizando ya no compra recompensa REAL. En el
mundo de la semilla 0 la recompensa real incluso CAE; por eso el programa
repite el experimento entero en 20 mundos distintos y dibuja la media con su
banda: promediando, la caída no está garantizada (la mediana de la caída es
prácticamente cero), pero el ESTANCAMIENTO sí. Eso es la sobre-optimización
("reward hacking"), y beta es el mando que decide cuánto te acercas al filo.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/21-rlhf/rlhf_desde_cero.py

Solo necesita numpy y matplotlib, así que también corre en el navegador (Pyodide).
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Sigmoide numéricamente estable, sirve para escalares y arrays."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def softmax(logits):
    """Softmax estable: convierte 'logits' en una distribución de probabilidad."""
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p, q):
    """KL(p || q) entre dos distribuciones discretas (con protección de log 0)."""
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def genera_preferencias(r_true, n_pares, rng):
    """Muestrea comparaciones humanas sintéticas con el modelo de Bradley-Terry.

    Devuelve dos arrays de índices (ganadores, perdedores): en cada comparación
    presentamos un par (i, j) al azar y el 'humano' prefiere i con probabilidad
    sigmoid(r*(i) - r*(j)).
    """
    K = len(r_true)
    i = rng.integers(0, K, size=n_pares)
    j = rng.integers(0, K, size=n_pares)
    distinto = i != j                       # descartamos pares (i, i)
    i, j = i[distinto], j[distinto]
    p_i_gana = sigmoid(r_true[i] - r_true[j])
    i_gana = rng.random(len(i)) < p_i_gana
    ganadores = np.where(i_gana, i, j)
    perdedores = np.where(i_gana, j, i)
    return ganadores, perdedores


def entrena_modelo_recompensa(ganadores, perdedores, K, lr=0.5, n_iter=3000):
    """Ajusta r_theta por máxima verosimilitud (pérdida de Bradley-Terry).

    Minimiza  L(theta) = -mean log sigmoid(r_theta(gan) - r_theta(perd))
    por descenso de gradiente de lote completo. La recompensa solo está
    identificada salvo una constante aditiva, así que centramos theta en cada
    paso (restarle la media no altera ninguna diferencia r_i - r_j).
    """
    theta = np.zeros(K)
    M = len(ganadores)
    historia_perdida = []
    for _ in range(n_iter):
        margen = theta[ganadores] - theta[perdedores]        # r(gan) - r(perd)
        # dL/d(margen) = -(1 - sigmoid(margen)) = -sigmoid(-margen)
        coef = -sigmoid(-margen) / M
        grad = np.zeros(K)
        np.add.at(grad, ganadores, coef)                     # +1 en el ganador
        np.add.at(grad, perdedores, -coef)                   # -1 en el perdedor
        theta -= lr * grad
        theta -= theta.mean()                                # fija la constante
        historia_perdida.append(float(-np.mean(np.log(sigmoid(margen) + 1e-12))))
    return theta, historia_perdida


def politica_optima_kl(theta, log_ref, beta):
    """Solución cerrada del objetivo RLHF: pi_beta ∝ pi_ref * exp(r_theta/beta)."""
    return softmax(log_ref + theta / beta)


def optimiza_politica_ascenso(theta, ref, beta, lr=0.5, n_iter=400):
    """Optimiza la política por ASCENSO DE GRADIENTE (el análogo de PPO).

    Maximiza  J(pi) = E_pi[r_theta] - beta * KL(pi || pi_ref) partiendo de la
    referencia. En este bandido podemos calcular el objetivo y su gradiente en
    forma exacta (sin muestrear), pero el espíritu es el de PPO: mover la
    política hacia la recompensa sin alejarse de la referencia. Registra la
    recompensa esperada y la KL en cada iteración.
    """
    log_ref = np.log(ref + 1e-12)
    phi = log_ref.copy()                       # empezamos EN la referencia
    hist_recompensa, hist_kl = [], []
    for _ in range(n_iter):
        p = softmax(phi)
        # g_k = d(objetivo)/dp_k  (salvo constante) = r_k - beta*log(p_k/ref_k)
        g = theta - beta * (np.log(p + 1e-12) - log_ref)
        grad = p * (g - p @ g)                 # regla de la cadena a través del softmax
        phi = phi + lr * grad
        hist_recompensa.append(float(p @ theta))
        hist_kl.append(kl(p, ref))
    return softmax(phi), hist_recompensa, hist_kl


BETAS = np.logspace(np.log10(4.0), np.log10(0.02), 60)   # de mucho freno a ninguno
N_MUNDOS = 20                                            # semillas 0..19


def simula_mundo(semilla, K=28, n_pares=220):
    """Ejecuta el pipeline entero de RLHF en UN mundo (una semilla).

    Mundo = una recompensa oculta r*, unas preferencias sintéticas, el
    r_theta ajustado sobre ellas y el barrido de beta. Devolvemos un dict para
    poder repetirlo en muchas semillas y no sacar conclusiones de una sola
    ejecución (la regla de oro del capítulo 22).
    """
    rng = np.random.default_rng(semilla)

    # 1) El mundo: K respuestas candidatas con una recompensa real oculta.
    r_true = rng.normal(0.0, 0.8, size=K)      # utilidad REAL para el humano (oculta)
    r_true -= r_true.mean()
    ref = np.ones(K) / K                       # modelo base: reparte por igual
    log_ref = np.log(ref)

    # 2) Preferencias sintéticas y modelo de recompensa (Bradley-Terry).
    # Pocas comparaciones: el modelo aprende r* bien, pero con ERROR de estimación,
    # justo el ingrediente que hace posible la sobre-optimización.
    ganadores, perdedores = genera_preferencias(r_true, n_pares=n_pares, rng=rng)
    theta, hist_perdida = entrena_modelo_recompensa(ganadores, perdedores, K)

    # 3) Barrido de beta: recompensa real vs. aprendida en función de la KL.
    kls, r_real, r_aprend = [], [], []
    for b in BETAS:
        pi_b = politica_optima_kl(theta, log_ref, b)
        kls.append(kl(pi_b, ref))
        r_real.append(float(pi_b @ r_true))
        r_aprend.append(float(pi_b @ theta))

    return dict(r_true=r_true, ref=ref, log_ref=log_ref, theta=theta,
                n_comparaciones=len(ganadores), hist_perdida=hist_perdida,
                kls=np.array(kls), r_real=np.array(r_real),
                r_aprend=np.array(r_aprend))


def promedia_mundos(mundos):
    """Pone los mundos en una rejilla común de KL y devuelve las curvas apiladas.

    Cada mundo llega a una KL máxima distinta, así que interpolamos sobre la
    rejilla más ancha que TODOS cubren: así promediamos sin extrapolar nada.
    Por la izquierda añadimos a mano el punto exacto de KL = 0 (beta -> infinito:
    la política ES la referencia), que conocemos sin simularlo.
    """
    kl_max = min(m["kls"][-1] for m in mundos)
    rejilla = np.linspace(0.0, kl_max, 60)

    def curva(m, y, y_en_cero):
        return np.interp(rejilla, np.r_[0.0, m["kls"]], np.r_[y_en_cero, y])

    real = np.array([curva(m, m["r_real"], m["ref"] @ m["r_true"]) for m in mundos])
    aprend = np.array([curva(m, m["r_aprend"], m["ref"] @ m["theta"]) for m in mundos])
    return rejilla, real, aprend


def main():
    mundos = [simula_mundo(s) for s in range(N_MUNDOS)]
    m0 = mundos[0]                             # el mundo que miramos con lupa
    r_true, ref, log_ref, theta = m0["r_true"], m0["ref"], m0["log_ref"], m0["theta"]
    K = len(r_true)

    corr = float(np.corrcoef(r_true, theta)[0, 1])
    print("=== Modelo de recompensa (Bradley-Terry), mundo 0 ===")
    print(f"Comparaciones usadas          : {m0['n_comparaciones']}")
    print(f"Pérdida BT inicial -> final   : {m0['hist_perdida'][0]:.4f} -> {m0['hist_perdida'][-1]:.4f}")
    print(f"Correlación r_theta vs r*     : {corr:.3f}")
    mejor_real = int(np.argmax(r_true))
    mejor_estim = int(np.argmax(theta))
    print(f"Mejor respuesta REAL           : #{mejor_real}  (r* = {r_true[mejor_real]:+.2f})")
    print(f"Mejor respuesta ESTIMADA       : #{mejor_estim}  "
          f"(r_theta = {theta[mejor_estim]:+.2f}, pero r* = {r_true[mejor_estim]:+.2f})")

    # --- Optimización de la política con penalización KL (análogo de PPO) ---
    beta_demo = 0.5
    pi_ascenso, _, _ = optimiza_politica_ascenso(theta, ref, beta_demo)
    pi_cerrada = politica_optima_kl(theta, log_ref, beta_demo)
    print(f"\n=== Política KL-regularizada (beta = {beta_demo}) ===")
    print(f"Diferencia máx. ascenso vs solución cerrada : {np.max(np.abs(pi_ascenso - pi_cerrada)):.2e}")
    print(f"Recompensa real  E[r*]  ref -> pi_beta       : "
          f"{ref @ r_true:+.3f} -> {pi_ascenso @ r_true:+.3f}")
    print(f"Divergencia KL(pi_beta || pi_ref)            : {kl(pi_ascenso, ref):.3f}")

    # --- Sobre-optimización en el mundo 0 (UNA sola semilla) ---
    i_pico = int(np.argmax(m0["r_real"]))      # dónde la recompensa REAL es máxima
    beta_pico = BETAS[i_pico]
    pi_pico = politica_optima_kl(theta, log_ref, beta_pico)
    pi_codicioso = politica_optima_kl(theta, log_ref, BETAS[-1])   # beta -> 0: colapso

    print("\n=== Sobre-optimización en el mundo 0 (una sola semilla) ===")
    print(f"E[r*] con el modelo base (ref)          : {ref @ r_true:+.3f}")
    print(f"E[r*] en el pico (beta ~ {beta_pico:.2f})          : {pi_pico @ r_true:+.3f}")
    print(f"E[r*] sin freno KL (beta -> 0, colapso) : {pi_codicioso @ r_true:+.3f}")
    print(f"El óptimo REAL alcanzable                : {r_true.max():+.3f}")

    # --- ...y lo mismo en 20 mundos: ¿es una ley o es la semilla? ---
    rejilla, real, aprend = promedia_mundos(mundos)
    media, desv = real.mean(axis=0), real.std(axis=0)
    caidas = np.array([m["r_real"].max() - m["r_real"][-1] for m in mundos])
    medio = len(rejilla) // 2
    print(f"\n=== ...y ahora en {N_MUNDOS} mundos (semillas 0..{N_MUNDOS - 1}) ===")
    print(f"Caída pico -> sin freno, por mundo   : mediana {np.median(caidas):.3f}, "
          f"media {caidas.mean():.3f}")
    print(f"Mundos que caen más de 0,05          : {(caidas > 0.05).sum()} de {N_MUNDOS}")
    print(f"E[r*] MEDIA, KL 0 -> {rejilla[medio]:.2f} -> {rejilla[-1]:.2f}    : "
          f"{media[0]:+.3f} -> {media[medio]:+.3f} -> {media[-1]:+.3f}")
    print(f"r_theta MEDIA en el mismo recorrido  : "
          f"{aprend.mean(axis=0)[0]:+.3f} -> {aprend.mean(axis=0)[medio]:+.3f} "
          f"-> {aprend.mean(axis=0)[-1]:+.3f}")
    print(f"Desviación entre mundos al final     : {desv[-1]:.3f}")

    # --- Gráfica: ajuste del modelo + sobre-optimización + colapso ---
    IND, CIAN, VERDE, AMBAR, ROJO, GRIS = (
        "#4f46e5", "#0ea5e9", "#059669", "#d97706", "#dc2626", "#94a3b8")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (A) Ajuste del modelo de recompensa: estimado vs. real (ambos centrados).
    ax1.axline((0, 0), slope=1, color=GRIS, ls="--", lw=1, label="ajuste perfecto")
    ax1.scatter(r_true, theta, color=IND, s=45, zorder=3)
    ax1.set_xlabel("Recompensa REAL  r*(y)  (oculta)")
    ax1.set_ylabel("Recompensa APRENDIDA  r_theta(y)")
    ax1.set_title(f"1) El modelo de recompensa\naprende r*  (corr = {corr:.2f})")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    # (B) Sobre-optimización: al aumentar la KL, la recompensa aprendida sube
    #     sin freno; la REAL se estanca y, según el mundo, además cae.
    ax2.plot(rejilla, aprend.mean(axis=0), color=AMBAR, lw=2,
             label=f"APRENDIDA, media de {N_MUNDOS} mundos")
    ax2.fill_between(rejilla, media - desv, media + desv, color=IND, alpha=0.18,
                     label="REAL, ±1 desviación entre mundos")
    ax2.plot(rejilla, media, color=IND, lw=2.2, label=f"REAL, media de {N_MUNDOS} mundos")
    ax2.plot(m0["kls"], m0["r_real"], color=IND, lw=1.2, ls="--", alpha=0.75,
             label="REAL, mundo 0 (una sola semilla)")
    ax2.scatter([m0["kls"][i_pico]], [m0["r_real"][i_pico]], color=VERDE, s=55, zorder=4,
                label=f"pico del mundo 0 (beta ~ {beta_pico:.2f})")
    ax2.set_xlabel("KL(pi || pi_ref)   (más optimización -->)")
    ax2.set_ylabel("Recompensa esperada")
    ax2.set_title("2) Sobre-optimización: la recompensa REAL\nse estanca (y a veces cae)")
    ax2.legend(loc="upper left", fontsize=7.5)
    ax2.grid(alpha=0.25)

    # (C) Las políticas del mundo 0: sin KL, todo el peso cae en UNA respuesta.
    x = np.arange(K)
    ax3.bar(x - 0.27, ref, width=0.27, color=GRIS, label="pi_ref (modelo base)")
    ax3.bar(x, pi_pico, width=0.27, color=IND, label="pi_beta (KL del pico)")
    ax3.bar(x + 0.27, pi_codicioso, width=0.27, color=ROJO, label="pi sin KL (colapso)")
    ax3.set_xlabel("Respuesta candidata  y")
    ax3.set_ylabel("Probabilidad  pi(y)")
    ax3.set_title("3) La KL evita el colapso\nsobre una sola respuesta")
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
