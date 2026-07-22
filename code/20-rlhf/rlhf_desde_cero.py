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
está sobrestimado (sesgo de maximización, igual que en Q-learning), la
recompensa REAL sube, alcanza un pico y luego CAE: es la sobre-optimización
("reward hacking"). La KL mantiene la política cerca del modelo base y evita el
colapso.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/20-rlhf/rlhf_desde_cero.py

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


def main():
    rng = np.random.default_rng(0)

    # --- 1) El mundo: K respuestas candidatas con una recompensa real oculta ---
    K = 28                                     # número de "respuestas" al prompt
    r_true = rng.normal(0.0, 0.8, size=K)      # utilidad REAL para el humano (oculta)
    r_true -= r_true.mean()
    ref = np.ones(K) / K                       # modelo base: reparte por igual
    log_ref = np.log(ref)

    # --- 2) Preferencias sintéticas y modelo de recompensa (Bradley-Terry) ---
    # Pocas comparaciones: el modelo aprende r* bien, pero con ERROR de estimación,
    # justo el ingrediente que hace posible la sobre-optimización.
    ganadores, perdedores = genera_preferencias(r_true, n_pares=220, rng=rng)
    theta, hist_perdida = entrena_modelo_recompensa(ganadores, perdedores, K)

    corr = float(np.corrcoef(r_true, theta)[0, 1])
    print("=== Modelo de recompensa (Bradley-Terry) ===")
    print(f"Comparaciones usadas          : {len(ganadores)}")
    print(f"Pérdida BT inicial -> final   : {hist_perdida[0]:.4f} -> {hist_perdida[-1]:.4f}")
    print(f"Correlación r_theta vs r*     : {corr:.3f}")
    mejor_real = int(np.argmax(r_true))
    mejor_estim = int(np.argmax(theta))
    print(f"Mejor respuesta REAL           : #{mejor_real}  (r* = {r_true[mejor_real]:+.2f})")
    print(f"Mejor respuesta ESTIMADA       : #{mejor_estim}  "
          f"(r_theta = {theta[mejor_estim]:+.2f}, pero r* = {r_true[mejor_estim]:+.2f})")

    # --- 3) Optimización de la política con penalización KL (análogo de PPO) ---
    beta_demo = 0.5
    pi_ascenso, hist_rec, hist_kl = optimiza_politica_ascenso(theta, ref, beta_demo)
    pi_cerrada = politica_optima_kl(theta, log_ref, beta_demo)
    print(f"\n=== Política KL-regularizada (beta = {beta_demo}) ===")
    print(f"Diferencia máx. ascenso vs solución cerrada : {np.max(np.abs(pi_ascenso - pi_cerrada)):.2e}")
    print(f"Recompensa real  E[r*]  ref -> pi_beta       : "
          f"{ref @ r_true:+.3f} -> {pi_ascenso @ r_true:+.3f}")
    print(f"Divergencia KL(pi_beta || pi_ref)            : {kl(pi_ascenso, ref):.3f}")

    # --- 4) Barrido de beta: recompensa real vs. aprendida en función de la KL ---
    betas = np.logspace(np.log10(4.0), np.log10(0.02), 60)
    kls, r_real, r_aprend = [], [], []
    for b in betas:
        pi_b = politica_optima_kl(theta, log_ref, b)
        kls.append(kl(pi_b, ref))
        r_real.append(pi_b @ r_true)
        r_aprend.append(pi_b @ theta)
    kls, r_real, r_aprend = map(np.array, (kls, r_real, r_aprend))

    i_pico = int(np.argmax(r_real))            # dónde la recompensa REAL es máxima
    beta_pico = betas[i_pico]
    pi_pico = politica_optima_kl(theta, log_ref, beta_pico)
    pi_codicioso = politica_optima_kl(theta, log_ref, betas[-1])   # beta -> 0: colapso

    print("\n=== Sobre-optimización (reward hacking) ===")
    print(f"E[r*] con el modelo base (ref)          : {ref @ r_true:+.3f}")
    print(f"E[r*] con KL óptima (beta ~ {beta_pico:.2f})     : {pi_pico @ r_true:+.3f}")
    print(f"E[r*] sin freno KL (beta -> 0, colapso) : {pi_codicioso @ r_true:+.3f}")
    print(f"El óptimo REAL alcanzable                : {r_true.max():+.3f}")

    # --- 5) Gráfica: ajuste del modelo + sobre-optimización + colapso ---
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
    #     sin freno, pero la REAL sube, hace pico y colapsa.
    ax2.plot(kls, r_aprend, color=AMBAR, lw=2, label="recompensa APRENDIDA (lo que optimizamos)")
    ax2.plot(kls, r_real, color=IND, lw=2, label="recompensa REAL (lo que importa)")
    ax2.axvline(kls[i_pico], color=VERDE, ls="--", lw=1.2)
    ax2.scatter([kls[i_pico]], [r_real[i_pico]], color=VERDE, s=60, zorder=4,
                label=f"KL óptima (beta ~ {beta_pico:.2f})")
    ax2.set_xlabel("KL(pi || pi_ref)   (más optimización -->)")
    ax2.set_ylabel("Recompensa esperada")
    ax2.set_title("2) Sobre-optimización:\nla recompensa REAL colapsa")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(alpha=0.25)

    # (C) Las políticas: sin KL, todo el peso cae en UNA respuesta (colapso).
    x = np.arange(K)
    ax3.bar(x - 0.27, ref, width=0.27, color=GRIS, label="pi_ref (modelo base)")
    ax3.bar(x, pi_pico, width=0.27, color=IND, label=f"pi_beta (KL óptima)")
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
