"""
TD(lambda) con trazas de elegibilidad en el paseo aleatorio de 19 estados.
=========================================================================

Capitulo 10 del Manual de Aprendizaje por Refuerzo.

Reproduce el experimento clasico que compara todo el espectro entre TD(0) y
Monte Carlo: aplicamos TD(lambda) (vista hacia atras, con trazas de
elegibilidad acumulativas) al paseo aleatorio de 19 estados, barriendo
lambda en {0, 0.4, 0.8, 0.9, 1.0} y, para cada lambda, la tasa de aprendizaje
alpha. Dibujamos el error RMS (promediado sobre los 10 primeros episodios y
varias repeticiones) frente a alpha. Aparece la tipica familia de curvas en U:
un lambda intermedio suele batir tanto a TD(0) como a Monte Carlo.

Como ejecutarlo:
    pip install -r requirements.txt
    python code/10-td-lambda/td_lambda.py

Solo necesita numpy y matplotlib, asi que tambien corre en el navegador
(Pyodide) desde la pagina del capitulo.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- El entorno: paseo aleatorio de 19 estados ---------------------------
# Estados no terminales: 1..19. Terminales: 0 (izquierda) y 20 (derecha).
# Se empieza en el centro. Al alcanzar la derecha la recompensa es +1; al
# alcanzar la izquierda, -1; el resto de transiciones dan 0. gamma = 1.
N = 19
IZQ, DER = 0, N + 1          # estados terminales
INICIO = (N + 1) // 2        # estado central = 10
GAMMA = 1.0

# Valores verdaderos: para un paseo simetrico, V(s) crece linealmente de
# -0.9 (estado 1) a +0.9 (estado 19), pasando por 0 en el centro.
ESTADOS = np.arange(1, N + 1)
V_TRUE = (ESTADOS - INICIO) / INICIO


def genera_episodio(rng):
    """Simula un paseo aleatorio y devuelve su lista de transiciones (s, r, s')."""
    s = INICIO
    trans = []
    while True:
        s_next = s + (1 if rng.random() < 0.5 else -1)
        if s_next == DER:
            trans.append((s, 1.0, s_next))
            break
        if s_next == IZQ:
            trans.append((s, -1.0, s_next))
            break
        trans.append((s, 0.0, s_next))
        s = s_next
    return trans


def td_lambda_episodio(V, lam, alpha, rng):
    """Ejecuta un episodio de TD(lambda) hacia atras y actualiza V in situ.

    En cada paso: se calcula el error TD delta, se desvanece la traza de
    todos los estados (gamma*lambda), se refuerza la del estado actual
    (traza acumulativa) y se reparte el error entre todos los estados en
    proporcion a su traza.
    """
    e = np.zeros(N + 2)                       # traza de elegibilidad por estado
    for (s, r, s_next) in genera_episodio(rng):
        delta = r + GAMMA * V[s_next] - V[s]  # error de diferencia temporal
        e *= GAMMA * lam                      # todas las trazas se desvanecen
        e[s] += 1.0                           # el estado visitado se hace "elegible"
        V += alpha * delta * e                # actualizacion hacia atras global


def rms(V):
    """Error cuadratico medio de V frente a los valores verdaderos."""
    return np.sqrt(np.mean((V[1:N + 1] - V_TRUE) ** 2))


def experimento(lambdas, alphas, n_runs=20, n_ep=10):
    """Devuelve un dict {lambda: curva de RMS frente a alpha}."""
    curvas = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for lam in lambdas:
            curva = np.zeros(len(alphas))
            for i, alpha in enumerate(alphas):
                acc = 0.0
                for run in range(n_runs):
                    # Mismos episodios para todo (lambda, alpha): comparacion justa.
                    rng = np.random.default_rng(run)
                    V = np.zeros(N + 2)              # sin informacion previa
                    err = 0.0
                    for _ in range(n_ep):
                        td_lambda_episodio(V, lam, alpha, rng)
                        err += rms(V)
                    acc += err / n_ep
                curva[i] = acc / n_runs
            # Los casos que divergen (alpha grande, lambda alto) se marcan NaN.
            curvas[lam] = np.where(np.isfinite(curva), curva, np.nan)
    return curvas


def main():
    lambdas = [0.0, 0.4, 0.8, 0.9, 1.0]
    alphas = np.linspace(0.0, 1.0, 11)
    curvas = experimento(lambdas, alphas)

    # Resumen numerico por pantalla.
    print("Paseo aleatorio de 19 estados - TD(lambda), error RMS minimo:")
    mejor_lam, mejor_rms = None, np.inf
    for lam in lambdas:
        c = curvas[lam]
        j = np.nanargmin(c)
        etiqueta = "Monte Carlo" if lam == 1.0 else ("TD(0)" if lam == 0.0 else "")
        print(f"  lambda = {lam:<3}  min RMS = {c[j]:.3f}  en alpha = {alphas[j]:.2f}  {etiqueta}")
        if c[j] < mejor_rms:
            mejor_rms, mejor_lam = c[j], lam
    print(f"-> El mejor resultado global lo da lambda = {mejor_lam} (un valor intermedio).")

    # Grafica: la familia de curvas en U.
    plt.figure(figsize=(8, 5))
    for lam in lambdas:
        etiqueta = f"lambda = {lam}"
        if lam == 0.0:
            etiqueta += "  (TD(0))"
        elif lam == 1.0:
            etiqueta += "  (Monte Carlo)"
        plt.plot(alphas, curvas[lam], "o-", linewidth=2, markersize=5, label=etiqueta)

    plt.xlabel("alpha  (tasa de aprendizaje)")
    plt.ylabel("Error RMS  (media de los 10 primeros episodios)")
    plt.title("TD(lambda) en el paseo aleatorio de 19 estados")
    plt.ylim(0.25, 0.55)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
