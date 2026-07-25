"""
per_blind_cliffwalk.py - Por que priorizar la experiencia cambia el orden de
magnitud del aprendizaje (Schaul et al., 2015, "Prioritized Experience Replay").

El entorno "Blind Cliffwalk" es un pasillo de n estados con DOS acciones. En cada
estado una de las dos es la buena (se sortea al azar y el agente no la conoce: de
ahi lo de "blind"). La buena avanza al estado siguiente; la mala termina el
episodio con recompensa 0. Solo al salir del ultimo estado por la puerta buena se
cobra +1. Una politica aleatoria acierta la secuencia entera con probabilidad
2^-n: la memoria se llena de basura y contiene UNA sola transicion con recompensa.

Con esa memoria congelada hacemos actualizaciones tabulares de Q-learning
(alpha = 1, entorno determinista, asi que cada actualizacion es exacta) y
comparamos tres formas de elegir que transicion repetir:

  * uniforme    : al azar, como el replay buffer de DQN del capitulo 12.
  * PER         : sorteo con probabilidad proporcional a p_i^ALPHA, con
                  p_i = |error TD| + eps refrescado SOLO cuando la transicion
                  sale sorteada (prioridades rancias incluidas, como en el
                  articulo). Las nuevas entran con la prioridad maxima.
  * oraculo     : siempre la de mayor |error TD| ahora mismo. Cuesta O(N) por
                  actualizacion, es impracticable, y marca el techo teorico.

Ejecucion en tu terminal:
    python code/13-mejoras-dqn/per_blind_cliffwalk.py
"""
import numpy as np
import matplotlib.pyplot as plt

N_MIN, N_MAX = 4, 12   # longitudes del pasillo que barremos
REPETICIONES = 5       # memorias distintas por longitud (el uniforme es ruidoso)
ALPHA = 0.6            # exponente de priorizacion de PER
EPS = 1e-6             # para que ninguna transicion tenga probabilidad cero
TOL = 1e-3             # cuando consideramos que Q ha convergido a Q*
MAX_ACTUALIZACIONES = 2_000_000

# Modos que se miden. Escribe ("uniforme", "per", "oraculo") para medir tambien
# el PER realista; tarda unos 10 segundos mas.
MODOS = ("uniforme", "oraculo")


def construir_memoria(n, rng):
    """Politica aleatoria hasta que UN episodio llega al final y cobra el +1."""
    buena = rng.integers(0, 2, size=n)   # accion correcta en cada estado
    memoria = []
    while True:
        s = 0
        while True:
            a = int(rng.integers(0, 2))
            if a != buena[s]:                     # accion mala -> fin, r = 0
                memoria.append((s, a, 0.0, -1, True))
                break
            if s == n - 1:                        # ultima puerta buena -> fin, r = 1
                memoria.append((s, a, 1.0, -1, True))
                return buena, memoria
            memoria.append((s, a, 0.0, s + 1, False))
            s += 1


def q_optima(n, buena, gamma):
    """Q*(s, buena) = gamma^(pasos que faltan); Q*(s, mala) = 0."""
    Q = np.zeros((n, 2))
    for s in range(n):
        Q[s, buena[s]] = gamma ** (n - 1 - s)
    return Q


def objetivo(Q, transicion, gamma):
    s, a, r, s2, fin = transicion
    return r if fin else r + gamma * Q[s2].max()


def repetir(memoria, n, gamma, Qopt, modo, rng):
    """Devuelve cuantas actualizaciones hacen falta para llegar a Q*."""
    Q = np.zeros((n, 2))
    M = len(memoria)
    prioridad = np.ones(M)   # las transiciones nuevas entran con la maxima
    for t in range(1, MAX_ACTUALIZACIONES + 1):
        if modo == "uniforme":
            i = int(rng.integers(0, M))
        elif modo == "per":                    # sorteo proporcional a p^ALPHA
            p = prioridad ** ALPHA
            i = int(rng.choice(M, p=p / p.sum()))
        else:                                  # oraculo: el |delta| mayor de todos
            errores = [abs(objetivo(Q, tr, gamma) - Q[tr[0], tr[1]]) for tr in memoria]
            i = int(np.argmax(errores))
        s, a, *_ = memoria[i]
        delta = objetivo(Q, memoria[i], gamma) - Q[s, a]
        prioridad[i] = abs(delta) + EPS        # solo se refresca la que ha salido
        Q[s, a] += delta                       # alpha = 1: actualizacion exacta
        if np.abs(Q - Qopt).max() < TOL:
            return t
    return MAX_ACTUALIZACIONES


COLOR = {"uniforme": "#dc2626", "per": "#f59e0b", "oraculo": "#4f46e5"}
ETIQUETA = {"uniforme": "replay uniforme",
            "per": "PER realista (p^0,6, prioridades rancias)",
            "oraculo": "oraculo gloton (techo teorico)"}

ns = list(range(N_MIN, N_MAX + 1))
resultado = {m: [] for m in MODOS}
tam = []
cabecera = "  n   |memoria|" + "".join(f"{m:>12}" for m in MODOS)
print(cabecera)
print("  " + "-" * (len(cabecera) - 2))
for n in ns:
    gamma = 1.0 - 1.0 / n
    acum = {m: [] for m in MODOS}
    t_mem = []
    for rep in range(REPETICIONES):
        rng = np.random.default_rng(1000 * n + rep)
        buena, memoria = construir_memoria(n, rng)
        Qopt = q_optima(n, buena, gamma)
        t_mem.append(len(memoria))
        for m in MODOS:
            acum[m].append(repetir(memoria, n, gamma, Qopt, m, rng))
    tam.append(np.mean(t_mem))
    for m in MODOS:
        resultado[m].append(np.mean(acum[m]))
    fila = "".join(f"{resultado[m][-1]:12.1f}" for m in MODOS)
    print(f" {n:2d}   {tam[-1]:8.0f}{fila}")

print()
print(f"Con n = {ns[-1]}: " + ",  ".join(
    f"{ETIQUETA[m].split(' (')[0]} {resultado[m][-1]:.0f}" for m in MODOS) + " actualizaciones.")
print(f"Factor uniforme/oraculo con n = {ns[-1]}: "
      f"x{resultado['uniforme'][-1] / resultado['oraculo'][-1]:.0f}")

plt.figure(figsize=(7.5, 4.5))
for m in MODOS:
    plt.plot(ns, resultado[m], "o-", color=COLOR[m], lw=2, label=ETIQUETA[m])
plt.yscale("log")
plt.xlabel("n (longitud del pasillo)")
plt.ylabel("actualizaciones hasta alcanzar Q*  (escala log)")
plt.title("Blind Cliffwalk: coste de aprender segun como se muestrea")
plt.legend()
plt.grid(alpha=0.25, which="both")
plt.tight_layout()
plt.show()
