"""
Capítulo 20 — RL de máxima entropía y empowerment.

Dos ideas sobre "explorar y mantener opciones abiertas", ilustradas en un
gridworld abierto donde hay MUCHOS caminos óptimos hasta la meta:

  1) RL de MÁXIMA ENTROPÍA (soft value iteration). Maximiza recompensa + entropía
     de la política. La temperatura alpha gradúa el compromiso:
       - alpha -> 0  : recuperamos el óptimo determinista de siempre.
       - alpha grande: la política se vuelve más aleatoria (explora/robusta)...
                       pero sacrifica recompensa. Es un COMPROMISO, no algo gratis.
     La política óptima es un softmax sobre los valores:  pi(a|s) ∝ exp(Q(s,a)/alpha).

  2) EMPOWERMENT: cuántos estados futuros distintos puede alcanzar el agente
     (aquí, un proxy: log2 del nº de estados alcanzables en k pasos). Formaliza
     "mantener opciones abiertas": el centro tiene más futuro posible que una esquina.

Ejecutar:
    python code/20-maxent-empowerment/maxent_empowerment.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- Gridworld abierto (muchos caminos óptimos a la meta) ----------
H, W = 7, 7
GOAL = (H - 1, W - 1)
GAMMA = 0.99
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # arriba, abajo, izquierda, derecha
A_NAMES = ["arriba", "abajo", "izquierda", "derecha"]
nS, nA = H * W, 4

def idx(r, c): return r * W + c
def rc(s): return divmod(s, W)
GOAL_S = idx(*GOAL)

def siguiente(s, a):
    """Estado resultante de aplicar la acción a en s (chocar con el muro = quedarse)."""
    r, c = rc(s)
    dr, dc = ACTIONS[a]
    nr = min(max(r + dr, 0), H - 1)
    nc = min(max(c + dc, 0), W - 1)
    return idx(nr, nc)

# Tabla de estado-siguiente (transiciones deterministas): NEXT[s, a] -> s'
NEXT = np.array([[siguiente(s, a) for a in range(nA)] for s in range(nS)])

# ---------- Utilidades numéricamente estables ----------
def logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)

def softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)

# ---------- Iteración de valor SUAVE (máxima entropía) ----------
def soft_value_iteration(alpha, iters=600):
    """Bellman suave: V(s) = alpha * logsumexp_a( Q(s,a)/alpha ).
    Devuelve V, Q y la política softmax pi(a|s) = softmax_a(Q(s,a)/alpha)."""
    V = np.zeros(nS)
    for _ in range(iters):
        Q = -1.0 + GAMMA * V[NEXT]                 # recompensa -1 por paso
        newV = alpha * logsumexp(Q / alpha, axis=1)  # "max suave"
        newV[GOAL_S] = 0.0                          # meta absorbente
        if np.max(np.abs(newV - V)) < 1e-10:
            V = newV
            break
        V = newV
    Q = -1.0 + GAMMA * V[NEXT]
    pol = softmax(Q / alpha, axis=1)
    return V, Q, pol

# ---------- Iteración de valor DURA (óptimo determinista, alpha -> 0) ----------
def hard_value_iteration(iters=600):
    V = np.zeros(nS)
    for _ in range(iters):
        Q = -1.0 + GAMMA * V[NEXT]
        newV = np.max(Q, axis=1)
        newV[GOAL_S] = 0.0
        if np.max(np.abs(newV - V)) < 1e-10:
            V = newV
            break
        V = newV
    return V

# ---------- Evaluar el RETORNO REAL de una política (sin el premio de entropía) ----------
def evaluar_politica(pol, iters=5000):
    V = np.zeros(nS)
    for _ in range(iters):
        newV = np.sum(pol * (-1.0 + GAMMA * V[NEXT]), axis=1)
        newV[GOAL_S] = 0.0
        if np.max(np.abs(newV - V)) < 1e-10:
            V = newV
            break
        V = newV
    return V

def entropia_por_estado(pol):
    return -np.sum(pol * np.log(pol + 1e-12), axis=1)

# ---------- 1) El compromiso recompensa <-> entropía ----------
V_opt = hard_value_iteration()
retorno_opt = V_opt[idx(0, 0)]
print(f"Retorno óptimo determinista (alpha -> 0): {retorno_opt:.2f}\n")

alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
retornos, entropias = [], []
print(f"{'alpha':>6}  {'retorno real':>12}  {'entropía media':>14}")
print("-" * 36)
for a in alphas:
    _, _, pol = soft_value_iteration(a)
    retorno = evaluar_politica(pol)[idx(0, 0)]
    H_media = entropia_por_estado(pol)[np.arange(nS) != GOAL_S].mean()
    retornos.append(retorno)
    entropias.append(H_media)
    print(f"{a:6.2f}  {retorno:12.2f}  {H_media:14.3f}")

# La política reparte probabilidad cuando hay varias acciones igual de buenas:
_, _, pol_demo = soft_value_iteration(0.1)
s0 = idx(0, 0)
print(f"\nEn el estado inicial (0,0) con alpha=0.1, la política soft reparte:")
for a in range(nA):
    print(f"  {A_NAMES[a]:>10}: {pol_demo[s0, a]:.2f}")
print("(dos caminos igual de cortos -> reparte entre 'abajo' y 'derecha')")

# ---------- 2) Empowerment: nº de futuros alcanzables ----------
def empowerment(k=4):
    emp = np.zeros(nS)
    for s in range(nS):
        alcanzables = {s}
        for _ in range(k):
            nuevos = set()
            for u in alcanzables:
                for a in range(nA):
                    nuevos.add(int(NEXT[u, a]))
            alcanzables |= nuevos
        emp[s] = np.log2(len(alcanzables))
    return emp.reshape(H, W)

EMP = empowerment(4)
print(f"\nEmpowerment (log2 de estados alcanzables en 4 pasos):")
print(f"  esquina (0,0): {EMP[0, 0]:.2f}   centro ({H // 2},{W // 2}): {EMP[H // 2, W // 2]:.2f}")
print("El centro 'abre' más posibilidades que una esquina, aunque la esquina no dé menos recompensa.")

# ---------- Gráficas ----------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.semilogx(alphas, retornos, "o-", color="#4f46e5")
ax1.axhline(retorno_opt, ls="--", color="#94a3b8", label="óptimo determinista")
ax1.set_xlabel("temperatura α (más entropía →)")
ax1.set_ylabel("retorno real")
ax1.set_title("Con más entropía se pierde recompensa")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2.semilogx(alphas, entropias, "o-", color="#059669")
ax2.axhline(np.log(nA), ls="--", color="#94a3b8", label="máxima (uniforme)")
ax2.set_xlabel("temperatura α")
ax2.set_ylabel("entropía media de la política")
ax2.set_title("...pero la política es más diversa")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
plt.tight_layout()

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 4.4))
H_estado = entropia_por_estado(pol_demo).reshape(H, W)
H_estado[GOAL] = np.nan
im3 = ax3.imshow(H_estado, cmap="viridis")
ax3.set_title("Entropía de la política por estado (α=0.1)")
ax3.text(GOAL[1], GOAL[0], "META", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
plt.colorbar(im3, ax=ax3, fraction=0.046)

im4 = ax4.imshow(EMP, cmap="magma")
ax4.set_title("Empowerment (opciones futuras)")
for (r, c), v in np.ndenumerate(EMP):
    ax4.text(c, r, f"{v:.1f}", ha="center", va="center", color="white", fontsize=7)
plt.colorbar(im4, ax=ax4, fraction=0.046)
plt.tight_layout()
plt.show()
