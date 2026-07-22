"""
Capítulo 1 — Tu primer agente en un pasillo.

Un robot vive en un pasillo de 5 casillas [0..4]. Empieza en la 0 y la meta
está en la 4. Cada paso cuesta -1; llegar a la meta da +10. Comparamos una
política aleatoria con una política sensata (ir siempre a la derecha).

No hay aprendizaje todavía: solo mostramos el bucle agente-entorno y la
recompensa. El resto del manual trata de cómo DESCUBRIR la buena política.

Ejecutar:
    python code/01-que-es-rl/pasillo.py
"""
import numpy as np

# --- El entorno: un pasillo de 5 casillas [0, 1, 2, 3, 4] ---
N = 5
META = N - 1

def paso(estado, accion):
    """Devuelve (nuevo_estado, recompensa, terminado)."""
    if accion == "derecha":
        estado = min(estado + 1, META)
    else:  # "izquierda"
        estado = max(estado - 1, 0)
    if estado == META:
        return estado, +10, True      # llegar a la meta
    return estado, -1, False          # cada paso cuesta

def ejecutar_episodio(politica, semilla):
    rng = np.random.default_rng(semilla)
    estado, total, terminado, pasos = 0, 0, False, 0
    while not terminado and pasos < 50:
        accion = politica(estado, rng)
        estado, r, terminado = paso(estado, accion)
        total += r
        pasos += 1
    return total, pasos

# --- Dos políticas (formas de decidir la acción) ---
def politica_azar(estado, rng):
    return rng.choice(["izquierda", "derecha"])

def politica_derecha(estado, rng):
    return "derecha"

# --- Comparamos ambas en 500 episodios ---
for nombre, pol in [("Al azar", politica_azar), ("Siempre derecha", politica_derecha)]:
    recompensas = [ejecutar_episodio(pol, semilla=i)[0] for i in range(500)]
    print(f"{nombre:16s} -> recompensa media = {np.mean(recompensas):6.2f}")
