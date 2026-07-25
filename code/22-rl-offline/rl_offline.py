"""
RL offline: aprender de datos ya registrados, sin entorno con el que probar nada.

Nos entregan un conjunto de datos FIJO D de transiciones (s, a, r, s') que
registró en su día una política de comportamiento pi_beta, y no tenemos entorno.
Hay que sacar de ahí la mejor política posible. Este programa compara sobre EL
MISMO D, y en dos escenarios, los tres candidatos del capítulo:

  1. Clonación de conducta (BC): aprendizaje supervisado puro. Copia a pi_beta.
  2. Q-learning offline "a secas": la actualización de siempre, pero repasando
     una y otra vez el mismo búfer, que ya nunca se rellena. SE ROMPE.
  3. Las dos curas: restringir el max al soporte de los datos (BCQ tabular,
     "quédate cerca de D") y CQL tabular ("sé pesimista con lo desconocido").

POR QUÉ SE ROMPE EL Q-LEARNING OFFLINE. El objetivo de Bellman contiene un
max_{a'} Q(s', a') que recorre TODAS las acciones, también las que nadie ejecutó
jamás. El valor de esas acciones no es un dato: es lo que la tabla (o la red) se
inventa. Con entorno, la realidad te desmiente en cuanto pruebas la acción; sin
entorno, no te desmiente nadie NUNCA, y el bootstrapping del capítulo 7 recicla
ese número inventado hacia atrás hasta envenenar la función de valor entera. Es
el sesgo de maximización de los capítulos 9 y 13 sin nadie que lo corrija.

PARTE 1 · Una cadena de cuatro casillas, para seguirla con el dedo. Todos los
valores verdaderos son enteros y la tabla Q se imprime entrada por entrada: se
ve el número inventado, se ve el max agarrándose a él y se ve el veneno subiendo
hacia atrás por la cadena.

PARTE 2 · Un gridworld 6x6 con trampas y suelo resbaladizo (de la misma familia
que las rejillas de los capítulos 5 y 6, pero mayor y estocástico), donde
además se ve lo que la clonación NO puede hacer: el método conservador
cose trozos de trayectorias distintas y sale por un camino que ningún episodio
del conjunto de datos recorrió entero.

El entorno se toca exactamente dos veces: para GENERAR D una vez y para EVALUAR
de verdad las políticas al final (con programación dinámica del capítulo 5, o
sea sin ruido de muestreo). El aprendizaje solo recibe D: ni siquiera se le pasa
el objeto del entorno. Ese es todo el juego.

Cómo ejecutarlo:
    pip install -r requirements.txt
    python code/22-rl-offline/rl_offline.py

Solo necesita numpy y matplotlib, así que también corre en el navegador (Pyodide).
"""
import numpy as np
import matplotlib.pyplot as plt

IND, CIAN, VERDE, AMBAR, ROJO, GRIS = (
    "#4f46e5", "#0ea5e9", "#059669", "#d97706", "#dc2626", "#94a3b8")


# =============================================================================
# 1. Un MDP tabular cualquiera, descrito por su modelo exacto
# =============================================================================
class MDP:
    """Modelo tabular: P[s,a,s'] y la recompensa Rsas[s,a,s'] de esa transición.

    Este modelo es del autor del experimento, NO del agente. Solo se usa para
    (a) generar el conjunto de datos una vez y (b) evaluar de verdad las
    políticas al final. Ningún algoritmo de aprendizaje lo recibe.
    """

    def __init__(self, P, Rsas, terminal, s0):
        self.P, self.Rsas, self.terminal, self.s0 = P, Rsas, terminal, s0
        self.nS, self.nA = P.shape[0], P.shape[1]
        self.R = np.einsum("sax,sax->sa", P, Rsas)   # recompensa esperada R(s,a)
        self.acum = np.cumsum(P, axis=2)             # para muestrear s' de un tirón


def registra(mdp, pi_beta, n_episodios, gamma, semilla, max_pasos=60):
    """Registra el histórico D siguiendo la política de comportamiento pi_beta.

    Devuelve D = (S, A, R, S2, FIN) —cinco vectores—, el retorno descontado que
    se observó en cada episodio y de qué episodio salió cada transición (esto
    último NO se le pasa a ningún algoritmo: solo lo usamos para comprobar al
    final si alguien recorrió de verdad el camino que sale del aprendizaje).
    D es TODO lo que verán los algoritmos: ni una llamada más al entorno.

    Los episodios que llegan a max_pasos se truncan con FIN=False: el episodio
    se corta, pero el futuro de ese estado sigue existiendo y sería mentira
    decirle a Bellman que a partir de ahí vale cero.
    """
    rng = np.random.default_rng(semilla)
    s = np.full(n_episodios, mdp.s0)
    vivo = np.ones(n_episodios, dtype=bool)
    ret, desc = np.zeros(n_episodios), np.ones(n_episodios)
    Sl, Al, Rl, S2l, Fl, El = [], [], [], [], [], []
    acum_pi = np.cumsum(pi_beta, axis=1)
    for _ in range(max_pasos):
        v = np.flatnonzero(vivo)
        if v.size == 0:
            break
        ss = s[v]
        u = rng.random(v.size)[:, None]
        a = np.minimum((u > acum_pi[ss]).sum(axis=1), mdp.nA - 1)
        u2 = rng.random(v.size)[:, None]
        s2 = np.minimum((u2 > mdp.acum[ss, a]).sum(axis=1), mdp.nS - 1)
        r = mdp.Rsas[ss, a, s2]
        fin = mdp.terminal[s2]
        Sl.append(ss); Al.append(a); Rl.append(r); S2l.append(s2); Fl.append(fin)
        El.append(v)
        ret[v] += desc[v] * r
        desc[v] *= gamma
        s[v] = s2
        vivo[v[fin]] = False
    D = (np.concatenate(Sl), np.concatenate(Al), np.concatenate(Rl),
         np.concatenate(S2l), np.concatenate(Fl))
    return D, ret, np.concatenate(El)


def evalua_exacto(mdp, pi, gamma, Rsas=None, tol=1e-12, max_barridos=400):
    """V^pi exacta por evaluación iterativa de política (capítulo 5).

    Aquí SÍ se usa el modelo verdadero: es el juez, no el alumno. Al ser exacta,
    el "retorno real" que sale no tiene ni un gramo de ruido de muestreo.
    """
    R = mdp.R if Rsas is None else np.einsum("sax,sax->sa", mdp.P, Rsas)
    V = np.zeros(mdp.nS)
    for _ in range(max_barridos):
        Vn = np.einsum("sa,sa->s", pi, R + gamma * (mdp.P @ V))
        if np.max(np.abs(Vn - V)) < tol:
            return Vn
        V = Vn
    return V


def valor_optimo(mdp, gamma, tol=1e-12, max_barridos=400):
    """V* exacta por iteración de valor: la vara de medir de arriba del todo."""
    V = np.zeros(mdp.nS)
    for _ in range(max_barridos):
        Vn = np.max(mdp.R + gamma * (mdp.P @ V), axis=1)
        if np.max(np.abs(Vn - V)) < tol:
            return Vn
        V = Vn
    return V


def politica_optima(mdp, gamma):
    """La política greedy respecto a V*, para saber cuánto se deja cada método."""
    V = valor_optimo(mdp, gamma)
    return matriz_politica(np.argmax(mdp.R + gamma * (mdp.P @ V), axis=1), mdp.nA)


def matriz_politica(acciones, nA):
    """Convierte un vector de acciones (una por estado) en una matriz pi(a|s)."""
    pi = np.zeros((len(acciones), nA))
    pi[np.arange(len(acciones)), acciones] = 1.0
    return pi


# =============================================================================
# 2. Los métodos offline. Todos reciben SOLO (D, nS, nA): jamás el entorno
# =============================================================================
def cuenta_pares(D, nS, nA):
    """N[s,a] = cuántas veces aparece el par (s,a) en el conjunto de datos."""
    plano = np.bincount(D[0] * nA + D[1], minlength=nS * nA)
    return plano.reshape(nS, nA).astype(float)


def clonacion_de_conducta(N, moda=False):
    """Aprendizaje supervisado puro: ajusta pi(a|s) a las frecuencias de D.

    No hay recompensas, ni gamma, ni ecuación de Bellman: es un clasificador.
    Por eso su techo es quien generó los datos. Con `moda=True` desplegamos la
    acción más frecuente en vez de muestrear, que es la versión más fuerte de la
    clonación (limpia el ruido de pi_beta, pero no puede cambiar su ruta).
    """
    total = N.sum(axis=1, keepdims=True)
    if moda:
        return matriz_politica(np.argmax(N, axis=1), N.shape[1])
    return np.where(total > 0, N / np.maximum(total, 1.0), 1.0 / N.shape[1])


def aprende_offline(D, N, gamma, soporte=False, c=0.0, lr=1.0, n_iter=300, Q0=None):
    """Q-learning offline: barridos completos sobre el conjunto FIJO D.

    Es la actualización de siempre —objetivo r + gamma·max_{a'} Q(s',a')— pero el
    búfer del capítulo 12 nunca se rellena: repasamos una y otra vez LAS MISMAS
    transiciones. Cada barrido recalcula Q(s,a) como la media de los objetivos de
    ese par (mismo punto fijo que dar pasitos de tamaño alpha, pero sin el ruido
    del muestreo, así que el resultado es exactamente reproducible).

    Los pares (s,a) que NO están en D no se actualizan jamás: se quedan con su
    valor inicial. Ese valor inventado es la extrapolación del caso tabular.

    Dos mandos, uno por cada familia de soluciones del capítulo:

      soporte=True -> el max solo recorre las acciones con N(s',a') > 0 (BCQ
          tabular: "quédate cerca de los datos", el primo tabular de la
          penalización KL contra pi_ref del capítulo 21).
      c > 0 -> término conservador de CQL: se resta c·softmax(Q(s,·)), es decir
          se empuja hacia abajo el valor de lo que la tabla cree bueno, y con más
          fuerza cuanto más alto lo cree. Las acciones respaldadas por datos
          tienen quien las defienda (el término de Bellman las devuelve a su
          sitio); las que nadie ejecutó nunca, no, así que se hunden hasta dejar
          de ser el máximo. No hay ninguna restricción explícita: la política
          final sigue siendo el argmax sobre TODAS las acciones.

    Devuelve (Q, mascara_de_soporte).
    """
    S, A, R, S2, FIN = D
    nS, nA = N.shape
    en_datos = N > 0
    mascara = en_datos.copy()
    huerfanos = ~en_datos.any(axis=1)        # estados que no salen en D
    mascara[huerfanos] = True                # ahí no hay nada que restringir
    con_datos = ~huerfanos
    par = S * nA + A
    Q = np.zeros((nS, nA)) if Q0 is None else Q0.copy()

    for _ in range(n_iter):
        # (1) objetivo TD. El (1 - FIN) es V(terminal) = 0, como siempre.
        V2 = np.where(mascara, Q, -np.inf).max(axis=1) if soporte else Q.max(axis=1)
        objetivo = R + gamma * np.where(FIN, 0.0, V2[S2])
        suma = np.bincount(par, weights=objetivo, minlength=nS * nA).reshape(nS, nA)
        media = suma / np.maximum(N, 1.0)
        Q = np.where(en_datos, Q + lr * (media - Q), Q)
        # (2) empujón conservador de CQL, solo en los estados que aparecen en D
        if c > 0.0:
            z = Q - Q.max(axis=1, keepdims=True)          # softmax estable
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            Q -= lr * c * np.where(con_datos[:, None], p, 0.0)
    return Q, mascara


def politica_greedy(Q, mascara=None):
    """Política determinista greedy respecto a Q (opcionalmente restringida)."""
    Qef = Q if mascara is None else np.where(mascara, Q, -np.inf)
    return matriz_politica(np.argmax(Qef, axis=1), Q.shape[1])


# =============================================================================
# 3. PARTE 1 · La cadena de cuatro casillas, para seguirla con el dedo
# =============================================================================
SEGUIR, ATAJAR = 0, 1


def construye_cadena(n=4, r_paso=-1.0, r_pozo=-20.0):
    """Un senderista va de la casilla 0 a la META. En cada casilla puede:

        SEGUIR -> avanza una casilla, recompensa -1 (cuesta caminar)
        ATAJAR -> se sale del sendero, cae al POZO, recompensa -20

    Con gamma = 1 la política óptima es obvia (seguir siempre) y V*(s0) = -4:
    todos los valores verdaderos son enteros y se comprueban de cabeza.
    """
    META, POZO, nS = n, n + 1, n + 2
    P = np.zeros((nS, 2, nS))
    Rsas = np.zeros((nS, 2, nS))
    for s in range(n):
        P[s, SEGUIR, s + 1] = 1.0
        Rsas[s, SEGUIR, s + 1] = r_paso
        P[s, ATAJAR, POZO] = 1.0
        Rsas[s, ATAJAR, POZO] = r_pozo
    terminal = np.zeros(nS, dtype=bool)
    for t in (META, POZO):
        terminal[t] = True
        P[t, :, t] = 1.0                  # absorbente y con recompensa 0
    return MDP(P, Rsas, terminal, s0=0)


def comportamiento_cadena(mdp, n, p_atajar, valladas):
    """pi_beta: ataja con probabilidad p_atajar... salvo en las casillas valladas.

    Ahí la probabilidad es CERO ESTRUCTURAL: no es que haya salido poco, es que
    no puede salir nunca. Ese es el agujero de cobertura que ningún volumen de
    datos podrá tapar. Y es lo normal en los datos reales: un médico jamás receta
    el fármaco X al paciente de tipo Y, un conductor jamás gira contra el tráfico.
    """
    pi = np.zeros((mdp.nS, 2))
    for s in range(n):
        p = 0.0 if s in valladas else p_atajar
        pi[s, ATAJAR], pi[s, SEGUIR] = p, 1.0 - p
    pi[n:, SEGUIR] = 1.0                  # irrelevante: son absorbentes
    return pi


def tabla_q(titulo, Q, N, V_opt, n):
    """Imprime la tabla Q casilla por casilla, marcando los huecos de cobertura."""
    print(f"\n{titulo}")
    print(f"{'casilla':>8} | {'Q(s,SEGUIR)':>12} | {'Q(s,ATAJAR)':>12} | "
          f"{'max_a Q(s,a)':>13} | {'V*(s) real':>11}")
    print("-" * 74)
    for s in range(n):
        marca = "*" if N[s, ATAJAR] == 0 else " "
        print(f"{s:>8} | {Q[s, SEGUIR]:>12.2f} | {Q[s, ATAJAR]:>11.2f}{marca} | "
              f"{Q[s].max():>13.2f} | {V_opt[s]:>11.2f}")


def parte1():
    """La cadena: el mecanismo del envenenamiento, número a número."""
    n, gamma = 4, 1.0
    mdp = construye_cadena(n)
    pi_beta = comportamiento_cadena(mdp, n, p_atajar=0.15, valladas={2})
    D, retornos, _ = registra(mdp, pi_beta, 400, gamma, semilla=22, max_pasos=n + 1)
    N = cuenta_pares(D, mdp.nS, mdp.nA)

    # --- Los cuatro métodos, todos sobre EXACTAMENTE el mismo D --------------
    Q_ing, _ = aprende_offline(D, N, gamma)
    Q_sop, masc = aprende_offline(D, N, gamma, soporte=True)
    Q_cql, _ = aprende_offline(D, N, gamma, c=0.5, lr=0.5, n_iter=600)

    V_opt = valor_optimo(mdp, gamma)
    metodos = {
        "Clonación de conducta": (clonacion_de_conducta(N), retornos.mean()),
        "Q-learning offline": (politica_greedy(Q_ing), Q_ing[0].max()),
        "Q-learning + soporte": (politica_greedy(Q_sop, masc),
                                 np.where(masc, Q_sop, -np.inf)[0].max()),
        "CQL tabular": (politica_greedy(Q_cql), Q_cql[0].max()),
    }

    print("=" * 78)
    print("PARTE 1 · Cadena de 4 casillas.  SEGUIR = -1 por paso,  ATAJAR = -20 (pozo)")
    print("=" * 78)
    print(f"Conjunto de datos D: {len(D[0])} transiciones de 400 episodios de pi_beta.")
    print("pi_beta ataja con probabilidad 0,15 ... salvo en la casilla 2, donde NUNCA.")
    print(f"Óptimo real V*(s0) = {V_opt[0]:.2f}   |   valor real de pi_beta = "
          f"{evalua_exacto(mdp, pi_beta, gamma)[0]:.2f}")

    print("\nCuántas veces aparece cada par (s,a) en D  [ * = hueco de cobertura ]")
    print(f"{'casilla':>8} | {'n(s, SEGUIR)':>13} | {'n(s, ATAJAR)':>13}")
    print("-" * 42)
    for s in range(n):
        marca = " *" if N[s, ATAJAR] == 0 else ""
        print(f"{s:>8} | {int(N[s, SEGUIR]):>13d} | {int(N[s, ATAJAR]):>13d}{marca}")

    tabla_q("Tabla Q del Q-LEARNING OFFLINE 'a secas'  [ * = par que NO está en D ]",
            Q_ing, N, V_opt, n)
    print("El par (2, ATAJAR) no se ejecutó jamás, así que nadie lo actualizó nunca y")
    print("sigue valiendo lo que valía al empezar: 0,00. Y 0,00 > -2,00, de modo que el")
    print("max de la casilla 2 se agarra a un número inventado... y el bootstrapping lo")
    print("arrastra hacia atrás: Q(1,SEGUIR) = -1 + 0 = -1 en vez de -3, y de ahí")
    print("Q(0,SEGUIR) = -1 + (-1) = -2 en vez de -4. Una sola celda envenena la cadena.")

    tabla_q("Tabla Q de CQL TABULAR (mismo D, mismo bucle, con el empujón hacia abajo)",
            Q_cql, N, V_opt, n)
    print(f"CQL hunde Q(2,ATAJAR) de 0,00 a {Q_cql[2, ATAJAR]:.2f} sin haber visto esa acción")
    print("jamás, el max vuelve a caer sobre SEGUIR y el veneno no llega a entrar.")

    print("\nLo que cree cada método frente a lo que de verdad consigue al desplegarlo")
    print(f"{'método':>23} | {'se cree':>9} | {'retorno real':>13} | {'brecha':>8} | "
          f"P(ATAJAR) en s0 s1 s2 s3")
    print("-" * 96)
    reales, creidos = {}, {}
    for k, (pi, cree) in metodos.items():
        real = evalua_exacto(mdp, pi, gamma)[0]
        reales[k], creidos[k] = real, cree
        probs = " ".join(f"{pi[s, ATAJAR]:>4.2f}" for s in range(n))
        print(f"{k:>23} | {cree:>9.2f} | {real:>13.2f} | {cree - real:>8.2f} |      {probs}")
    print("-" * 96)
    print("El Q-learning offline se cree MEJOR que el óptimo real de la tarea (-4,00),")
    print("cosa imposible por construcción, y al desplegarlo se tira al pozo desde la")
    print("casilla 2 y cosecha -22,00. Factor 11 entre lo que cree y lo que consigue.")
    print("La clonación no se engaña (no estima nada: le ponemos como 'se cree' el")
    print("retorno medio observado en D), pero tampoco mejora: reproduce a pi_beta.")

    # --- Dos controles honestos ---------------------------------------------
    Q_pes, _ = aprende_offline(D, N, gamma, Q0=np.full((mdp.nS, mdp.nA), -20.0))
    pi_pes = politica_greedy(Q_pes)
    print("\nControl 1: el mismo Q-learning offline, pero con Q inicializada a -20")
    print(f"  se cree {Q_pes[0].max():>6.2f} y obtiene "
          f"{evalua_exacto(mdp, pi_pes, gamma)[0]:>6.2f}. Con una inicialización PESIMISTA")
    print("  no se rompe. Eso no invalida el ejemplo: ES la moraleja. En una tabla, el")
    print("  valor de lo que nunca viste es la inicialización; con una red neuronal es")
    print("  lo que la red extrapola, y ahí no eliges tú. Lo único sistemático es que el")
    print("  max se queda con los valores altos. Ser pesimista con lo desconocido no es")
    print("  un truco de implementación: es el algoritmo conservador.")

    D2, _, _ = registra(mdp, pi_beta, 40000, gamma, semilla=23, max_pasos=n + 1)
    N2 = cuenta_pares(D2, mdp.nS, mdp.nA)
    Q2, _ = aprende_offline(D2, N2, gamma)
    print("\nControl 2: el mismo Q-learning offline con CIEN VECES más datos")
    print(f"  {len(D2[0])} transiciones -> se cree {Q2[0].max():>6.2f} y obtiene "
          f"{evalua_exacto(mdp, politica_greedy(Q2), gamma)[0]:>6.2f}. Idéntico.")
    print("  El agujero de cobertura es estructural, no estadístico: lo que falta no")
    print("  son más datos, es OTRO tipo de datos, y el conjunto ya está cerrado.")

    print("\nControl 3: si la clonación desplegara la MODA en vez de muestrear, aquí")
    pi_moda = clonacion_de_conducta(N, moda=True)
    print(f"  sacaría {evalua_exacto(mdp, pi_moda, gamma)[0]:.2f}, o sea el óptimo: en esta cadena el defecto de")
    print("  pi_beta es puro ruido y la moda lo limpia. En el gridworld de la Parte 2")
    print("  no habrá tanta suerte, porque allí el defecto de pi_beta es sistemático.")
    return {"metodos": list(metodos), "cree": creidos, "real": reales,
            "V_opt": V_opt[:n], "Q_ing": Q_ing[:n], "Q_cql": Q_cql[:n], "n": n}


# =============================================================================
# 4. PARTE 2 · Un gridworld 6x6 con trampas y suelo resbaladizo
# =============================================================================
FILAS, COLS, NA = 6, 6, 4
DR = np.array([-1, 0, 1, 0])              # 0=arriba, 1=derecha, 2=abajo, 3=izquierda
DC = np.array([0, 1, 0, -1])
SIMBOLOS = ["^", ">", "v", "<"]
INICIO, META = (5, 0), (0, 5)
TRAMPAS = {(1, 1), (1, 4), (3, 1), (3, 3)}
R_PASO, R_TRAMPA, R_META = -1.0, -20.0, 0.0
GAMMA_G, DESLIZ = 0.95, 0.10


def idx(f, c):
    return f * COLS + c


def construye_gridworld():
    """Rejilla 6x6 con meta, cuatro trampas y suelo resbaladizo.

    Cada paso cuesta -1, caer en una trampa -20 y llegar a la meta 0 (llegar no
    premia: simplemente deja de costar). Con probabilidad 0,1 te desvías 90
    grados. Ojo a este detalle, que es la clave del experimento: TODAS las
    recompensas son <= 0, así que ningún retorno puede llegar a cero. El cero con
    el que se inicializa la tabla es, por tanto, el valor más optimista
    concebible: exactamente lo que hace visible el error de extrapolación.
    """
    nS = FILAS * COLS
    P = np.zeros((nS, NA, nS))
    Rsas = np.zeros((nS, NA, nS))
    terminal = np.zeros(nS, dtype=bool)
    terminal[idx(*META)] = True
    for t in TRAMPAS:
        terminal[idx(*t)] = True
    for f in range(FILAS):
        for c in range(COLS):
            s = idx(f, c)
            if terminal[s]:
                P[s, :, s] = 1.0
                continue
            for a in range(NA):
                for a_ef, prob in ((a, 1 - DESLIZ), ((a + 1) % 4, DESLIZ / 2),
                                   ((a + 3) % 4, DESLIZ / 2)):
                    nf = min(max(f + DR[a_ef], 0), FILAS - 1)   # el borde te frena
                    nc = min(max(c + DC[a_ef], 0), COLS - 1)
                    s2 = idx(nf, nc)
                    P[s, a, s2] += prob
                    Rsas[s, a, s2] = (R_TRAMPA if (nf, nc) in TRAMPAS
                                      else R_META if (nf, nc) == META else R_PASO)
    return MDP(P, Rsas, terminal, s0=idx(*INICIO))


def controlador_heuristico(mdp):
    """El sistema que ya estaba en producción y que generó el histórico.

    "Sube hasta la fila de la meta y luego ve hacia ella; si la casilla de
    destino es una trampa, muévete en el otro eje". Es sensato (llega en 10
    pasos, el mínimo) pero NO es óptimo: sube pegado al borde izquierdo rozando
    dos trampas, y con el suelo resbaladizo eso se paga.
    """
    pol = np.zeros(mdp.nS, dtype=int)
    for f in range(FILAS):
        for c in range(COLS):
            if mdp.terminal[idx(f, c)]:
                continue
            candidatas = ([0] if f > META[0] else [2] if f < META[0] else [])
            candidatas += ([1] if c < META[1] else [3] if c > META[1] else [])
            elegida = candidatas[0] if candidatas else 0
            for a in candidatas:
                nf = min(max(f + DR[a], 0), FILAS - 1)
                nc = min(max(c + DC[a], 0), COLS - 1)
                if (nf, nc) not in TRAMPAS:
                    elegida = a
                    break
            pol[idx(f, c)] = elegida
    return pol


def epsilon_greedy(pol, epsilon, nS):
    """pi_beta = epsilon-greedy sobre el controlador: así es un histórico real,
    con su ruido de exploración y sus rutinas."""
    pi = np.full((nS, NA), epsilon / NA)
    pi[np.arange(nS), pol] += 1.0 - epsilon
    return pi


def prob_meta(mdp, pi):
    """Probabilidad de acabar en la meta: la misma evaluación exacta, pero con
    recompensa 1 al entrar en la meta, sin descuento y sin nada más."""
    Rmeta = np.zeros_like(mdp.Rsas)
    Rmeta[:, :, idx(*META)] = 1.0
    Rmeta[mdp.terminal] = 0.0            # los terminales son absorbentes: no cuentan
    return 100.0 * evalua_exacto(mdp, pi, 1.0, Rsas=Rmeta)[mdp.s0]


def camino(mdp, pi, max_pasos=20):
    """Recorrido de la política SIN deslizamiento, para leer qué hace de verdad.

    Devuelve la lista de estados que pisa y su versión en texto.
    """
    ruta, vistos, s, atasco = [mdp.s0], {mdp.s0}, mdp.s0, False
    while len(ruta) <= max_pasos and not mdp.terminal[s]:
        f, c = divmod(s, COLS)
        a = int(np.argmax(pi[s]))
        s = idx(min(max(f + DR[a], 0), FILAS - 1), min(max(c + DC[a], 0), COLS - 1))
        if s in vistos:
            atasco = True
            break
        ruta.append(s)
        vistos.add(s)
    txt = " ".join(f"({p // COLS},{p % COLS})" for p in ruta)
    if atasco:
        txt += f" [se atasca en ({ruta[-1] // COLS},{ruta[-1] % COLS})]"
    elif mdp.terminal[s]:
        txt += "=META" if s == idx(*META) else "=TRAMPA"
    return ruta, txt


def secuencias(D, ep):
    """Reconstruye la lista de estados que visitó cada episodio del histórico."""
    orden = np.argsort(ep, kind="stable")
    S, S2, e = D[0][orden], D[3][orden], ep[orden]
    cortes = np.flatnonzero(np.diff(e)) + 1
    return [list(S[t]) + [S2[t[-1]]] for t in np.split(np.arange(len(e)), cortes)]


def trozo_mas_largo(secs, ruta):
    """Longitud del trozo más largo de `ruta` que alguien recorrió DE SEGUIDO.

    Si sale igual que la ruta entera, esa política se limita a repetir algo que
    ya estaba en los datos. Si sale mucho más corto, la política ha COSIDO
    fragmentos de trayectorias distintas: eso es lo que la clonación no sabe
    hacer, y la razón de que el RL offline no sea solo imitación.
    """
    def esta(sub):
        return any(any(s[i:i + len(sub)] == sub for i in range(len(s) - len(sub) + 1))
                   for s in secs)
    mejor = 0
    for i in range(len(ruta)):
        j = i + mejor + 1
        while j <= len(ruta) and esta(ruta[i:j]):
            mejor, j = j - i, j + 1
    return mejor


def entrena_todo(D, N):
    """Los cuatro métodos sobre el mismo D. Devuelve {nombre: (pi, valor que cree)}."""
    Q_ing, _ = aprende_offline(D, N, GAMMA_G)
    Q_sop, masc = aprende_offline(D, N, GAMMA_G, soporte=True)
    Q_cql, _ = aprende_offline(D, N, GAMMA_G, c=0.5, lr=0.5, n_iter=1500)
    s0 = idx(*INICIO)
    return {
        "Clonación de conducta": (clonacion_de_conducta(N, moda=True), np.nan),
        "Q-learning offline": (politica_greedy(Q_ing), Q_ing[s0].max()),
        "Q-learning + soporte": (politica_greedy(Q_sop, masc),
                                 np.where(masc, Q_sop, -np.inf)[s0].max()),
        "CQL tabular": (politica_greedy(Q_cql), Q_cql[s0].max()),
    }


def parte2():
    """El gridworld: el mismo fenómeno y, además, lo que la clonación no puede."""
    mdp = construye_gridworld()
    pol_h = controlador_heuristico(mdp)
    pi_beta = epsilon_greedy(pol_h, 0.40, mdp.nS)
    pares = int((~mdp.terminal).sum()) * NA
    V_opt = valor_optimo(mdp, GAMMA_G)
    pi_opt = politica_optima(mdp, GAMMA_G)

    print("\n" + "=" * 78)
    print("PARTE 2 · Gridworld 6x6 resbaladizo (meta, 4 trampas, cada paso -1)")
    print("=" * 78)
    print(f"gamma = {GAMMA_G} | desliz = {DESLIZ} | todas las recompensas <= 0, así que")
    print("NINGÚN retorno puede llegar a cero: el cero de la tabla es inalcanzable.")

    # Cinco conjuntos de datos con la misma pi_beta: nada depende de una semilla.
    datos, episodios = [], []
    for sem in range(5):
        D, _, ep = registra(mdp, pi_beta, 300, GAMMA_G, semilla=sem)
        datos.append((D, cuenta_pares(D, mdp.nS, mdp.nA)))
        episodios.append(ep)
    cob = np.mean([100.0 * (N[~mdp.terminal] > 0).sum() / pares for _, N in datos])
    print(f"\nD: 300 episodios de pi_beta = epsilon-greedy(0,40) sobre el controlador")
    print(f"heurístico -> {np.mean([len(D[0]) for D, _ in datos]):.0f} transiciones. Cubren el {cob:.0f} % de los "
          f"{pares} pares (s,a).")
    print(f"El {100 - cob:.0f} % restante nunca se ejecutó: su valor es una invención de la")
    print("tabla que nadie puede desmentir.")

    acum = {}
    politicas0 = None
    for D, N in datos:
        modelos = entrena_todo(D, N)      # <- hasta aquí NADIE ha tocado el entorno
        if politicas0 is None:
            politicas0 = {k: v[0] for k, v in modelos.items()}
        for k, (pi, cree) in modelos.items():
            real = evalua_exacto(mdp, pi, GAMMA_G)[mdp.s0]   # <- y solo aquí se despliega
            acum.setdefault(k, []).append((cree, real, prob_meta(mdp, pi)))

    print("\nLo que cada método CREE que vale su política frente a lo que de verdad")
    print("obtiene al desplegarla (media de 5 conjuntos de datos):")
    print("-" * 74)
    print(f"{'método':<24}{'V(inicio) que cree':>19}{'retorno real':>14}{'llega':>9}")
    print("-" * 74)
    resumen = {}
    for k, filas in acum.items():
        cree, real, meta = np.mean(np.array(filas), axis=0)
        resumen[k] = (cree, real, meta)
        print(f"{k:<24}{'—' if np.isnan(cree) else f'{cree:.2f}':>19}{real:>14.2f}{meta:>8.0f} %")
    print("-" * 74)
    print(f"{'pi_beta (generó D)':<24}{'—':>19}"
          f"{evalua_exacto(mdp, pi_beta, GAMMA_G)[mdp.s0]:>14.2f}"
          f"{prob_meta(mdp, pi_beta):>8.0f} %")
    pi_h = matriz_politica(pol_h, NA)
    print(f"{'el controlador sin ruido':<24}{'—':>19}"
          f"{evalua_exacto(mdp, pi_h, GAMMA_G)[mdp.s0]:>14.2f}"
          f"{prob_meta(mdp, pi_h):>8.0f} %")
    print(f"{'óptimo (con el modelo)':<24}{V_opt[mdp.s0]:>19.2f}"
          f"{evalua_exacto(mdp, pi_opt, GAMMA_G)[mdp.s0]:>14.2f}{prob_meta(mdp, pi_opt):>8.0f} %")
    print("-" * 74)
    ql = resumen["Q-learning offline"]
    print(f"Otra vez: el Q-learning offline se cree {ql[0]:.2f} en un problema donde lo mejor")
    print(f"posible es {V_opt[mdp.s0]:.2f}, se equivoca en {ql[0] - ql[1]:+.1f} puntos sobre su propia política")
    print("y casi nunca llega a la meta. Los conservadores se creen poco y lo cumplen.")
    print("La clonación, por su parte, recupera el controlador limpio de ruido: ese es")
    print("su techo, y no lo pasa. Los conservadores sí lo pasan. Veamos por qué.")

    print("\nEl camino de cada política (sin deslizamiento, para verlo limpio):")
    print("-" * 74)
    rutas = {}
    for k in ("Clonación de conducta", "Q-learning offline", "CQL tabular"):
        rutas[k], txt = camino(mdp, politicas0[k])
        print(f"  {k:<22} {txt}")
    print(f"  {'óptimo (referencia)':<22} {camino(mdp, pi_opt)[1]}")
    print("-" * 74)
    secs = secuencias(datos[0][0], episodios[0])
    print("¿Y cuánto de cada camino recorrió alguien DE VERDAD? El trozo más largo que")
    print("aparece seguido en un solo episodio del conjunto de datos:")
    for k in ("Clonación de conducta", "CQL tabular"):
        r = rutas[k]
        print(f"  {k:<22} {trozo_mas_largo(secs, r):>2d} casillas de {len(r)}")
    print("-" * 74)
    print("Ahí está lo que la clonación NO puede hacer. La clonación repite entero el")
    print("camino de pi_beta por el borde izquierdo, que pasa rozando dos trampas: ese")
    print("es su techo, y por eso su camino aparece tal cual en los datos. CQL se saca")
    print("un camino por el centro que NADIE recorrió entero; cose trozos sueltos de la")
    print("exploración de episodios distintos. Eso, y no la imitación, es lo que hace")
    print("que valga la pena hacer RL offline. Y el Q-learning offline, en cambio, se")
    print("queda empujando la pared de abajo, convencido de que esa acción, que nunca")
    print("vio ejecutar, vale cero.")

    # --- Control: ¿es culpa del algoritmo o del agujero en los datos? --------
    print("\nControl: mismo código, distinta COBERTURA (subimos el epsilon de pi_beta)")
    print("-" * 74)
    print(f"{'epsilon':>8}{'cobertura':>12}{'BC real':>10}{'QL cree':>10}{'QL real':>10}"
          f"{'CQL cree':>11}{'CQL real':>11}")
    print("-" * 74)
    barrido = []
    for eps in (0.05, 0.20, 0.40, 0.60, 0.80, 1.00):
        pi_b = epsilon_greedy(pol_h, eps, mdp.nS)
        fila = np.zeros(6)
        for sem in range(3):
            D, _, _ = registra(mdp, pi_b, 300, GAMMA_G, semilla=100 + sem)
            N = cuenta_pares(D, mdp.nS, mdp.nA)
            m = entrena_todo(D, N)
            fila += np.array([
                100.0 * (N[~mdp.terminal] > 0).sum() / pares,
                evalua_exacto(mdp, m["Clonación de conducta"][0], GAMMA_G)[mdp.s0],
                m["Q-learning offline"][1],
                evalua_exacto(mdp, m["Q-learning offline"][0], GAMMA_G)[mdp.s0],
                m["CQL tabular"][1],
                evalua_exacto(mdp, m["CQL tabular"][0], GAMMA_G)[mdp.s0]])
        fila /= 3
        barrido.append((eps, *fila))
        print(f"{eps:>8.2f}{fila[0]:>11.0f} %{fila[1]:>10.2f}{fila[2]:>10.2f}"
              f"{fila[3]:>10.2f}{fila[4]:>11.2f}{fila[5]:>11.2f}")
    print("-" * 74)
    print("El Q-learning offline solo deja de mentir cuando la cobertura llega al 100 %:")
    print("basta un puñado de pares (s,a) sin datos para que el max se agarre a ellos.")
    print("El problema nunca fue el algoritmo, era el hueco. Fíjate también en la")
    print("clonación: por muchos datos que le des se queda clavada en el techo de quien")
    print("los generó, y con eps = 1 se hunde con él. Y en la otra punta, con muy poca")
    print("cobertura, ser conservador tampoco regala nada: si D no contiene material")
    print("mejor, no hay nada que cocinar. Ningún método gana siempre: depende de D.")
    return {"mdp": mdp, "pi_beta": pi_beta, "pol_h": pol_h, "N0": datos[0][1],
            "politicas0": politicas0, "barrido": barrido, "V_opt": V_opt}


# =============================================================================
# 5. La figura
# =============================================================================
def dibuja_tablero(ax, mdp, pi, titulo, N):
    ax.set_title(titulo, fontsize=9)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(FILAS - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for f in range(FILAS):
        for c in range(COLS):
            s = idx(f, c)
            ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, fill=False,
                                       edgecolor="#cbd5e0", lw=0.8))
            if (f, c) in TRAMPAS:
                ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, color="#f6b6b6"))
                ax.text(c, f, "X", ha="center", va="center", color="#7a1f1f")
            elif (f, c) == META:
                ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, color="#bfe3bf"))
                ax.text(c, f, "G", ha="center", va="center", fontweight="bold")
            else:
                if (N[s] == 0).any():
                    ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, color="#eef2f7"))
                ax.text(c, f, SIMBOLOS[int(np.argmax(pi[s]))], ha="center",
                        va="center", fontsize=12)
                if (f, c) == INICIO:
                    ax.text(c - 0.38, f - 0.28, "S", fontsize=8, color="#475569")


def dibuja(p1, p2):
    fig = plt.figure(figsize=(13.0, 8.2))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.35, 1.0], hspace=0.38, wspace=2.0,
                          left=0.06, right=0.985, top=0.93, bottom=0.06)

    # (A) LA IMAGEN DEL CAPÍTULO: lo que se cree frente a lo que obtiene.
    ax1 = fig.add_subplot(gs[0, 0:4])
    ms = p1["metodos"]
    x = np.arange(len(ms))
    ax1.bar(x - 0.2, [p1["cree"][m] for m in ms], 0.4, color=AMBAR,
            label="valor que se CREE")
    ax1.bar(x + 0.2, [p1["real"][m] for m in ms], 0.4, color=IND,
            label="retorno REAL al desplegarla")
    ax1.axhline(p1["V_opt"][0], ls="--", lw=1.2, color=VERDE,
                label=f"óptimo real ({p1['V_opt'][0]:.0f})")
    for i, m in enumerate(ms):
        ax1.annotate("", xy=(i + 0.2, p1["real"][m]), xytext=(i - 0.2, p1["cree"][m]),
                     arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.4))
        ax1.text(i - 0.2, p1["cree"][m] + 0.6, f"{p1['cree'][m]:.1f}", ha="center",
                 fontsize=8, color="#92400e")
        ax1.text(i + 0.2, p1["real"][m] + 0.6, f"{p1['real'][m]:.1f}", ha="center",
                 fontsize=8, color="#3730a3")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["BC", "Q offline", "Q + soporte", "CQL"], fontsize=8.5)
    ax1.set_ylim(-30.0, 2.5)
    ax1.set_ylabel("Valor desde la casilla 0")
    ax1.set_title("A) La cadena: se cree -2 y cosecha -22", fontsize=10.5)
    ax1.legend(fontsize=7.5, loc="lower center")
    ax1.grid(alpha=0.25, axis="y")

    # (B) El veneno entrando por la casilla 2 y subiendo hacia atrás.
    ax2 = fig.add_subplot(gs[0, 4:8])
    cas = np.arange(p1["n"])
    ax2.plot(cas, p1["V_opt"], "o--", color=VERDE, lw=1.6, label="V*(s) real")
    ax2.plot(cas, p1["Q_ing"].max(axis=1), "o-", color=ROJO, lw=2,
             label="V(s) que cree Q offline")
    ax2.plot(cas, p1["Q_cql"].max(axis=1), "o-", color=IND, lw=2,
             label="V(s) que cree CQL")
    ax2.axvline(2, color=ROJO, ls=":", lw=1.2)
    ax2.annotate("(2, ATAJAR) no está en D:\nsu valor sigue siendo 0,00\nporque nadie lo desmintió",
                 xy=(2.0, 0.15), xytext=(1.55, 3.0), fontsize=7.5, color=ROJO,
                 ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.2))
    ax2.annotate("", xy=(0.06, -7.4), xytext=(1.65, -7.4),
                 arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.6))
    ax2.text(0.72, -7.25, "el bootstrapping lo arrastra atrás",
             fontsize=7, color=ROJO, ha="center", va="bottom")
    ax2.set_xticks(cas)
    ax2.set_ylim(-8.6, 4.2)
    ax2.set_xlabel("Casilla de la cadena")
    ax2.set_ylabel("Valor del estado")
    ax2.set_title("B) El veneno entra por la casilla 2", fontsize=10.5)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(alpha=0.25)

    # (C) El barrido de cobertura del gridworld: no era el algoritmo, era el hueco.
    ax3 = fig.add_subplot(gs[0, 8:12])
    b = np.array(p2["barrido"])
    xs = np.arange(len(b))
    ax3.plot(xs, b[:, 3], "o-", color=AMBAR, label="Q offline: cree")
    ax3.plot(xs, b[:, 4], "o--", color=AMBAR, label="Q offline: obtiene")
    ax3.plot(xs, b[:, 5], "s-", color=CIAN, label="CQL: cree")
    ax3.plot(xs, b[:, 6], "s--", color=CIAN, label="CQL: obtiene")
    ax3.plot(xs, b[:, 2], "^:", color=GRIS, label="clonación: obtiene")
    opt = p2["V_opt"][p2["mdp"].s0]
    ax3.axhline(opt, ls="--", lw=1.2, color=VERDE)
    ax3.text(0.05, opt + 0.6, f"óptimo real ({opt:.1f})", color=VERDE, fontsize=7.5)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"{r[0]:.2f}\n{r[1]:.0f} %" for r in b], fontsize=7.5)
    ax3.set_xlabel("epsilon de pi_beta  /  cobertura de D")
    ax3.set_ylabel("Retorno descontado")
    ax3.set_ylim(-31.0, 2.5)
    ax3.set_title("C) El gridworld: no era el algoritmo,\nera el hueco", fontsize=10.5)
    ax3.legend(fontsize=7, loc="lower left", ncol=2)
    ax3.grid(alpha=0.25)

    # Abajo: los tableros. Gris = estados con alguna acción que D nunca vio.
    tableros = [("pi_beta (generó D)", p2["pi_beta"]),
                ("Clonación de conducta", p2["politicas0"]["Clonación de conducta"]),
                ("Q-learning offline", p2["politicas0"]["Q-learning offline"]),
                ("CQL tabular", p2["politicas0"]["CQL tabular"])]
    for i, (titulo, pi) in enumerate(tableros):
        ax = fig.add_subplot(gs[1, 3 * i:3 * i + 3])
        dibuja_tablero(ax, p2["mdp"], pi, titulo, p2["N0"])
    fig.text(0.5, 0.015, "Gris = estados donde alguna acción no aparece en el conjunto "
             "de datos.  X = trampa,  G = meta,  S = salida.", ha="center",
             fontsize=8.5, color="#475569")
    plt.show()


if __name__ == "__main__":
    resumen1 = parte1()
    resumen2 = parte2()
    dibuja(resumen1, resumen2)
