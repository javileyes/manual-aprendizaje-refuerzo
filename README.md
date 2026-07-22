# 🎯 Manual de Aprendizaje por Refuerzo

Un manual web **interactivo** que explica el Aprendizaje por Refuerzo (RL) **de la
intuición a las matemáticas**, con ejemplos en Python que puedes **ejecutar en el
propio navegador** (gracias a [Pyodide](https://pyodide.org)) o descargar y correr
en tu **terminal**.

El objetivo: que termines entendiendo *de verdad* las matemáticas y los algoritmos —
desde los bandidos multibrazo hasta PPO y el RLHF que hay detrás de los grandes
modelos de lenguaje— apoyándote siempre en explicaciones intuitivas y ejemplos.

- **22 capítulos** en 5 partes, del bucle agente-entorno a RLHF.
- **27 scripts de Python**, todos ejecutados y verificados.
- Los algoritmos profundos van en **doble versión**: una *desde cero* en NumPy
  (ejecutable en el navegador) y otra idiomática con **PyTorch + Gymnasium**.

---

## 🚀 Cómo abrir el manual (el sitio web)

El manual es un sitio **estático** (HTML + CSS + JS, sin backend). Recomendado:

```bash
cd AprendizajeRefuerzo
python3 -m http.server 8000
# Abre http://localhost:8000 en tu navegador
```

> ℹ️ La primera vez que pulses **«▶ Ejecutar»** en un ejemplo, el navegador descarga el
> intérprete de Python (~10 MB). A partir de ahí es instantáneo. Necesita conexión a
> internet para esa primera descarga y para las librerías (numpy, matplotlib).

También puedes abrir `index.html` con doble clic (el texto y las matemáticas se ven
perfectos; para ejecutar Python en el navegador es mejor el servidor local de arriba).

---

## 🐍 Cómo ejecutar los ejemplos en tu terminal (entorno virtual)

Todos los ejemplos están en [`code/`](code/), organizados por capítulo, y también se
pueden **copiar o descargar** desde cada página del manual.

### 1. Requisito: versión de Python

Usa **Python 3.10 – 3.13**. PyTorch todavía no publica ruedas (*wheels*) para las
versiones más nuevas del intérprete (p. ej. 3.14), así que si tu `python3` es 3.14 o
posterior, crea el entorno con una versión soportada:

```bash
python3 --version            # comprueba tu versión

# Si es 3.14+, usa una versión concreta soportada por torch, por ejemplo:
python3.13 -m venv .venv     # o python3.12 / python3.11 / python3.10
```

### 2. Crea el entorno virtual e instala las dependencias

```bash
cd AprendizajeRefuerzo

# crea el entorno virtual (usa python3.13 si tu python3 es 3.14+)
python3 -m venv .venv

# actívalo
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell/CMD)

# actualiza pip e instala todo
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instala:

| Paquete       | Para qué                                                        |
|---------------|-----------------------------------------------------------------|
| `numpy`       | Núcleo de casi todos los ejemplos (y de los del navegador).     |
| `matplotlib`  | Las gráficas de todos los ejemplos.                             |
| `torch`       | Deep RL «de verdad» (DQN, PPO, SAC…) — capítulos 12+.           |
| `gymnasium`   | Entornos estándar (CartPole, Pendulum…).                        |
| `pygame`      | Solo para *renderizar* algunos entornos clásicos de control.    |

> 💡 La descarga de `torch` es de varios cientos de MB; la primera instalación puede
> tardar unos minutos.

### 3. Ejecuta cualquier ejemplo

Con el entorno **activado** (`source .venv/bin/activate`):

```bash
# Ejemplos "desde cero" en NumPy (parte 0-3 y versiones didácticas):
python code/02-bandido-multibrazo/bandidos.py
python code/09-q-learning/q_learning_cliff.py
python code/16-ppo/ppo_desde_cero.py

# Ejemplos con PyTorch + Gymnasium (los ficheros *_torch.py):
python code/12-dqn/dqn_torch.py
python code/14-policy-gradient-reinforce/reinforce_torch.py
python code/17-control-continuo/sac_torch.py
```

Los ejemplos que generan una gráfica abrirán una ventana de matplotlib al terminar.
Para ejecutarlos **sin ventana** (por ejemplo, en un servidor), usa el backend `Agg`:

```bash
MPLBACKEND=Agg python code/12-dqn/dqn_torch.py
```

### Dos familias de ejemplos

- **`*.py` (NumPy):** implementaciones *desde cero*. Son las que corren en el navegador
  y las más didácticas: verás cada línea del algoritmo, sin cajas negras.
- **`*_torch.py` (PyTorch + Gymnasium):** la versión idiomática y práctica que usarías
  en el mundo real. Demasiado pesadas para el navegador; se ejecutan en la terminal.

---

## 🗺️ Estructura del proyecto

```
AprendizajeRefuerzo/
├── index.html                 # Portada + índice completo
├── README.md
├── requirements.txt
├── .venv/                     # Entorno virtual (lo creas tú; ignorado por git)
├── assets/
│   ├── css/styles.css         # Sistema de diseño (tema claro/oscuro)
│   └── js/
│       ├── nav.js             # Índice: barra lateral y navegación (fuente única)
│       ├── main.js            # Tema, resaltado de sintaxis, copiar/descargar
│       └── pyodide-runner.js  # Ejecuta Python en el navegador
├── chapters/                  # Un archivo HTML por capítulo (01…21)
└── code/                      # Los mismos ejemplos como scripts .py por capítulo
```

---

## 📚 Índice

**Parte 0 · Intuición** — 1) ¿Qué es el RL? · 2) Bandido multibrazo · 3) MDP · 4) Retornos y valor
**Parte 1 · Tabular** — 5) Programación dinámica · 6) Monte Carlo · 7) TD(0) · 8) SARSA · 9) Q-Learning · 10) TD(λ)
**Parte 2 · Deep RL** — 11) Aproximación de funciones · 12) DQN · 13) Mejoras de DQN
**Parte 3 · Política** — 14) REINFORCE · 15) Actor-Crítico · 16) PPO · 17) Control continuo
**Parte 4 · Fronteras** — 18) Model-based · 19) Exploración avanzada · 20) Máxima entropía y empowerment · 21) RLHF · 22) Panorama

Empieza por [`chapters/01-que-es-rl.html`](chapters/01-que-es-rl.html).

---

## 🛠️ Tecnología

- **Sin frameworks ni build**: HTML/CSS/JS puro.
- **Matemáticas**: [MathJax 3](https://www.mathjax.org/).
- **Python en el navegador**: [Pyodide](https://pyodide.org) (WebAssembly).
- **Deep RL en terminal**: [PyTorch](https://pytorch.org) + [Gymnasium](https://gymnasium.farama.org/).

Hecho para aprender haciendo. 🧠
