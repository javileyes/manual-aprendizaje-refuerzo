# 🎯 Manual de Aprendizaje por Refuerzo

### 📖 [**Léelo en vivo aquí → javileyes.github.io/manual-aprendizaje-refuerzo**](https://javileyes.github.io/manual-aprendizaje-refuerzo/)

> Ábrelo en el navegador y **ejecuta el código Python sin instalar nada** (pulsa ▶ Ejecutar en cualquier ejemplo).
> El código además es **editable**: cambia lo que quieras y vuelve a ejecutarlo para hacer los
> «🧪 Experimenta» de cada capítulo ahí mismo. Y puedes **anotar cualquier párrafo**. Tus cambios
> y tus notas se guardan en el navegador y se exportan **juntos a un `.json`** desde
> «💾 Mis cambios», para llevártelos a otro sitio o guardar varias tandas de pruebas.

---

Un manual web **interactivo** que explica el Aprendizaje por Refuerzo (RL) **de la
intuición a las matemáticas**, con ejemplos en Python que puedes **ejecutar en el
propio navegador** (gracias a [Pyodide](https://pyodide.org)) o descargar y correr
en tu **terminal**.

El objetivo: que termines entendiendo *de verdad* las matemáticas y los algoritmos —
desde los bandidos multibrazo hasta PPO y el RLHF que hay detrás de los grandes
modelos de lenguaje— apoyándote siempre en explicaciones intuitivas y ejemplos.

- **23 capítulos** en 5 partes, del bucle agente-entorno a RLHF y al RL offline.
- **29 scripts de Python**, todos ejecutados y verificados: cada cifra y cada salida
  de terminal que aparece en el manual sale de una ejecución real, nunca escrita a mano.
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

### 4. Tu copia del manual: código editado y notas

**El código es editable en la propia página.** Cambia lo que quieras en cualquier bloque
y pulsa ▶ Ejecutar: corre tu versión. Cada bloque tiene un **↺ Original** que restaura
solo ese ejemplo.

**Casi cualquier cosa se puede anotar**: párrafos, fórmulas, títulos de sección, puntos de
lista, recuadros completos (🧠 ➗ 🔑 ⚠️ 🧪) y filas de tabla. Al pasar el ratón por encima
aparece un **✎ crear nota**; lo que ya tiene una lleva un **📝** —en el margen izquierdo, o
dentro del bloque si es un recuadro o una fila—. Pasando el cursor por ese 📝 se lee la nota
en un globo, sin abrir nada; haciendo clic se edita o se elimina.

| Qué | Cuántos |
|---|---|
| Párrafos | 1.200 |
| Recuadros | 413 |
| Puntos de lista | 396 |
| Títulos de sección | 287 |
| Fórmulas | 223 |
| Filas de tabla | 211 |
| **Total anotable** | **2.730** |

> ℹ️ Las filas de tabla son la única excepción al globo: la tabla vive en un contenedor con
> desplazamiento horizontal que **recorta** todo lo que se salga, así que el globo asomaría
> cortado. En ellas el 📝 abre directamente la nota.

Todo se guarda en el navegador y sobrevive a recargar y a cambiar de capítulo. Desde el
botón **💾 Mis cambios** de la barra lateral puedes:

- **⬇ Exportar `.json`** — baja a un único fichero `manual-rl-mis-cambios.json` *todo*
  lo tuyo: el código editado de todos los capítulos **y todas las notas**.
- **⬆ Importar `.json`** — recupéralo en otro navegador u otro ordenador. Guarda varios
  ficheros si quieres tener distintas tandas de pruebas.
- **↺ Restaurar código** y **🗑 Borrar notas**, por separado, para no llevarte una cosa
  por delante al querer descartar la otra.

> ℹ️ **Cómo se identifica cada cosa, y por qué.** Nada se identifica por su **posición**:
> así, insertar o quitar contenido en un capítulo no descoloca lo que hayas anotado más
> abajo. Los ejemplos de código van por su **nombre de fichero** (`ppo_desde_cero.py`), de
> modo que un renumerado de capítulos tampoco los deja huérfanos. Los títulos de sección,
> por su **`id`**. Las fórmulas, por su **código LaTeX**: su texto renderizado no distingue
> una fórmula de otra, así que las 223 compartirían clave. Y los párrafos, listas y
> recuadros, por una **huella de su texto**. Además se guarda una huella del original, de
> forma que si el manual cambia después ese ejemplo o ese párrafo se te avisa, en vez de
> aplicar tu versión sobre otra base o perderla en silencio.

**Qué aguantan tus notas si el manual se edita.** Medido sobre un capítulo real con una
nota en cada uno de sus 112 elementos anotables (`test_estabilidad.mjs`):

| Edición del manual | Notas que siguen en su sitio |
|---|---|
| Añadir párrafos —al principio, en medio, dentro de un recuadro— | 112/112 |
| Añadir una sección entera nueva | 112/112 |
| Reordenar secciones | 112/112 |
| Corregir una errata en otro párrafo | 112/112 |
| Cambiar el LaTeX de una fórmula sin nota | 112/112 |
| Borrar un párrafo | 111/112 — solo la suya |
| Editar el texto de un párrafo anotado | 111/112 — solo la suya |
| Cambiar el `id` de un título anotado | 111/112 — solo la suya |

Una nota que se desengancha **no se pierde**: sigue en «💾 Mis cambios», con su texto y con
el aviso «el párrafo ha cambiado», para que puedas releerla y volver a colocarla.

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
├── chapters/                  # Un archivo HTML por capítulo (01…23)
└── code/                      # Los mismos ejemplos como scripts .py por capítulo
```

---

## 📚 Índice

**Parte 0 · Intuición** — 1) ¿Qué es el RL? · 2) Bandido multibrazo · 3) MDP · 4) Retornos y valor
**Parte 1 · Tabular** — 5) Programación dinámica · 6) Monte Carlo · 7) TD(0) · 8) SARSA · 9) Q-Learning · 10) TD(λ)
**Parte 2 · Deep RL** — 11) Aproximación de funciones · 12) DQN · 13) Mejoras de DQN
**Parte 3 · Política** — 14) REINFORCE · 15) Actor-Crítico · 16) PPO · 17) Control continuo
**Parte 4 · Fronteras** — 18) Model-based · 19) Exploración avanzada · 20) Máxima entropía y empowerment · 21) RLHF · 22) RL offline · 23) Panorama

Empieza por [`chapters/01-que-es-rl.html`](chapters/01-que-es-rl.html).

---

## 🛠️ Tecnología

- **Sin frameworks ni build**: HTML/CSS/JS puro.
- **Matemáticas**: [MathJax 3](https://www.mathjax.org/).
- **Python en el navegador**: [Pyodide](https://pyodide.org) (WebAssembly).
- **Deep RL en terminal**: [PyTorch](https://pytorch.org) + [Gymnasium](https://gymnasium.farama.org/).

Hecho para aprender haciendo. 🧠
