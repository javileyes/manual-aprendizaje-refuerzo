/* =========================================================================
   pyodide-runner.js — Ejecuta los ejemplos de Python EN EL NAVEGADOR.
   Usa Pyodide (Python compilado a WebAssembly). Carga perezosa: Pyodide solo
   se descarga la primera vez que el usuario pulsa "Ejecutar".
   Captura stdout/stderr en vivo y renderiza figuras de matplotlib como PNG.
   ========================================================================= */
(function () {
  "use strict";

  const PYODIDE_VERSION = "0.26.4";
  const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

  let pyodidePromise = null;      // promesa singleton de carga de Pyodide
  const loadedPackages = new Set();
  let currentOutputEl = null;      // dónde escribir stdout mientras se ejecuta

  /* -------- Carga del script de Pyodide desde el CDN -------- */
  function loadPyodideScript() {
    return new Promise((resolve, reject) => {
      if (window.loadPyodide) return resolve();
      const s = document.createElement("script");
      s.src = PYODIDE_URL + "pyodide.js";
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("No se pudo cargar Pyodide desde el CDN (¿sin conexión?)."));
      document.head.appendChild(s);
    });
  }

  /* -------- Inicializa Pyodide una sola vez -------- */
  async function getPyodide(statusCb) {
    if (pyodidePromise) return pyodidePromise;
    pyodidePromise = (async () => {
      statusCb && statusCb("Descargando el intérprete de Python (~10 MB, solo la primera vez)…");
      await loadPyodideScript();
      const py = await window.loadPyodide({ indexURL: PYODIDE_URL });
      // stdout/stderr en vivo hacia el bloque de salida activo.
      // OJO: el callback "batched" entrega cada línea SIN el salto final; hay que
      // volver a añadir "\n" o la salida saldría toda pegada en una sola línea.
      py.setStdout({ batched: (msg) => appendText(currentOutputEl, msg + "\n", false) });
      py.setStderr({ batched: (msg) => appendText(currentOutputEl, msg + "\n", true) });
      return py;
    })();
    return pyodidePromise;
  }

  /* -------- Asegura que los paquetes pedidos están cargados -------- */
  async function ensurePackages(py, pkgs, statusCb) {
    const need = pkgs.filter((p) => p && !loadedPackages.has(p));
    if (need.length === 0) return;
    statusCb && statusCb("Cargando paquetes: " + need.join(", ") + " …");
    await py.loadPackage(need);
    need.forEach((p) => loadedPackages.add(p));
    if (need.includes("matplotlib")) {
      // Backend sin ventana + show() como no-op silencioso (evita el UserWarning
      // de AGG "cannot show the figure"). Las figuras abiertas se capturan al
      // terminar en renderFigures(), así que show() no necesita hacer nada.
      await py.runPythonAsync(
        "import matplotlib; matplotlib.use('AGG')\n" +
        "import matplotlib.pyplot as _plt\n" +
        "_plt.show = lambda *a, **k: None"
      );
    }
  }

  /* -------- Utilidades de salida -------- */
  function outPre(outputEl, stderr) {
    let pre = outputEl.querySelector(stderr ? "pre.stderr" : "pre.stdout");
    if (!pre) {
      pre = document.createElement("pre");
      pre.className = stderr ? "stderr" : "stdout";
      outputEl.appendChild(pre);
    }
    return pre;
  }
  function appendText(outputEl, text, stderr) {
    if (!outputEl) return;
    const pre = outPre(outputEl, stderr);
    pre.appendChild(document.createTextNode(text));
    outputEl.scrollTop = outputEl.scrollHeight;
  }
  function setStatus(outputEl, text) {
    let row = outputEl.querySelector(".spinner-row");
    if (!row) {
      row = document.createElement("div");
      row.className = "spinner-row";
      row.innerHTML = '<span class="spinner"></span><span class="msg"></span>';
      outputEl.prepend(row);
    }
    row.querySelector(".msg").textContent = text;
  }
  function clearStatus(outputEl) {
    const row = outputEl.querySelector(".spinner-row");
    if (row) row.remove();
  }

  /* -------- Captura de figuras de matplotlib como imágenes PNG base64 -------- */
  async function renderFigures(py, outputEl) {
    if (!loadedPackages.has("matplotlib")) return;
    const b64list = await py.runPythonAsync(`
import base64, io
_imgs = []
try:
    import matplotlib.pyplot as _plt
    for _num in _plt.get_fignums():
        _fig = _plt.figure(_num)
        _buf = io.BytesIO()
        _fig.savefig(_buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
        _imgs.append(base64.b64encode(_buf.getvalue()).decode('ascii'))
    _plt.close('all')
except Exception:
    pass
_imgs
`);
    const arr = b64list.toJs ? b64list.toJs() : b64list;
    arr.forEach((b64) => {
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + b64;
      img.alt = "Figura generada por matplotlib";
      outputEl.appendChild(img);
    });
    if (b64list.destroy) b64list.destroy();
  }

  /* -------- Ejecuta un bloque -------- */
  async function runExample(exampleEl) {
    const btn = exampleEl.querySelector(".btn-run");
    let outputEl = exampleEl.querySelector(".code-output");
    if (!outputEl) {
      outputEl = document.createElement("div");
      outputEl.className = "code-output";
      exampleEl.appendChild(outputEl);
    }
    outputEl.hidden = false;
    outputEl.innerHTML = '<div class="output-label">Salida</div>';
    currentOutputEl = outputEl;

    const pkgs = (exampleEl.dataset.packages || "").split(",").map((s) => s.trim()).filter(Boolean);
    const code = window.RL.getCodeText(exampleEl);

    btn && (btn.disabled = true);
    const origLabel = btn ? btn.innerHTML : "";
    btn && (btn.innerHTML = "Ejecutando…");

    try {
      setStatus(outputEl, "Iniciando Python…");
      const py = await getPyodide((m) => setStatus(outputEl, m));
      currentOutputEl = outputEl; // por si otro bloque cambió el puntero
      await ensurePackages(py, pkgs, (m) => setStatus(outputEl, m));
      setStatus(outputEl, "Ejecutando el ejemplo…");

      // Espacio de nombres nuevo por ejecución (reproducible, sin fugas entre bloques).
      // __name__ = "__main__" para que el patrón  if __name__ == "__main__":  se
      // ejecute igual que en la terminal; si no, el bloque main no correría y el
      // ejemplo no imprimiría ni dibujaría nada.
      const ns = py.toPy({ __name__: "__main__" });
      try {
        await py.runPythonAsync(code, { globals: ns });
      } finally {
        clearStatus(outputEl);
      }
      await renderFigures(py, outputEl);
      ns.destroy && ns.destroy();

      if (!outputEl.querySelector("pre") && !outputEl.querySelector("img")) {
        appendText(outputEl, "(el programa terminó sin imprimir nada)", false);
      }
    } catch (err) {
      clearStatus(outputEl);
      const msg = (err && err.message) ? err.message : String(err);
      appendText(outputEl, "\n" + msg, true);
    } finally {
      btn && (btn.disabled = false);
      btn && (btn.innerHTML = origLabel || "▶ Ejecutar");
    }
  }

  /* -------- Enlaza los botones "Ejecutar" -------- */
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".code-example .btn-run").forEach((btn) => {
      btn.addEventListener("click", () => {
        const ex = btn.closest(".code-example");
        if (ex) runExample(ex);
      });
    });
  });
})();
