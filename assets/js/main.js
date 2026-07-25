/* =========================================================================
   main.js — tema, navegación móvil, barra de progreso, resaltado de sintaxis,
   utilidades (copiar / descargar). Sin dependencias externas.
   ========================================================================= */
(function () {
  "use strict";

  /* ---------- Tema claro / oscuro ---------- */
  const root = document.documentElement;
  const stored = localStorage.getItem("rl-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-theme", stored || (prefersDark ? "dark" : "light"));

  function toggleTheme() {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("rl-theme", next);
    updateThemeButtons();
  }
  function updateThemeButtons() {
    const isDark = root.getAttribute("data-theme") === "dark";
    document.querySelectorAll("[data-theme-btn]").forEach((b) => {
      b.innerHTML = isDark ? "☀️ <span>Claro</span>" : "🌙 <span>Oscuro</span>";
    });
  }
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-theme-btn]")) toggleTheme();
  });

  /* ---------- Menú lateral (móvil) ---------- */
  function setupMobileNav() {
    const sidebar = document.querySelector(".sidebar");
    const overlay = document.querySelector(".overlay");
    document.addEventListener("click", (e) => {
      if (e.target.closest("[data-menu-btn]")) {
        sidebar && sidebar.classList.toggle("open");
        overlay && overlay.classList.toggle("show");
      } else if (e.target.classList.contains("overlay")) {
        sidebar && sidebar.classList.remove("open");
        overlay && overlay.classList.remove("show");
      }
    });
    // Cierra al navegar
    document.querySelectorAll(".nav a").forEach((a) =>
      a.addEventListener("click", () => {
        if (window.innerWidth <= 980) {
          sidebar && sidebar.classList.remove("open");
          overlay && overlay.classList.remove("show");
        }
      })
    );
  }

  /* ---------- Barra de progreso de lectura ---------- */
  function setupProgressBar() {
    const bar = document.querySelector(".progress-bar");
    if (!bar) return;
    const onScroll = () => {
      const h = document.documentElement;
      const scrolled = h.scrollTop / (h.scrollHeight - h.clientHeight || 1);
      bar.style.width = Math.min(100, Math.max(0, scrolled * 100)) + "%";
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  /* ---------- Resalta el enlace activo en la barra lateral ---------- */
  function highlightActiveNav() {
    const path = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".nav a").forEach((a) => {
      const href = a.getAttribute("href");
      if (href && href.split("/").pop() === path) {
        a.classList.add("active");
        a.scrollIntoView({ block: "nearest" });
      }
    });
  }

  /* ---------- Resaltado de sintaxis Python (ligero, sin dependencias) ---------- */
  const PY_KEYWORDS = new Set(("False None True and as assert async await break class continue def del " +
    "elif else except finally for from global if import in is lambda nonlocal not or pass raise return " +
    "try while with yield match case").split(" "));
  const PY_BUILTINS = new Set(("print range len int float str list dict tuple set bool abs min max sum " +
    "enumerate zip map filter sorted reversed round type isinstance super property staticmethod " +
    "classmethod open format np random plt torch nn gym gymnasium self").split(" "));

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlightPython(src) {
    // Tokeniza por líneas para tratar comentarios y strings de forma sencilla.
    const out = [];
    const lines = src.split("\n");
    for (let line of lines) {
      let res = "";
      let i = 0;
      while (i < line.length) {
        const rest = line.slice(i);
        // Comentario
        if (line[i] === "#") { res += '<span class="tok-com">' + escapeHtml(line.slice(i)) + "</span>"; break; }
        // Strings (triple no soportado por simplicidad de línea; suficiente para snippets)
        const strM = rest.match(/^(f|r|b|rb|fr)?("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/);
        if (strM) { res += '<span class="tok-str">' + escapeHtml(strM[0]) + "</span>"; i += strM[0].length; continue; }
        // Números
        const numM = rest.match(/^(\d+\.?\d*(e[-+]?\d+)?|\.\d+)/i);
        if (numM) { res += '<span class="tok-num">' + escapeHtml(numM[0]) + "</span>"; i += numM[0].length; continue; }
        // Identificadores / palabras
        const idM = rest.match(/^[A-Za-z_]\w*/);
        if (idM) {
          const w = idM[0];
          const after = line[i + w.length];
          if (PY_KEYWORDS.has(w)) res += '<span class="tok-kw">' + w + "</span>";
          else if (after === "(") res += '<span class="tok-fn">' + w + "</span>";
          else if (PY_BUILTINS.has(w)) res += '<span class="tok-bi">' + w + "</span>";
          else res += escapeHtml(w);
          i += w.length; continue;
        }
        res += escapeHtml(line[i]); i++;
      }
      out.push(res);
    }
    return out.join("\n");
  }

  /* ---------- Código editable en la propia página ----------
     Los bloques de código son editables para que los «🧪 Experimenta» de cada
     capítulo se puedan hacer aquí mismo: cambias un número, pulsas ▶ y ves qué
     pasa. Los cambios NO se guardan (al recargar vuelve el original) y hay un
     botón «↺ Original» para deshacerlo todo de golpe.

     Detalle de implementación: mientras escribes no re-coloreamos (sería saltón
     y caro en los ejemplos largos); normalizamos y volvemos a colorear al salir
     del bloque, que es también justo antes de que se ejecute. */

  // ¿Soporta el navegador contenteditable="plaintext-only"? Si no, usamos "true"
  // y nos encargamos nosotros del Enter.
  const PLANO = (() => {
    const d = document.createElement("div");
    try { d.contentEditable = "plaintext-only"; } catch (_) { return false; }
    return d.contentEditable === "plaintext-only";
  })();

  /* Lee el contenido como texto plano. No basta textContent: al pulsar Enter el
     navegador puede insertar <br> o <div>, y textContent se comería el salto. */
  function textoDe(el) {
    let out = "";
    (function walk(node) {
      node.childNodes.forEach((n) => {
        if (n.nodeType === 3) out += n.nodeValue;
        else if (n.nodeName === "BR") out += "\n";
        else {
          if (/^(DIV|P)$/.test(n.nodeName) && out && !out.endsWith("\n")) out += "\n";
          walk(n);
        }
      });
    })(el);
    return out;
  }

  function marcarModificado(ex) {
    const code = ex.querySelector("pre.code-source code");
    const btn = ex.querySelector(".btn-restore");
    if (!code || !btn) return;
    const cambiado = textoDe(code) !== code.dataset.original;
    ex.classList.toggle("modificado", cambiado);
    btn.hidden = !cambiado;
  }

  function repinta(code) {
    const txt = textoDe(code);
    code.innerHTML = highlightPython(txt);
    return txt;
  }

  function prepareCodeBlocks() {
    document.querySelectorAll(".code-example").forEach((ex) => {
      const code = ex.querySelector("pre.code-source code");
      if (!code || code.dataset.listo) return;

      // El original intacto, para poder restaurarlo siempre.
      code.dataset.original = code.textContent;
      code.innerHTML = highlightPython(code.dataset.original);
      code.dataset.listo = "1";

      code.setAttribute("contenteditable", PLANO ? "plaintext-only" : "true");
      code.setAttribute("spellcheck", "false");
      code.setAttribute("role", "textbox");
      code.setAttribute("aria-multiline", "true");
      code.setAttribute("aria-label", "Código de ejemplo, editable");
      code.setAttribute("autocapitalize", "off");
      code.setAttribute("autocorrect", "off");

      // Pista visual de que se puede tocar.
      const nombre = ex.querySelector(".file-name");
      if (nombre && !nombre.querySelector(".editable-hint")) {
        const hint = document.createElement("span");
        hint.className = "editable-hint";
        hint.textContent = "editable";
        hint.title = "Puedes cambiar el código y volver a ejecutarlo. No se guarda: al recargar vuelve el original.";
        nombre.appendChild(hint);
      }

      // Botón de restauración, oculto mientras no haya cambios.
      const acciones = ex.querySelector(".code-actions");
      if (acciones && !acciones.querySelector(".btn-restore")) {
        const btn = document.createElement("button");
        btn.className = "btn-restore";
        btn.type = "button";
        btn.hidden = true;
        btn.title = "Descartar mis cambios y volver al código original";
        btn.innerHTML = "↺ <span>Original</span>";
        btn.addEventListener("click", () => {
          code.innerHTML = highlightPython(code.dataset.original);
          marcarModificado(ex);
          code.focus();
        });
        acciones.insertBefore(btn, acciones.querySelector(".btn-copy") || null);
      }

      code.addEventListener("input", () => marcarModificado(ex));

      // Al salir del bloque: normalizamos lo que haya metido el navegador y
      // devolvemos el coloreado.
      code.addEventListener("blur", () => {
        repinta(code);
        marcarModificado(ex);
      });

      code.addEventListener("keydown", (e) => {
        // Tab indenta (en Python es esencial) en vez de saltar de foco.
        if (e.key === "Tab") {
          e.preventDefault();
          document.execCommand("insertText", false, "    ");
          return;
        }
        // Sin plaintext-only, el Enter del navegador mete <div>/<br>: lo forzamos.
        if (e.key === "Enter" && !PLANO) {
          e.preventDefault();
          document.execCommand("insertText", false, "\n");
        }
      });

      // Pegar siempre como texto plano, nunca con formato.
      code.addEventListener("paste", (e) => {
        e.preventDefault();
        const t = (e.clipboardData || window.clipboardData).getData("text/plain");
        document.execCommand("insertText", false, t);
      });
    });
  }

  /* ---------- Copiar y descargar ---------- */
  function getCodeText(exampleEl) {
    const code = exampleEl.querySelector("pre.code-source code");
    return code ? textoDe(code) : "";
  }
  function setupCodeButtons() {
    document.querySelectorAll(".code-example").forEach((ex) => {
      const copyBtn = ex.querySelector(".btn-copy");
      const dlBtn = ex.querySelector(".btn-download");
      if (copyBtn) copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(getCodeText(ex));
          const t = copyBtn.innerHTML; copyBtn.innerHTML = "✓ Copiado";
          setTimeout(() => (copyBtn.innerHTML = t), 1400);
        } catch (_) {}
      });
      if (dlBtn) dlBtn.addEventListener("click", () => {
        const name = (ex.querySelector(".file-name")?.textContent || "ejemplo.py").trim();
        const blob = new Blob([getCodeText(ex)], { type: "text/x-python" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = name; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
    });
  }

  /* ---------- Init ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    updateThemeButtons();
    setupMobileNav();
    setupProgressBar();
    highlightActiveNav();
    prepareCodeBlocks();
    setupCodeButtons();
  });

  // Exponer utilidades para el runner
  window.RL = { getCodeText, escapeHtml, textoDe };
})();
