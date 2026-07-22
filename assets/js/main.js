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

  function applyHighlighting() {
    document.querySelectorAll("pre.code-source code").forEach((code) => {
      if (code.dataset.hl) return;
      const raw = code.textContent;
      code.dataset.raw = raw;
      code.innerHTML = highlightPython(raw);
      code.dataset.hl = "1";
    });
  }

  /* ---------- Copiar y descargar ---------- */
  function getCodeText(exampleEl) {
    const code = exampleEl.querySelector("pre.code-source code");
    return code ? (code.dataset.raw || code.textContent) : "";
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
    applyHighlighting();
    setupCodeButtons();
  });

  // Exponer utilidades para el runner
  window.RL = { getCodeText, escapeHtml };
})();
