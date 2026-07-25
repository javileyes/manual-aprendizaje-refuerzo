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

  /* ---------- Mis cambios: se guardan solos, y se exportan a un .json ----------
     Los cambios que haces en cualquier bloque de código se guardan en el propio
     navegador (localStorage), así que sobreviven a recargar y a cambiar de
     capítulo. Desde «Mis cambios» puedes bajártelos TODOS a un único fichero
     .json y volver a importarlos —en otro navegador, en otro ordenador, o para
     tener varias tandas de pruebas guardadas—.

     La clave de cada bloque es el NOMBRE DEL FICHERO (pasillo.py, ppo_desde_cero.py…),
     que es único en todo el manual y no depende del número del capítulo: si algún
     día se renumera un capítulo, tus cambios siguen encontrando su sitio. */
  const ALMACEN = "rl-ediciones";
  const FORMATO = "manual-rl-ediciones";
  const FICHERO_EXPORT = "manual-rl-mis-cambios.json";

  /* Huella corta del código ORIGINAL, para avisarte si el manual cambió ese
     ejemplo después de que tú guardaras tu versión. */
  function huella(s) {
    let h = 5381;
    for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
    return (h >>> 0).toString(16);
  }

  function leeEdiciones() {
    try {
      const v = JSON.parse(localStorage.getItem(ALMACEN));
      return v && typeof v === "object" ? v : {};
    } catch (_) { return {}; }
  }
  function escribeEdiciones(obj) {
    try { localStorage.setItem(ALMACEN, JSON.stringify(obj)); return true; }
    catch (_) { return false; }   // cuota llena, modo privado, file:// restringido…
  }

  function idDe(ex) {
    const n = ex.querySelector(".file-name");
    if (!n) return null;
    // .textContent incluiría la pastilla «editable»; nos quedamos con el primer nodo de texto.
    const t = [...n.childNodes].find((x) => x.nodeType === 3);
    return (t ? t.nodeValue : n.textContent).trim();
  }

  function guardaEdicion(ex) {
    const code = ex.querySelector("pre.code-source code");
    const id = code && code.dataset.id;
    if (!id) return;
    const txt = textoDe(code);
    const st = leeEdiciones();
    if (txt === code.dataset.original) delete st[id];
    else st[id] = {
      codigo: txt,
      huella: huella(code.dataset.original),
      capitulo: document.body.dataset.chapter || "",
      pagina: (location.pathname.split("/").pop() || ""),
      titulo: (document.title.split("·")[0] || "").trim(),
    };
    if (!escribeEdiciones(st)) avisaFalloAlGuardar(ex);
    refrescaBotonCambios();
  }

  let yaAvisado = false;
  function avisaFalloAlGuardar(ex) {
    if (yaAvisado) return;
    yaAvisado = true;
    const o = document.createElement("p");
    o.className = "pyodide-banner";
    o.textContent = "⚠️ No he podido guardar tus cambios en este navegador (puede ser el modo privado o la " +
      "cuota llena). Puedes seguir editando y ejecutando, pero se perderán al recargar: exporta el .json si quieres conservarlos.";
    ex.insertAdjacentElement("afterend", o);
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
      code.dataset.id = idDe(ex) || "";
      code.dataset.listo = "1";

      // ¿Hay una versión tuya guardada de este ejemplo? Se aplica al abrir.
      const guardada = leeEdiciones()[code.dataset.id];
      if (guardada && typeof guardada.codigo === "string") {
        code.innerHTML = highlightPython(guardada.codigo);
        if (guardada.huella && guardada.huella !== huella(code.dataset.original)) {
          ex.classList.add("desfasado");
        }
      } else {
        code.innerHTML = highlightPython(code.dataset.original);
      }

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
          ex.classList.remove("desfasado");
          marcarModificado(ex);
          guardaEdicion(ex);
          code.focus();
        });
        acciones.insertBefore(btn, acciones.querySelector(".btn-copy") || null);
      }

      let temporizador = null;
      code.addEventListener("input", () => {
        marcarModificado(ex);
        clearTimeout(temporizador);          // guardamos al parar de teclear
        temporizador = setTimeout(() => guardaEdicion(ex), 600);
      });

      // Al salir del bloque: normalizamos lo que haya metido el navegador y
      // devolvemos el coloreado.
      code.addEventListener("blur", () => {
        repinta(code);
        marcarModificado(ex);
        clearTimeout(temporizador);
        guardaEdicion(ex);
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

      // Si venía una versión tuya guardada, el bloque ya nace «modificado».
      marcarModificado(ex);
    });
  }

  /* ---------- Panel «Mis cambios»: exportar / importar / restaurar ---------- */
  const enCapitulos = location.pathname.includes("/chapters/");
  const hrefPagina = (p) => (enCapitulos ? p : "chapters/" + p);

  function refrescaBotonCambios() {
    const btn = document.querySelector(".btn-cambios");
    if (!btn) return;
    const n = Object.keys(leeEdiciones()).length;
    btn.querySelector(".cc-num").textContent = n ? ` (${n})` : "";
    btn.classList.toggle("hay-cambios", n > 0);
  }

  /* Vuelve a pintar los bloques de ESTA página con lo que haya guardado. */
  function aplicaEdicionesEnPagina() {
    const st = leeEdiciones();
    document.querySelectorAll(".code-example").forEach((ex) => {
      const code = ex.querySelector("pre.code-source code");
      if (!code || !code.dataset.listo) return;
      const g = st[code.dataset.id];
      const txt = g && typeof g.codigo === "string" ? g.codigo : code.dataset.original;
      code.innerHTML = highlightPython(txt);
      ex.classList.toggle("desfasado",
        !!(g && g.huella && g.huella !== huella(code.dataset.original)));
      marcarModificado(ex);
    });
  }

  function exportaCambios() {
    const ediciones = leeEdiciones();
    if (!Object.keys(ediciones).length) return;
    const datos = { formato: FORMATO, version: 1, guardado: new Date().toISOString(), ediciones };
    const blob = new Blob([JSON.stringify(datos, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = FICHERO_EXPORT; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function importaCambios(file, avisar) {
    const lector = new FileReader();
    lector.onload = () => {
      let d;
      try { d = JSON.parse(lector.result); }
      catch (_) { return avisar("Ese archivo no es un JSON válido.", true); }
      if (!d || d.formato !== FORMATO || !d.ediciones || typeof d.ediciones !== "object") {
        return avisar("Ese archivo no parece un fichero de cambios del manual.", true);
      }
      const st = leeEdiciones();
      let n = 0, desfasados = 0;
      Object.keys(d.ediciones).forEach((id) => {
        const e = d.ediciones[id];
        if (!e || typeof e.codigo !== "string") return;
        st[id] = e; n++;
        const aqui = document.querySelector(`pre.code-source code[data-id="${CSS.escape(id)}"]`);
        if (aqui && e.huella && e.huella !== huella(aqui.dataset.original)) desfasados++;
      });
      if (!n) return avisar("El archivo no contenía ningún cambio.", true);
      if (!escribeEdiciones(st)) return avisar("No he podido guardarlos en este navegador.", true);
      aplicaEdicionesEnPagina();
      refrescaBotonCambios();
      avisar(`Importados ${n} ejemplo${n === 1 ? "" : "s"}.` +
        (desfasados ? " Ojo: alguno se guardó sobre una versión anterior del manual." : "") +
        " Los de otros capítulos se aplicarán al abrirlos.", false);
    };
    lector.onerror = () => avisar("No he podido leer el archivo.", true);
    lector.readAsText(file);
  }

  function montaPanelCambios() {
    const tools = document.querySelector(".sidebar-tools");
    if (!tools || tools.querySelector(".btn-cambios")) return;

    const btn = document.createElement("button");
    btn.className = "btn-cambios";
    btn.type = "button";
    btn.innerHTML = '💾 <span>Mis cambios</span><span class="cc-num"></span>';
    btn.title = "Guardar en un fichero los cambios que has hecho en el código, o recuperarlos";
    tools.appendChild(btn);

    const dlg = document.createElement("dialog");
    dlg.className = "cambios-dlg";
    document.body.appendChild(dlg);

    const aviso = (txt, malo) => {
      // Primero refrescamos la lista (pinta() rehace el HTML del panel) y solo
      // después escribimos el mensaje, o se lo llevaría por delante.
      if (!malo) pinta();
      const p = dlg.querySelector(".cc-aviso");
      if (!p) return;
      p.textContent = txt;
      p.className = "cc-aviso" + (malo ? " malo" : " bien");
      p.hidden = false;
    };

    function pinta() {
      const st = leeEdiciones();
      const ids = Object.keys(st).sort((a, b) =>
        (parseInt(st[a].capitulo, 10) || 0) - (parseInt(st[b].capitulo, 10) || 0));
      const lista = ids.length
        ? `<ul class="cc-lista">` + ids.map((id) => {
            const e = st[id];
            const donde = e.pagina
              ? `<a href="${hrefPagina(e.pagina)}">${e.capitulo ? "Cap. " + e.capitulo + " · " : ""}${e.titulo || e.pagina}</a>`
              : (e.titulo || "");
            return `<li><code>${escapeHtml(id)}</code><span>${donde}</span></li>`;
          }).join("") + `</ul>`
        : `<p class="cc-vacio">Todavía no has cambiado ningún ejemplo. Edita cualquier bloque de
             código del manual y aparecerá aquí.</p>`;

      dlg.innerHTML = `
        <form method="dialog" class="cc-cerrar"><button aria-label="Cerrar">✕</button></form>
        <h3>Mis cambios en el código</h3>
        <p class="cc-intro">
          Lo que editas en cualquier bloque de código se guarda en este navegador y se conserva al
          recargar y al cambiar de capítulo. Aquí puedes bajarlo todo a un fichero
          <code>.json</code> y volver a cargarlo cuando quieras —en otro ordenador, o para tener
          varias tandas de pruebas guardadas—.
        </p>
        <div class="cc-cuenta">${ids.length
          ? `<strong>${ids.length}</strong> ejemplo${ids.length === 1 ? "" : "s"} con cambios tuyos`
          : "Sin cambios guardados"}</div>
        ${lista}
        <p class="cc-aviso" hidden></p>
        <div class="cc-acciones">
          <button type="button" class="cc-exportar" ${ids.length ? "" : "disabled"}>⬇ Exportar .json</button>
          <button type="button" class="cc-importar">⬆ Importar .json</button>
          <button type="button" class="cc-reset" ${ids.length ? "" : "disabled"}>↺ Restaurar todo</button>
        </div>
        <input type="file" accept="application/json,.json" class="cc-fichero" hidden />`;

      dlg.querySelector(".cc-exportar").addEventListener("click", exportaCambios);
      dlg.querySelector(".cc-importar").addEventListener("click", () => dlg.querySelector(".cc-fichero").click());
      dlg.querySelector(".cc-fichero").addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) importaCambios(e.target.files[0], aviso);
        e.target.value = "";
      });
      dlg.querySelector(".cc-reset").addEventListener("click", () => {
        if (!window.confirm("Se descartarán TODOS tus cambios en el código de todo el manual. " +
                            "Si quieres conservarlos, expórtalos antes. ¿Seguimos?")) return;
        escribeEdiciones({});
        aplicaEdicionesEnPagina();
        refrescaBotonCambios();
        aviso("Listo: todo el manual vuelve a su código original.", false);
      });
    }

    btn.addEventListener("click", () => {
      pinta();
      if (typeof dlg.showModal === "function") dlg.showModal();
      else dlg.setAttribute("open", "");     // navegadores sin <dialog> modal
    });
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });

    refrescaBotonCambios();
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

  /* ---------- Init ----------
     Cada pieza va en su propio try: si una falla (un navegador viejo, una API
     que no existe), las demás siguen funcionando. Lo importante es que los
     bloques de código queden siempre editables y ejecutables. */
  function arranca(nombre, fn) {
    try { fn(); }
    catch (e) { console.error("Manual de RL · fallo en " + nombre + ":", e); }
  }

  document.addEventListener("DOMContentLoaded", () => {
    arranca("tema", updateThemeButtons);
    arranca("menú móvil", setupMobileNav);
    arranca("barra de progreso", setupProgressBar);
    arranca("navegación", highlightActiveNav);
    arranca("bloques de código", prepareCodeBlocks);
    arranca("botones de código", setupCodeButtons);
    arranca("panel de cambios", montaPanelCambios);
  });

  // Exponer utilidades para el runner
  window.RL = { getCodeText, escapeHtml, textoDe };
})();
