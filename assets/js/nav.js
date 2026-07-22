/* =========================================================================
   nav.js — Fuente única de verdad del índice del manual.
   Construye: barra lateral, navegación anterior/siguiente e índice de portada.
   ========================================================================= */
(function () {
  "use strict";

  const PARTS = [
    { id: 0, title: "Parte 0 · Intuición" },
    { id: 1, title: "Parte 1 · Métodos tabulares clásicos" },
    { id: 2, title: "Parte 2 · Aprendizaje profundo (valor)" },
    { id: 3, title: "Parte 3 · Gradiente de política" },
    { id: 4, title: "Parte 4 · Fronteras del campo" },
  ];

  const CH = [
    { n: 1,  part: 0, slug: "01-que-es-rl",               title: "¿Qué es el aprendizaje por refuerzo?", desc: "Agente, entorno, recompensa: el bucle fundamental." },
    { n: 2,  part: 0, slug: "02-bandido-multibrazo",      title: "El bandido multibrazo",                 desc: "Explorar vs. explotar. ε-greedy, UCB." },
    { n: 3,  part: 0, slug: "03-mdp",                     title: "Procesos de Decisión de Markov",        desc: "El marco matemático: estados, acciones, transiciones." },
    { n: 4,  part: 0, slug: "04-retornos-y-valor",        title: "Retornos y funciones de valor",         desc: "Descuento, V, Q y las ecuaciones de Bellman." },

    { n: 5,  part: 1, slug: "05-programacion-dinamica",   title: "Programación dinámica",                 desc: "Planificar con el modelo: iteración de política y valor." },
    { n: 6,  part: 1, slug: "06-monte-carlo",             title: "Métodos de Monte Carlo",                desc: "Aprender de episodios completos, sin modelo." },
    { n: 7,  part: 1, slug: "07-td-aprendizaje",          title: "Diferencias temporales TD(0)",          desc: "Aprender en cada paso: bootstrapping." },
    { n: 8,  part: 1, slug: "08-sarsa",                   title: "SARSA",                                 desc: "Control TD on-policy." },
    { n: 9,  part: 1, slug: "09-q-learning",              title: "Q-Learning",                            desc: "Control TD off-policy: el clásico por excelencia." },
    { n: 10, part: 1, slug: "10-td-lambda",               title: "Trazas de elegibilidad TD(λ)",          desc: "El puente entre TD y Monte Carlo." },

    { n: 11, part: 2, slug: "11-aproximacion-funciones",  title: "Aproximación de funciones",             desc: "De la tabla a las features: generalizar." },
    { n: 12, part: 2, slug: "12-dqn",                     title: "Deep Q-Networks (DQN)",                 desc: "Una red neuronal como tabla Q. Replay y target." },
    { n: 13, part: 2, slug: "13-mejoras-dqn",             title: "Mejoras de DQN",                        desc: "Double, Dueling y Prioritized Replay." },

    { n: 14, part: 3, slug: "14-policy-gradient-reinforce", title: "Gradiente de política y REINFORCE",   desc: "Optimizar la política directamente." },
    { n: 15, part: 3, slug: "15-actor-critico",           title: "Actor-Crítico (A2C)",                   desc: "Un crítico reduce la varianza. Ventaja y GAE." },
    { n: 16, part: 3, slug: "16-ppo",                     title: "PPO",                                   desc: "El caballo de batalla moderno: pasos seguros." },
    { n: 17, part: 3, slug: "17-control-continuo",        title: "Control continuo: DDPG, TD3, SAC",      desc: "Acciones reales y continuas." },

    { n: 18, part: 4, slug: "18-model-based",             title: "RL basado en modelo",                   desc: "Imaginar antes de actuar: Dyna-Q, AlphaZero, MuZero." },
    { n: 19, part: 4, slug: "19-exploracion-avanzada",    title: "Exploración avanzada",                  desc: "Curiosidad y motivación intrínseca (RND, ICM)." },
    { n: 20, part: 4, slug: "20-maxent-empowerment",     title: "Máxima entropía y empowerment",         desc: "Explorar y mantener opciones abiertas como parte del objetivo." },
    { n: 21, part: 4, slug: "21-rlhf",                    title: "RLHF: el RL detrás de los LLM",         desc: "Alinear modelos con preferencias humanas." },
    { n: 22, part: 4, slug: "22-panorama",               title: "Panorama y buenas prácticas",           desc: "Evaluar bien, no engañarte, y hacia dónde va todo." },
  ];

  window.RL_CH = CH;
  window.RL_PARTS = PARTS;

  // ¿Estamos en la raíz o dentro de /chapters/?
  const inChapters = location.pathname.includes("/chapters/");
  const chHref = (slug) => (inChapters ? slug + ".html" : "chapters/" + slug + ".html");
  const homeHref = inChapters ? "../index.html" : "index.html";
  const asset = (p) => (inChapters ? "../" + p : p);

  function buildSidebar() {
    const sb = document.getElementById("sidebar");
    if (!sb) return;
    let html = `
      <div class="brand">
        <a href="${homeHref}">
          <span class="logo"><span class="dot">λ</span> Manual de RL</span>
        </a>
        <span class="tagline">Aprendizaje por Refuerzo, de la intuición a las matemáticas</span>
      </div>
      <div class="sidebar-tools">
        <button class="theme-btn" data-theme-btn>🌙 <span>Oscuro</span></button>
      </div>
      <nav class="nav">
        <a href="${homeHref}"><span class="num">★</span> Portada e índice</a>`;
    PARTS.forEach((p) => {
      html += `<div class="nav-part">${p.title}</div>`;
      CH.filter((c) => c.part === p.id).forEach((c) => {
        html += `<a href="${chHref(c.slug)}"><span class="num">${c.n}</span> ${c.title}</a>`;
      });
    });
    html += `</nav>`;
    sb.innerHTML = html;
  }

  function buildChapterNav() {
    const holder = document.getElementById("chapter-nav");
    if (!holder) return;
    const cur = parseInt(document.body.dataset.chapter || "0", 10);
    const prev = CH.find((c) => c.n === cur - 1);
    const next = CH.find((c) => c.n === cur + 1);
    let html = "";
    if (prev) html += `<a class="prev" href="${chHref(prev.slug)}"><div class="cn-label">← Anterior</div><div class="cn-title">${prev.n}. ${prev.title}</div></a>`;
    else html += `<a class="prev" href="${homeHref}"><div class="cn-label">←</div><div class="cn-title">Portada e índice</div></a>`;
    if (next) html += `<a class="next" href="${chHref(next.slug)}"><div class="cn-label">Siguiente →</div><div class="cn-title">${next.n}. ${next.title}</div></a>`;
    holder.innerHTML = html;
  }

  function buildIndexTOC() {
    const holder = document.getElementById("index-toc");
    if (!holder) return;
    let html = "";
    PARTS.forEach((p) => {
      html += `<div class="toc-part"><div class="toc-part-title">${p.title}</div><div class="toc-grid">`;
      CH.filter((c) => c.part === p.id).forEach((c) => {
        html += `<a class="toc-card" href="${chHref(c.slug)}">
          <span class="n">${c.n}</span>
          <span><span class="tc-title">${c.title}</span><span class="tc-desc">${c.desc}</span></span>
        </a>`;
      });
      html += `</div></div>`;
    });
    holder.innerHTML = html;
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildSidebar();
    buildChapterNav();
    buildIndexTOC();
  });
})();
