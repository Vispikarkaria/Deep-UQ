window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

const MATHJAX_FALLBACKS = [
  "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
  "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js",
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js",
];

let mathJaxRenderScheduled = false;
let mathJaxLoadPromise = null;

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${url}"]`);
    if (existing) {
      if (existing.dataset.loaded === "1") {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error(`Failed to load ${url}`)), {
        once: true,
      });
      return;
    }

    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.onload = () => {
      script.dataset.loaded = "1";
      resolve();
    };
    script.onerror = () => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(script);
  });
}

async function ensureMathJaxRuntime() {
  if (window.MathJax && MathJax.typesetPromise && MathJax.startup) {
    return true;
  }

  if (!mathJaxLoadPromise) {
    mathJaxLoadPromise = (async () => {
      for (const url of MATHJAX_FALLBACKS) {
        try {
          await loadScript(url);
          if (window.MathJax && MathJax.typesetPromise && MathJax.startup) {
            return true;
          }
        } catch (_err) {
          // Try the next CDN.
        }
      }
      mathJaxLoadPromise = null;
      return false;
    })();
  }

  return await mathJaxLoadPromise;
}

async function typesetArithmatex() {
  const available = await ensureMathJaxRuntime();
  if (!available) {
    return false;
  }

  await MathJax.startup.promise;

  const nodes = document.querySelectorAll(".arithmatex");
  if (!nodes.length) {
    return true;
  }

  MathJax.startup.output.clearCache();
  MathJax.typesetClear(nodes);
  MathJax.texReset();
  await MathJax.typesetPromise(nodes);
  return true;
}

function scheduleMathJaxRender(attempt = 0) {
  if (mathJaxRenderScheduled) {
    return;
  }

  mathJaxRenderScheduled = true;

  requestAnimationFrame(async () => {
    mathJaxRenderScheduled = false;

    const rendered = await typesetArithmatex();
    if (!rendered && attempt < 40) {
      setTimeout(() => scheduleMathJaxRender(attempt + 1), 100);
    }
  });
}

function subscribeMaterialNavigation(attempt = 0) {
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => {
      scheduleMathJaxRender();
    });
    return;
  }

  if (attempt < 40) {
    setTimeout(() => subscribeMaterialNavigation(attempt + 1), 100);
  }
}

subscribeMaterialNavigation();

document.addEventListener("DOMContentLoaded", () => {
  scheduleMathJaxRender();
});

window.addEventListener("load", () => {
  scheduleMathJaxRender();
});
