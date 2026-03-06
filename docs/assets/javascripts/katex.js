const KATEX_JS_FALLBACKS = [
  "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
  "https://unpkg.com/katex@0.16.11/dist/katex.min.js",
  "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js",
];

const KATEX_AUTORENDER_FALLBACKS = [
  "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js",
  "https://unpkg.com/katex@0.16.11/dist/contrib/auto-render.min.js",
  "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/contrib/auto-render.min.js",
];

let katexEnsurePromise = null;
let renderScheduled = false;

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

async function loadFirstSuccessful(urls) {
  for (const url of urls) {
    try {
      await loadScript(url);
      return true;
    } catch (_err) {
      // Try the next CDN fallback.
    }
  }
  return false;
}

function extractMathPayload(rawText, preferDisplay) {
  const text = (rawText || "").replace(/\u00a0/g, " ").trim();
  const wrapped = [
    { left: "\\[", right: "\\]", displayMode: true },
    { left: "\\(", right: "\\)", displayMode: false },
    { left: "$$", right: "$$", displayMode: true },
    { left: "$", right: "$", displayMode: false },
  ];

  for (const item of wrapped) {
    if (text.startsWith(item.left) && text.endsWith(item.right) && text.length > item.left.length + item.right.length) {
      return {
        math: text.slice(item.left.length, text.length - item.right.length).trim(),
        displayMode: item.displayMode,
      };
    }
  }

  return { math: text, displayMode: preferDisplay };
}

async function ensureKatexRuntime() {
  if (typeof katex === "object" && typeof renderMathInElement === "function") {
    return;
  }

  if (!katexEnsurePromise) {
    katexEnsurePromise = (async () => {
      if (typeof katex !== "object") {
        await loadFirstSuccessful(KATEX_JS_FALLBACKS);
      }
      if (typeof renderMathInElement !== "function") {
        await loadFirstSuccessful(KATEX_AUTORENDER_FALLBACKS);
      }
      if (typeof katex !== "object" || typeof renderMathInElement !== "function") {
        throw new Error("KaTeX runtime is unavailable.");
      }
    })()
      .catch((err) => {
        katexEnsurePromise = null;
        throw err;
      });
  }

  await katexEnsurePromise;
}

function renderArithmatexNodes(root) {
  root.querySelectorAll(".arithmatex").forEach((node) => {
    if (node.dataset.katexRendered === "1") {
      return;
    }

    const isDisplayNode = node.tagName.toLowerCase() === "div";
    const payload = extractMathPayload(node.textContent, isDisplayNode);
    if (!payload.math) {
      return;
    }

    katex.render(payload.math, node, {
      displayMode: payload.displayMode,
      throwOnError: false,
      strict: "ignore",
      trust: false,
    });

    node.dataset.katexRendered = "1";
  });
}

async function renderKatexMath(root) {
  if (!root) {
    return;
  }

  try {
    await ensureKatexRuntime();
  } catch (_err) {
    return;
  }

  renderArithmatexNodes(root);

  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoredClasses: ["arithmatex"],
    throwOnError: false,
    strict: "ignore",
  });
}

function scheduleKatexRender() {
  if (renderScheduled) {
    return;
  }
  renderScheduled = true;

  const run = () => {
    renderScheduled = false;
    renderKatexMath(document.body);
  };

  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(run);
  } else {
    setTimeout(run, 0);
  }
}

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    scheduleKatexRender();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  scheduleKatexRender();
});

window.addEventListener("load", () => {
  scheduleKatexRender();
});
