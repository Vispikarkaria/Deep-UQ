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

function renderArithmatexNodes(root) {
  if (typeof katex !== "object" || !root) {
    return;
  }

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

function renderKatexMath(root) {
  if (!root) {
    return;
  }

  renderArithmatexNodes(root);

  if (typeof renderMathInElement !== "function") {
    return;
  }

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

function resolveRoot(payload) {
  if (payload && payload.body && payload.body.nodeType === 1) {
    return payload.body;
  }
  return document.body;
}

if (typeof document$ !== "undefined") {
  document$.subscribe((payload) => {
    renderKatexMath(resolveRoot(payload));
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderKatexMath(document.body);
});
