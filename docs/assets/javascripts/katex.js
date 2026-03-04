function renderKatexMath(root) {
  if (typeof renderMathInElement !== "function" || !root) {
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
    throwOnError: false,
    strict: "ignore",
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(({ body }) => {
    renderKatexMath(body);
  });
} else {
  document.addEventListener("DOMContentLoaded", () => {
    renderKatexMath(document.body);
  });
}
