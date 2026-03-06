function typesetMathJax() {
  if (!window.MathJax || !MathJax.typesetPromise || !MathJax.startup) {
    return;
  }

  MathJax.startup.promise.then(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}

if (window.document$ && typeof window.document$.subscribe === "function") {
  window.document$.subscribe(() => {
    typesetMathJax();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  typesetMathJax();
});

window.addEventListener("load", () => {
  typesetMathJax();
});
