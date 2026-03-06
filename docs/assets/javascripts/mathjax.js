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

let mathJaxRenderScheduled = false;

async function typesetArithmatex() {
  if (!window.MathJax || !MathJax.typesetPromise || !MathJax.startup) {
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

document$.subscribe(() => {
  scheduleMathJaxRender();
});

window.addEventListener("load", () => {
  scheduleMathJaxRender();
});
