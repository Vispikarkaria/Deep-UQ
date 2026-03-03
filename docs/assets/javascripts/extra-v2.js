function applyRevealAnimation() {
  document.documentElement.classList.add("js-reveal");
  const items = document.querySelectorAll(".reveal");
  items.forEach((el, idx) => {
    el.style.transitionDelay = `${60 * idx}ms`;
    el.classList.add("visible");
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    applyRevealAnimation();
  });
} else {
  document.addEventListener("DOMContentLoaded", () => {
    applyRevealAnimation();
  });
}
