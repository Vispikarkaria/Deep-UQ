document.addEventListener("DOMContentLoaded", () => {
  const items = document.querySelectorAll(".reveal");
  items.forEach((el, idx) => {
    setTimeout(() => {
      el.classList.add("visible");
    }, 60 * idx);
  });
});
