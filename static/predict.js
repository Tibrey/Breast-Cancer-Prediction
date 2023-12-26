document.addEventListener("DOMContentLoaded", function () {
  const labels = document.querySelectorAll(".form_container > label");

  labels.forEach((label) => {
    const input = label.querySelector("input");
    const span = label.querySelector("span");

    input.addEventListener("focus", () => {
      label.classList.add("focus");
      span.style.top = "0";
    });

    input.addEventListener("blur", () => {
      if (!input.value.trim()) {
        label.classList.remove("focus");
        span.style.top = "50%";
      }
    });
  });
});
