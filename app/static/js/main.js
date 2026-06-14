const form = document.querySelector("form");
const button = document.querySelector(".predict-button");

if (form && button) {
    form.addEventListener("submit", () => {
        button.innerText = "Predicting...";
        button.style.opacity = "0.85";
    });
}