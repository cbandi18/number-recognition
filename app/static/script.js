document.getElementById("upload").addEventListener("change", function(event) {
    let canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");
    let file = event.target.files[0];

    if (file) {
        let img = new Image();
        img.onload = function () {
            canvas.width = 28;
            canvas.height = 28;
            ctx.drawImage(img, 0, 0, 28, 28);

            // Convert image to grayscale
            let imageData = ctx.getImageData(0, 0, 28, 28);
            let grayPixels = [];
            for (let i = 0; i < imageData.data.length; i += 4) {
                let gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
                grayPixels.push(gray / 255.0);  // Normalize to [0,1]
            }

            localStorage.setItem("imageData", JSON.stringify(grayPixels));
        };
        img.src = URL.createObjectURL(file);
    }
});

function predict() {
    let imageData = localStorage.getItem("imageData");
    if (!imageData) {
        alert("Please upload an image first!");
        return;
    }

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Predicted Digit: " + data.prediction;
    })
    .catch(error => console.error("Error:", error));
}