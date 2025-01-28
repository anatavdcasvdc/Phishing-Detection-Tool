document.getElementById('checkBtn').addEventListener('click', async () => {
    const urlInput = document.getElementById('urlInput').value;
    const resultElement = document.getElementById('result');

    // Clear previous results
    resultElement.textContent = '';

    if (!urlInput) {
        resultElement.textContent = "Please enter a URL.";
        return;
    }

    try {
        // Send the URL to the Flask API
        const response = await fetch('http://127.0.0.1:5000/predict-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: urlInput })
        });

        if (response.ok) {
            const data = await response.json();
            resultElement.textContent = `Prediction: ${data.prediction}`;
        } else {
            resultElement.textContent = "Error: Unable to get a response from the server.";
        }
    } catch (error) {
        console.error("Error:", error);
        resultElement.textContent = "Error: Unable to connect to the server.";
    }
});
