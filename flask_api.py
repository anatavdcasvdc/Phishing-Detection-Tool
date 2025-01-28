from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
model = joblib.load("phishing_detection_model.pkl")

# Function to extract features from a URL
def extract_features_from_url(url):
    # Extract features in the same way as in the training script
    return pd.DataFrame([{
        'url_length': len(url),
        'num_special_chars': sum(1 for char in url if char in "!@#$%^&*()"),
        'https': int("https" in url.lower()),
        'num_subdomains': url.count('.') - 1,
        'has_suspicious_words': int(any(word in url.lower() for word in ['login', 'secure', 'account', 'verify', 'update', 'free', 'click'])),
        'domain_length': len(url.split('/')[2]) if '//' in url else len(url),
    }])

# Root route to test if the API is running
@app.route('/')
def home():
    return "Welcome to the Phishing Detection API! Use /predict-url to get predictions."

# Prediction route
@app.route('/predict-url', methods=['POST'])
def predict_url():
    try:
        # Get the URL from the POST request
        data = request.json
        url = data.get("url")
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Extract features for the URL
        features = extract_features_from_url(url)

        # Predict using the trained model
        prediction = model.predict(features)[0]
        result = "Phishing" if prediction == 1 else "Legitimate"

        # Return the prediction result
        return jsonify({"url": url, "prediction": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
