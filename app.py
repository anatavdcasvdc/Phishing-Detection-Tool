from flask import Flask, request, jsonify
import joblib
import re

# Load the trained model
model = joblib.load("E:/Phishing Project/phishing_detection_model.pkl")

# Flask app
app = Flask(__name__)

def extract_features(url):
    """
    Extract features from the URL for prediction.
    """
    url_length = len(url)
    num_special_chars = sum(url.count(char) for char in '!#$%&*+/:=?@_~')
    https = int('https' in url)
    domain = re.search(r'(https?:\/\/)?([^\/]+)', url).group(2) if url else None
    return [[url_length, num_special_chars, https]]  # Return as a list of features

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests to make phishing predictions.
    """
    data = request.json  # Expecting JSON payload
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # Extract features and make a prediction
    features = extract_features(url)
    prediction = model.predict(features)
    result = "Phishing" if prediction[0] == 1 else "Legitimate"
    return jsonify({'url': url, 'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
