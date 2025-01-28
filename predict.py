import joblib
import pandas as pd
import validators

# Load the trained model
model_path = "E:\\Phishing Project\\phishing_detection_model.pkl"
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please check the file path.")
    exit()

# Function to preprocess a single URL (ensure consistency with model training)
def preprocess_input(url):
    if not validators.url(url):
        print(f"Error: The URL '{url}' is not valid.")
        return None
    
    # Extract features
    url_length = len(url)  # Length of the URL
    num_special_chars = sum(url.count(char) for char in '!#$%&*+/:=?@_~')  # Count of special characters
    https = int('https' in url)  # 1 if URL uses HTTPS, otherwise 0

    # Return as a DataFrame (consistent with training format)
    return pd.DataFrame([[url_length, num_special_chars, https]], 
                        columns=['url_length', 'num_special_chars', 'https'])

# Function to predict phishing or legitimate
def predict_url(url):
    # Preprocess the input URL
    input_data = preprocess_input(url)
    if input_data is None:
        return  # Invalid URL, skip prediction

    # Predict using the loaded model
    prediction = model.predict(input_data)

    # Interpret the result
    if prediction[0] == 1:
        print(f"The URL '{url}' is predicted to be PHISHING.")
    else:
        print(f"The URL '{url}' is predicted to be LEGITIMATE.")

# Main function
if __name__ == "__main__":
    # Example URLs to test
    test_urls = [
        "https://example-phishing-site.com",
        "http://legitimate-website.com",
        "https://malicious-site.net/login",
        "invalid-url"
    ]

    # Predict for each URL
    for url in test_urls:
        print(f"\nEvaluating URL: {url}")
        predict_url(url)
