# Phishing-Detection-Tool
A phishing detection tool that uses machine learning to classify URLs as legitimate or phishing. This project includes a browser extension for real-time URL analysis and a Flask API backend for predictions.
Features
Browser Extension: Simple interface to check URLs instantly.
Flask API: Backend server to process and classify URLs.
Real-Time Detection: Immediate phishing classification.
Feature Engineering: Custom features like URL length, subdomain count, and suspicious keywords.
Tech Stack
Python: Flask, pandas, scikit-learn
JavaScript: Browser extension logic
HTML/CSS: User interface for the extension
How to Use
Run the Flask API:

bash
Copy
Edit
python flask_api.py
The API will start at http://127.0.0.1:5000.

Load the Browser Extension:

Open Chrome and navigate to chrome://extensions.
Enable Developer Mode.
Click Load unpacked and select the extension/ folder.
Check a URL:

Enter a URL in the extension and get real-time predictions.
How It Works
The model is trained on real phishing and legitimate URLs.
Extracted features include:
URL length
Special characters
Presence of HTTPS
Subdomains
Suspicious keywords
The Gradient Boosting Classifier predicts whether a URL is phishing or legitimate.


Future Improvements
Add support for real-time WHOIS data.
Deploy the API to a cloud server for global access.
