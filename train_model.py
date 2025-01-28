import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Function to extract features from URLs
def extract_features(df):
    # Length of the URL
    df['url_length'] = df['url'].apply(len)

    # Count of special characters in the URL
    df['num_special_chars'] = df['url'].apply(lambda x: sum(1 for char in x if char in "!@#$%^&*()"))

    # Presence of HTTPS
    df['https'] = df['url'].apply(lambda x: int("https" in x.lower()))

    # Number of subdomains
    df['num_subdomains'] = df['url'].apply(lambda x: x.count('.') - 1)

    # Check for suspicious keywords in the URL
    suspicious_keywords = ['login', 'secure', 'account', 'verify', 'update', 'free', 'click']
    df['has_suspicious_words'] = df['url'].apply(
        lambda x: int(any(word in x.lower() for word in suspicious_keywords))
    )

    # Length of the domain name
    df['domain_length'] = df['domain'].apply(len)

    return df

# Main script
if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv("processed_dataset.csv")

    # Extract features
    print("Extracting features...")
    data = extract_features(data)

    # Define feature columns and target column
    feature_columns = ['url_length', 'num_special_chars', 'https', 'num_subdomains', 'has_suspicious_words', 'domain_length']
    X = data[feature_columns]
    y = data['label']

    # Split the dataset into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("Training the model...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    print("Saving the model...")
    joblib.dump(model, "phishing_detection_model.pkl")
    print("Model saved as phishing_detection_model.pkl")

    print("Training completed successfully!")
