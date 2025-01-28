import pandas as pd
import validators
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# --- Step 1: Load and Clean Data ---
def load_and_clean_data(phishing_file, legitimate_file):
    try:
        # Load phishing data
        phishing_df = pd.read_csv(phishing_file)
        phishing_df.dropna(inplace=True)
        phishing_df.drop_duplicates(inplace=True)
        phishing_df['valid_url'] = phishing_df['url'].apply(lambda x: validators.url(x))
        phishing_df = phishing_df[phishing_df['valid_url'] == True]
        phishing_df['label'] = 1  # Phishing

        # Load legitimate data
        legitimate_df = pd.read_csv(legitimate_file)  # Now assumes 'rank' and 'url' columns exist
        if 'url' not in legitimate_df.columns:
            raise ValueError("Legitimate dataset must contain a column named 'url'.")
        legitimate_df['label'] = 0  # Legitimate

        # Combine datasets
        combined_df = pd.concat([phishing_df, legitimate_df[['url', 'label']]], ignore_index=True).drop_duplicates(subset='url', keep='first')
        combined_df.fillna('Unknown', inplace=True)

        if combined_df.empty:
            raise ValueError("Combined dataset is empty after cleaning. Check your input files.")

        print(f"Phishing URLs: {len(phishing_df)}, Legitimate URLs: {len(legitimate_df)}, Combined: {len(combined_df)}")
        return combined_df
    except Exception as e:
        print(f"Error during data loading and cleaning: {e}")
        raise

# --- Step 2: Feature Engineering ---
def extract_domain(url):
    try:
        return re.search(r'(https?:\/\/)?([^\/]+)', url).group(2)
    except Exception:
        return None

def feature_engineering(df):
    try:
        df['url_length'] = df['url'].apply(len)
        df['num_special_chars'] = df['url'].apply(lambda x: sum(x.count(char) for char in '!#$%&*+/:=?@_~'))
        df['https'] = df['url'].apply(lambda x: int('https' in x))
        df['domain'] = df['url'].apply(extract_domain)
        return df
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        raise

# --- Step 3: Prepare Data for Model ---
def prepare_data_for_model(df):
    try:
        X = df[['url_length', 'num_special_chars', 'https', 'domain']]
        y = df['label']
        
        # Train-test-validation split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # One-hot encode the 'domain' column
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X_train[['domain']])
        
        def encode_data(X):
            domain_encoded = encoder.transform(X[['domain']])
            domain_encoded_df = pd.DataFrame(domain_encoded, columns=encoder.get_feature_names_out(['domain']))
            return pd.concat([X.drop('domain', axis=1).reset_index(drop=True), domain_encoded_df.reset_index(drop=True)], axis=1)
        
        X_train_encoded = encode_data(X_train)
        X_val_encoded = encode_data(X_val)
        X_test_encoded = encode_data(X_test)
        
        return X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test, encoder
    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise

# --- Step 4: Train Model ---
def train_model(X_train, y_train):
    try:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

# --- Step 5: Evaluate Model ---
def evaluate_model(model, X_val, y_val):
    try:
        y_pred_val = model.predict(X_val)
        print("Validation Metrics:")
        print("Accuracy:", accuracy_score(y_val, y_pred_val))
        print("Precision:", precision_score(y_val, y_pred_val))
        print("Recall:", recall_score(y_val, y_pred_val))
        print("F1-score:", f1_score(y_val, y_pred_val))
        print("AUC-ROC:", roc_auc_score(y_val, y_pred_val))
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise

# --- Step 6: Save Processed Data and Model ---
def save_processed_data(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error during saving processed data: {e}")
        raise

def save_model(model, encoder, model_path, encoder_path):
    try:
        joblib.dump(model, model_path)
        joblib.dump(encoder, encoder_path)
        print(f"Model saved to {model_path}")
        print(f"Encoder saved to {encoder_path}")
    except Exception as e:
        print(f"Error during saving model or encoder: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    # File paths
    phishing_file = r"E:\Phishing Project\phishing_data.csv"  # Path for phishing data
    legitimate_file = r"E:\Phishing Project\alexa_top_1m_processed.csv"  # Updated file with headers
    processed_file = r"E:\Phishing Project\processed_dataset.csv"
    model_save_path = r"E:\Phishing Project\phishing_detection_model.pkl"
    encoder_save_path = r"E:\Phishing Project\domain_encoder.pkl"

    # Pipeline execution
    combined_data = load_and_clean_data(phishing_file, legitimate_file)
    combined_data = feature_engineering(combined_data)
    save_processed_data(combined_data, processed_file)

    # Prepare data for model
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = prepare_data_for_model(combined_data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    # Save the trained model and encoder
    save_model(model, encoder, model_save_path, encoder_save_path)
