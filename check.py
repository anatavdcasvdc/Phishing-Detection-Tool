from sklearn.utils import resample
import pandas as pd

# Load the dataset
data = pd.read_csv("processed_dataset.csv")

# Separate phishing and legitimate samples
phishing = data[data['label'] == 1]
legitimate = data[data['label'] == 0]

# Downsample legitimate URLs to match the number of phishing URLs
legitimate_downsampled = resample(legitimate, 
                                  replace=False,  # Don't resample with replacement
                                  n_samples=len(phishing),  # Match phishing count
                                  random_state=42)

# Combine phishing and downsampled legitimate data
balanced_data = pd.concat([phishing, legitimate_downsampled])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset
balanced_data.to_csv("balanced_dataset.csv", index=False)
print("Balanced dataset saved as balanced_dataset.csv!")
