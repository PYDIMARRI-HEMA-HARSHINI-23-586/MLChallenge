import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import re

# Define the SMAPE metric
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator[denominator == 0] = 1
    return np.mean(numerator / denominator) * 100

# Function to preprocess text
def preprocess_text(text):
    # Simple preprocessing: lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 1. Load Data
try:
    train_df = pd.read_csv('C:\\Users\\HEMA HARSHINI\\Desktop\\ml\\MLChallenge\\68e8d1d70b66d_student_resource\\student_resource\\dataset\\train.csv')
except FileNotFoundError:
    print("train.csv not found. Exiting.")
    exit()

# Apply preprocessing to the text column
train_df['cleaned_content'] = train_df['catalog_content'].apply(preprocess_text)


# 2. Split Data
X = train_df['cleaned_content']
y = train_df['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create a pipeline
baseline_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('ridge', Ridge(alpha=1.0))
])

# 4. Train the pipeline
print("Training the baseline model...")
baseline_pipeline.fit(X_train, y_train)
print("Training complete.")

# 5. Make predictions on the validation set
print("Making predictions on the validation set...")
y_pred = baseline_pipeline.predict(X_val)

# Ensure predictions are non-negative
y_pred[y_pred < 0] = 0

# 6. Evaluate the model
validation_smape = smape(y_val.to_numpy(), y_pred)
print(f"\nValidation SMAPE Score: {validation_smape:.4f}%")

print("\nBaseline model training and evaluation complete.")
