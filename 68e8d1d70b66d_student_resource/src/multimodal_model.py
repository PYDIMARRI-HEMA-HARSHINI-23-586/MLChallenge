
import pandas as pd
import numpy as np
from scipy.sparse import hstack, load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
import joblib
import lightgbm as lgb

# Define the SMAPE metric based on the guideline
def smape(y_true, y_pred):
    return np.mean((2 * np.abs(y_pred - y_true)) / (np.abs(y_true) + np.abs(y_pred))) * 100

def main():
    # Load the pre-computed sentence embeddings
    sentence_embeddings_data = np.load("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/sentence_embeddings.npz")
    text_features = sentence_embeddings_data['features']

    # Load the image features
    image_features_data = np.load("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/image_features_1000.npz")
    image_features = image_features_data['features']

    # Load the training data
    train_df = pd.read_csv("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv")
    
    # Ensure the number of samples match
    num_samples = 1000
    train_df_sample = train_df.head(num_samples)

    # Combine text and image features
    combined_features = np.hstack([text_features[:num_samples], image_features[:num_samples]])

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        combined_features, train_df_sample['price'], test_size=0.2, random_state=42
    )

    # Define hyperparameter grid for LightGBM
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 31],
        'max_depth': [-1, 10]
    }

    best_smape = float('inf')
    best_params = {}

    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for num_leaves in param_grid['num_leaves']:
                for max_depth in param_grid['max_depth']:
                    print(f"Training with n_estimators={n_estimators}, learning_rate={learning_rate}, num_leaves={num_leaves}, max_depth={max_depth}")
                    model = lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        max_depth=max_depth,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_pred[y_pred < 0] = 0
                    smape_score = smape(y_val, y_pred)
                    print(f"SMAPE: {smape_score:.4f}")

                    if smape_score < best_smape:
                        best_smape = smape_score
                        best_params = {
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'num_leaves': num_leaves,
                            'max_depth': max_depth
                        }
    
    print(f"\nBest SMAPE found: {best_smape:.4f} with parameters: {best_params}")

if __name__ == '__main__':
    main()
