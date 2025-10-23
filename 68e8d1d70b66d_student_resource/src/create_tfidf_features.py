
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

def main():
    # Load the training data
    train_df = pd.read_csv("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv")

    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

    # Fit and transform the catalog_content
    tfidf_features = tfidf_vectorizer.fit_transform(train_df['catalog_content'].fillna(''))

    # Save the TF-IDF features
    save_npz("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/tfidf_features.npz", tfidf_features)

    print("TF-IDF features created and saved to dataset/tfidf_features.npz")

if __name__ == '__main__':
    main()
