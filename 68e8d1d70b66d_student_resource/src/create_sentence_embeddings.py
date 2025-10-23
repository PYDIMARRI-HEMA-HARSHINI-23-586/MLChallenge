
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    # Load the training data
    train_df = pd.read_csv("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/sample_test.csv")

    # Load a pre-trained Sentence Transformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Encode the catalog_content to get sentence embeddings
    sentence_embeddings = model.encode(train_df['catalog_content'].fillna('').tolist(), show_progress_bar=True)

    # Save the embeddings to a .npz file
    np.savez_compressed(
        "C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/sentence_embeddings.npz",
        features=sentence_embeddings
    )

    print("Sentence embeddings created and saved to dataset/sentence_embeddings.npz")

if __name__ == '__main__':
    main()
