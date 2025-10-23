import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import sys

# Open a log file
log_file = open("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/src/embedding_generation.log", "w")
sys.stdout = log_file
sys.stderr = log_file

def main():
    print("Starting the script...")

    # Load the training data
    print("Loading the training data...")
    start_time = time.time()
    train_df = pd.read_csv("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv")
    end_time = time.time()
    print(f"Training data loaded in {end_time - start_time:.2f} seconds.")

    # Load a pre-trained Sentence Transformer model
    print("Loading the Sentence Transformer model...")
    start_time = time.time()
    model = SentenceTransformer('all-mpnet-base-v2')
    end_time = time.time()
    print(f"Sentence Transformer model loaded in {end_time - start_time:.2f} seconds.")

    # Encode the catalog_content to get sentence embeddings
    print("Encoding the catalog_content to get sentence embeddings...")
    start_time = time.time()
    sentences = train_df['catalog_content'].fillna('').tolist()
    total_sentences = len(sentences)
    batch_size = 1000
    sentence_embeddings = []

    for i in range(0, total_sentences, batch_size):
        batch = sentences[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        sentence_embeddings.extend(batch_embeddings)
        print(f"Processed {i+len(batch)} out of {total_sentences} sentences...")

    end_time = time.time()
    print(f"Encoding finished in {end_time - start_time:.2f} seconds.")

    # Save the embeddings to a .npz file
    print("Saving the embeddings to a .npz file...")
    start_time = time.time()
    np.savez_compressed(
        "C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/sentence_embeddings_full.npz",
        features=np.array(sentence_embeddings)
    )
    end_time = time.time()
    print(f"Embeddings saved in {end_time - start_time:.2f} seconds.")

    print("Script finished successfully.")

if __name__ == '__main__':
    main()

# Close the log file
log_file.close()