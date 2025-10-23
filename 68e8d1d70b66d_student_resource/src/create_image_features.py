
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os

def main():
    # Load the training data
    train_df = pd.read_csv("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv")

    # Load the ResNet50 model pre-trained on ImageNet data
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Path to the directory where images are downloaded
    images_dir = "C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/images"

    # List to store the features
    features_list = []

    # Iterate over the first 1000 rows of the dataframe
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        image_filename = row['image_link'].split('/')[-1]
        img_path = os.path.join(images_dir, image_filename)

        if os.path.exists(img_path):
            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Extract features
            features = model.predict(img_array)
            features_list.append(features.flatten())
        else:
            # If image does not exist, append a vector of zeros
            features_list.append(np.zeros(model.output_shape[1]))

    # Convert the list of features to a numpy array
    features_array = np.array(features_list)

    # Save the features to a .npz file
    np.savez_compressed(
        "C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/image_features_1000.npz",
        features=features_array
    )

    print("Image features created and saved to dataset/image_features_1000.npz")

if __name__ == '__main__':
    main()
