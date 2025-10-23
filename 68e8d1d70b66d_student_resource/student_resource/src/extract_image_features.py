import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

print("Starting image feature extraction process...")

# --- Configuration ---
# Use CPU for broader compatibility, as not everyone has a CUDA-enabled GPU
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

SAMPLE_SIZE = 1000
IMAGE_FOLDER = 'C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/images'
TRAIN_CSV_PATH = 'C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv'
OUTPUT_FEATURES_PATH = 'C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/image_features_1000.npz'

# --- Model and Transforms ---
print("Loading pre-trained ResNet50 model...")
# 1. Load pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
model.eval() # Set to evaluation mode

# 2. Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Hook to extract features from the 'avgpool' layer
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_features('avgpool'))

# --- Feature Extraction ---
# 4. Load data and get image paths
try:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
except FileNotFoundError:
    print(f"Error: {TRAIN_CSV_PATH} not found. Exiting.")
    exit()

sample_df = train_df.head(SAMPLE_SIZE)
image_features_dict = {}

print(f"Extracting features from {SAMPLE_SIZE} images...")

for index, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0], desc="Processing images"):
    try:
        image_link = row['image_link']
        sample_id = row['sample_id']
        image_filename = Path(str(image_link)).name
        image_path = os.path.join(IMAGE_FOLDER, image_filename)

        if not os.path.exists(image_path):
            # This check is important because some downloads may have failed
            # print(f"Warning: Image not found at {image_path}. Skipping.")
            continue

        # 5. Open, preprocess, and get features for one image
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _ = model(img_tensor)
            # The hook populates the 'features' dictionary
            image_features_dict[str(sample_id)] = features['avgpool'].cpu().numpy().flatten()

    except Exception as e:
        print(f"Error processing image for sample_id {row.get('sample_id', 'N/A')}: {e}")


# --- Save Features ---
# 6. Save the extracted features
if image_features_dict:
    np.savez_compressed(OUTPUT_FEATURES_PATH, **image_features_dict)
    print(f"\nFeature extraction complete.")
    print(f"Extracted features for {len(image_features_dict)} images.")
    print(f"Features saved to {OUTPUT_FEATURES_PATH}")
else:
    print("\nNo features were extracted. Please check image paths and download status.")
