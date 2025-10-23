import pandas as pd
import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_image_sequentially(image_link, save_path):
    """Downloads a single image and saves it."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_link, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as ex:
        print(f'\nWarning: Failed to download {image_link}\n{ex}')
        return False

# --- Main script ---
print("Starting robust, sequential image download process...")

# Load data
try:
    train_df = pd.read_csv('C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv')
except FileNotFoundError:
    print("train.csv not found. Exiting.")
    exit()

# Define a small sample size
SAMPLE_SIZE = 1000
image_links_sample = train_df['image_link'].head(SAMPLE_SIZE)

# Define download folder
DOWNLOAD_FOLDER = 'C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/images'
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

print(f"Downloading {SAMPLE_SIZE} images sequentially to {DOWNLOAD_FOLDER}...")

success_count = 0
for link in tqdm(image_links_sample, desc="Downloading images", unit="image"):
    # Use a try-except block to handle potential errors in file path creation
    try:
        filename = Path(str(link)).name
        save_path = os.path.join(DOWNLOAD_FOLDER, filename)
    except Exception as e:
        print(f"\nError creating path for link {link}: {e}")
        continue
    
    if os.path.exists(save_path):
        success_count += 1
        continue

    if download_image_sequentially(link, save_path):
        success_count += 1

# Final verification
print(f"\nDownload process complete.")
print(f"Successfully downloaded or found {success_count} out of {SAMPLE_SIZE} images.")
