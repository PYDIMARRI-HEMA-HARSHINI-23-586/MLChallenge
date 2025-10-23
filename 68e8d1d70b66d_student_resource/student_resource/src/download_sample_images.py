import pandas as pd
import os
import re
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib

def download_image(image_link, savefolder):
    if(isinstance(image_link, str)):
        try:
            filename = Path(image_link).name
            image_save_path = os.path.join(savefolder, filename)
            if(not os.path.exists(image_save_path)):
                urllib.request.urlretrieve(image_link, image_save_path)
        except Exception as ex:
            print(f'Warning: Not able to download - {image_link}\n{ex}')
    return

def download_images_parallel(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    # Use a reasonable number of processes
    with multiprocessing.Pool(processes=10) as pool:
        download_func = partial(download_image, savefolder=download_folder)
        # tqdm shows a progress bar
        list(tqdm(pool.imap(download_func, image_links), total=len(image_links)))


# --- Main script ---
print("Starting image download process...")

# Load data
try:
    train_df = pd.read_csv('C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv')
except FileNotFoundError:
    print("train.csv not found. Exiting.")
    exit()

# Define a small sample size
SAMPLE_SIZE = 10
image_links_sample = train_df['image_link'].head(SAMPLE_SIZE)

# Define download folder
DOWNLOAD_FOLDER = 'C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/images'

print(f"Downloading {SAMPLE_SIZE} images to {DOWNLOAD_FOLDER}...")

# Download images
download_images_parallel(image_links_sample, DOWNLOAD_FOLDER)

# Verify download
try:
    downloaded_files = os.listdir(DOWNLOAD_FOLDER)
    print(f"\nDownload complete.")
    print(f"Successfully downloaded {len(downloaded_files)} out of {SAMPLE_SIZE} images.")
except FileNotFoundError:
    print("\nDownload folder not created. There might have been an issue with all downloads.")
