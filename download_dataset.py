import os
import urllib.request
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle

# URLs for the Sign Language MNIST dataset
TRAIN_URL = "https://storage.googleapis.com/kaggle-data-sets/110/230/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240506%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240506T092131Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8ab27efef764e5ad4f7f7f9be06e88caa73f8daba0879d5dbac7c40cca56f61c5d0a4d38cccc17baa9a7b0fd3c3a1e0baef78d9beeeffd32e618e0fd7a28f9e30c15d7ff17e20a42fef02f784ed1e8c0e8bb3fef56af31ca49c1b3bd17f42fcb20e17fb8c99bcce5ab3aaea40d1f67e7f3b38ed59be87ebbd0a3c1fb5a6c4f9a0511bfc15dad11c7adacb9f2e19f17c55fd69d877c9dec7651b1be4da9c74f1c43b45a1e62d3e0c49e6d8cd1764fe8aeb43a79b9eda8d36b62a59f0cab0cee5e1b75e38d3be9de834c7c7be8cdea68fc9dee7dce8686bf7b34878ee6fc05664cb86d8df0723fdff775babb74ddb2f31347b9da8b59fa7d7a2e92"

# Local file paths
ZIP_PATH = "sign_language_mnist.zip"
EXTRACTED_DIR = "sign_language_mnist"
MODEL_PATH = "sign_language_model.pkl"
SCALER_PATH = "scaler.pkl"

def download_dataset():
    # Create dataset directory if it doesn't exist
    if not os.path.exists(EXTRACTED_DIR):
        os.makedirs(EXTRACTED_DIR)
    
    # Download the dataset
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading dataset to {ZIP_PATH}...")
        urllib.request.urlretrieve(TRAIN_URL, ZIP_PATH)
        print("Download complete.")
    
    # Extract the dataset
    import zipfile
    if not os.path.exists(os.path.join(EXTRACTED_DIR, "sign_mnist_train.csv")):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_DIR)
        print("Extraction complete.")

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    
    # Load the CSV files
    train_data = pd.read_csv(os.path.join(EXTRACTED_DIR, "sign_mnist_train.csv"))
    test_data = pd.read_csv(os.path.join(EXTRACTED_DIR, "sign_mnist_test.csv"))
    
    # Combine train and test data
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Extract labels and features
    y = data["label"].values
    X = data.drop("label", axis=1).values
    
    # Reshape images to 28x28
    X = X.reshape(-1, 28, 28).astype(np.float32)
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data loaded. Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def save_as_images():
    print("Saving dataset as images...")
    
    # Load the CSV files
    train_data = pd.read_csv(os.path.join(EXTRACTED_DIR, "sign_mnist_train.csv"))
    test_data = pd.read_csv(os.path.join(EXTRACTED_DIR, "sign_mnist_test.csv"))
    
    # Combine train and test data
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Create a directory for each label
    for label in range(26):  # 0-25 for A-Z (excluding J and Z which require motion)
        label_dir = os.path.join("dataset", chr(65 + label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
    
    # Extract label samples and save as images
    for i, row in data.iterrows():
        label = row["label"]
        # Skip if we've already saved enough images for this label
        label_dir = os.path.join("dataset", chr(65 + label))
        if len(os.listdir(label_dir)) >= 1000:  # Limit to 1000 images per label
            continue
            
        # Get the image data and reshape to 28x28
        img_data = row.drop("label").values.reshape(28, 28).astype(np.uint8)
        
        # Save the image
        img_path = os.path.join(label_dir, f"img_{i}.png")
        cv2.imwrite(img_path, img_data)
        
        # Print progress
        if i % 1000 == 0:
            print(f"Processed {i} images")
    
    print("All images saved.")

if __name__ == "__main__":
    # Make sure the dataset directory exists
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    
    # Download and extract the dataset
    download_dataset()
    
    # Save as images for consistency with existing app
    save_as_images()
    
    print("Dataset preparation complete!") 