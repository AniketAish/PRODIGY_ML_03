import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

## --- CONFIGURATION ---

# Define the path to your dataset
# Make sure this path matches your folder structure
DATASET_PATH = os.path.join('dataset', 'train')
CAT_DIR = os.path.join(DATASET_PATH, 'cats')
DOG_DIR = os.path.join(DATASET_PATH, 'dogs')

# Number of images to use for training from each class.
# Training on all 25,000 images can take a very long time on a CPU.
# Start with a smaller number to test the process.
NUM_SAMPLES = 500

# Define the size for resizing images
IMAGE_SIZE = (224, 224)

## --- MODEL LOADING ---

print("Loading VGG16 feature extractor...")
base_model = vgg16.VGG16(weights='imagenet')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
print("Model loaded successfully.")

## --- FEATURE EXTRACTION FUNCTION ---

def process_and_extract_features(image_dir, num_samples):
    """
    Reads images from a directory, preprocesses them, and extracts features using VGG16.
    """
    feature_list = []
    image_files = os.listdir(image_dir)[:num_samples] # Get a slice of files

    print(f"Extracting features from {len(image_files)} images in '{image_dir}'...")

    for i, filename in enumerate(image_files):
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_samples} images...")

        image_path = os.path.join(image_dir, filename)

        try:
            # Read and resize the image
            image = cv2.imread(image_path)
            if image is None: continue # Skip if image cannot be read
            image_resized = cv2.resize(image, IMAGE_SIZE)

            # Preprocess the image for VGG16
            image_preprocessed = vgg16.preprocess_input(img_to_array(image_resized).reshape(1, *IMAGE_SIZE, 3))

            # Extract features and flatten
            features = feature_extractor.predict(image_preprocessed, verbose=0).flatten()
            feature_list.append(features)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"Feature extraction complete for '{image_dir}'.")
    return np.array(feature_list)

## --- MAIN TRAINING SCRIPT ---

if __name__ == "__main__":
    # Extract features for cats and dogs
    cat_features = process_and_extract_features(CAT_DIR, NUM_SAMPLES)
    dog_features = process_and_extract_features(DOG_DIR, NUM_SAMPLES)

    # Combine features into a single dataset
    X = np.vstack([cat_features, dog_features])

    # Create corresponding labels (0 for cat, 1 for dog)
    y = np.array([0] * len(cat_features) + [1] * len(dog_features))

    print("\nStarting SVM model training...")
    # Using verbose=True will show you the training progress
    svm_model = SVC(kernel='linear', verbose=True)
    svm_model.fit(X, y)
    print("SVM training complete.")

    # Save the trained model
    joblib.dump(svm_model, 'svm_model.joblib')
    print(f"\n✅ Success! Real model saved as 'svm_model.joblib'.")
    print("You can now run your Streamlit app.")