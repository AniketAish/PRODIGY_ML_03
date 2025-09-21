import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Cat vs. Dog Classifier", page_icon="🐾", layout="centered")

# --- CONSTANTS ---
MODEL_PATH = "svm_model.joblib"
IMAGE_SIZE = (224, 224)
# Confidence threshold: if the prediction score is between -T and T, classify as "None"
CONFIDENCE_THRESHOLD = 0.6

# --- MODEL LOADING ---

@st.cache_resource
def load_feature_extractor():
    """Loads the VGG16 model pre-trained on ImageNet, configured for feature extraction."""
    base_model = vgg16.VGG16(weights='imagenet')
    # We use the output of the 'fc1' layer as our feature vector
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    return feature_extractor

@st.cache_resource
def load_svm_classifier():
    """Loads the pre-trained SVM classifier from disk."""
    try:
        svm_model = joblib.load(MODEL_PATH)
        return svm_model
    except FileNotFoundError:
        return None

# Load the models
feature_extractor = load_feature_extractor()
svm_model = load_svm_classifier()

# --- HELPER FUNCTIONS ---

def extract_features(image_array):
    """
    Preprocesses an image and extracts features using the VGG16 model.
    """
    # Resize image to the target size
    resized_image = cv2.resize(image_array, IMAGE_SIZE)
    
    # Convert image to a 4D tensor and preprocess it for VGG16
    reshaped_image = resized_image.reshape(1, *IMAGE_SIZE, 3)
    preprocessed_image = vgg16.preprocess_input(reshaped_image)
    
    # Extract features
    features = feature_extractor.predict(preprocessed_image, verbose=0)
    return features.flatten()


# --- STREAMLIT UI ---

st.title("Is it a Cat or a Dog? 🐱 vs 🐶")

st.markdown("""
Upload an image and our AI will determine if it's a cat, a dog, or something else entirely!

This app uses a **Support Vector Machine (SVM)** classifier built on top of features extracted from a pre-trained **VGG16** deep learning model.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if svm_model is None:
    st.error("Model file not found! Please download `svm_model.joblib` and place it in the same folder as `app.py`.")
elif uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Your Uploaded Image', use_column_width=True)

    # Convert the PIL image to a NumPy array for processing
    image_array = img_to_array(image)

    # Show a spinner while processing
    with st.spinner('Analyzing the image... 🤔'):
        # 1. Extract features from the image
        image_features = extract_features(image_array)

        # 2. Get the prediction score from the SVM
        prediction_score = svm_model.decision_function([image_features])[0]

        # 3. Classify based on the score and the threshold
        if prediction_score < -CONFIDENCE_THRESHOLD:
            prediction = "It's a Cat! 🐈"
            st.balloons()
        elif prediction_score > CONFIDENCE_THRESHOLD:
            prediction = "It's a Dog! 🐕"
            st.balloons()
        else:
            prediction = "Hmm... this doesn't look like a cat or a dog. 🧐"

    st.success(f"**Result:** {prediction}")
    st.write(f"Confidence Score: `{prediction_score:.2f}` (Negative = Cat, Positive = Dog)")