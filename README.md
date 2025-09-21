🐾 Cat vs. Dog Image Classifier

A simple yet powerful web application that classifies images as either a cat or a dog. This project demonstrates a common machine learning technique called transfer learning, where a pre-trained deep learning model (VGG16) is used for feature extraction, and a classical machine learning algorithm (Support Vector Machine) is used for the final classification.

The project is presented as an interactive web app built with Streamlit.

A placeholder for your app's screenshot. You can easily create one and update this link.
✨ Features

    Interactive UI: Clean and simple web interface powered by Streamlit for easy image uploads.

    Accurate Predictions: Leverages the power of the VGG16 model, pre-trained on the massive ImageNet dataset, to extract robust image features.

    Fast Classification: Uses a lightweight and efficient Support Vector Machine (SVM) for the final prediction step.

    Confidence Score: Provides feedback on whether the image is clearly a cat/dog or if it's ambiguous.

    Modular Code: The project is split into a model training script (create_model.py) and a web application script (main.py).

🛠️ How It Works

The core of this project is a two-step classification process:

    Feature Extraction (Deep Learning): Instead of training a complex Convolutional Neural Network (CNN) from scratch, we use the VGG16 model, which has already learned to recognize thousands of features from the ImageNet dataset. We load the model without its final classification layer and use the output of its first fully-connected layer (fc1) as a high-quality feature vector (a list of numbers) that represents the input image.

    Classification (Machine Learning): This numerical feature vector is then fed into a Support Vector Machine (SVM) with a linear kernel. The SVM's job is to find the optimal hyperplane that separates the feature vectors of cats from those of dogs. This approach is much faster to train than a full deep learning model and often yields excellent results.

The entire process is wrapped in a Streamlit application, allowing users to interact with the trained model directly from their browser.
📂 Project Structure

.
├── 📄 main.py               # The main Streamlit web application script
├── 📄 create_model.py         # The script to train the SVM model
├── 📄 requirements.txt        # Python dependencies for the project
├── 📦 svm_model.joblib        # The pre-trained SVM classifier
└── 📄 README.md               # This file

🚀 Getting Started

To run this project on your local machine, please follow the steps below.
1. Prerequisites

    Python 3.8 or higher

    pip (Python package installer)

2. Installation & Setup

    Clone the repository:

    git clone [https://github.com/AniketAish/PRODIGY_ML_03.git](https://github.com/AniketAish/PRODIGY_ML_03.git)
    cd your-repo-name

    Create and activate a virtual environment (highly recommended):

        On macOS/Linux:

        python3 -m venv venv
        source venv/bin/activate

        On Windows:

        python -m venv venv
        .\venv\Scripts\activate

    Install the required packages:
    The project requires specific versions of some libraries, especially TensorFlow. Install them using the provided requirements.txt file.

    pip install -r requirements.txt

    This may take a few minutes as it needs to download TensorFlow.

3. Running the Web Application

Once the installation is complete, you can run the Streamlit app:

streamlit run main.py

Your web browser should automatically open a new tab with the application running. You can now upload a JPG, JPEG, or PNG image to get a prediction!
🧠 Training Your Own Model (Optional)

This repository includes the script (create_model.py) used to train the svm_model.joblib file. If you wish to retrain the model on your own data, follow these steps:

    Download the Dataset:
    This model was trained on a subset of the Kaggle "Dogs vs. Cats" dataset. You can download it from here.

    Organize Your Data:
    Create a dataset/train/ directory in the project's root folder and organize your images like this:

    dataset/
    └── train/
        ├── cats/
        │   ├── cat.0.jpg
        │   ├── cat.1.jpg
        │   └── ...
        └── dogs/
            ├── dog.0.jpg
            ├── dog.1.jpg
            └── ...

    Run the Training Script:
    You can adjust the NUM_SAMPLES variable inside create_model.py to control how many images from each class are used for training. A higher number leads to better accuracy but requires more time and memory.

    Execute the script from your terminal:

    python create_model.py

    This will process the images, extract features, train a new SVM, and save it as svm_model.joblib, overwriting the old one.

📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.
