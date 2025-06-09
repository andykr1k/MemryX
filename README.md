# MemryX

## Smile Counter AI

😊 Smile Counter AI (v2 - Custom Model)Smile Counter AI is a web-based application that uses your webcam to detect faces and count unique smiles in real-time. This version uses a high-performance MediaPipe detector and a custom-trained Convolutional Neural Network (CNN) for smile classification.✨ FeaturesHigh-Performance Face Detection: Uses Google's MediaPipe for fast and accurate face detection.Custom-Trained AI Model: Employs a Keras/TensorFlow CNN trained on thousands of images to determine if a person is smiling.Unique Smile Counting: Uses face encodings to ensure each person's smile is only counted once.Interactive Interface: A clean, simple web UI to view the live camera feed, see the smile count, and adjust settings.Trainable & Extensible: Includes a script to train your own model on any binary image classification task.🚀 ArchitectureThe application is now a complete AI pipeline:Model Training (train_model.py):A script to train a CNN on a dataset of smiling and non-smiling faces.It preprocesses the data, builds a Keras model, trains it, and saves the final smile_cnn_model.h5 file.Backend (Backend.py):Loads the pre-trained smile_cnn_model.h5.Provides an API endpoint that accepts an image.Uses MediaPipe to detect the location of faces in the image.For each face, it crops the region, passes it to the custom CNN for smile/no-smile classification.It also uses the face_recognition library to generate a unique face encoding for preventing duplicate smile counts.Returns all this data to the frontend.Frontend (Frontend.py):Captures video from the user's webcam via the browser.Sends frames to the backend and displays the returned results.Manages the session state to count unique smiles.🛠️ Installation & SetupFollow these steps to get the application running on your local machine.PrerequisitesPython 3.8+pip (Python package installer)cmake (Required for dlib installation).Step-by-Step GuideGet the Project Files:Place Frontend.py, Backend.py, run.py, train_model.py, and requirements.txt in the same project directory.Set up the Dataset:Create a folder named dataset in your project directory.Download a suitable dataset. We recommend the Smiling Faces Dataset on Kaggle.After downloading and unzipping, make sure the images are organized into two subfolders inside dataset:/dataset
  /smiling
      /image1.jpg, image2.jpg, ...
  /not_smiling
      /image1.jpg, image2.jpg, ...

## Create a Virtual Environment & Install Dependencies:Open your terminal in the project directory and run:

### Create and activate virtual environment
python -m venv smile_env
source smile_env/bin/activate


### Install all dependencies
pip install -r requirements.txt

Open the URL provided by Streamlit (e.g., http://localhost:8501) in your browser, click "START", and enjoy your custom-built smile detector!💻 Technologies UsedBackend: FastAPI, Uvicorn, TensorFlow, MediaPipe, face_recognition, OpenCVFrontend: Streamlit, streamlit-webrtc, RequestsLanguage: Python