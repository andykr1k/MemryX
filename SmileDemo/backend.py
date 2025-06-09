# Backend.py

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import base64
from PIL import Image
import io
import os

# --- Initialize Models ---

# Initialize FastAPI app
app = FastAPI(
    title="Smile Detection API v2",
    description="An API that uses MediaPipe for face detection and a custom CNN for smile classification.",
    version="2.0.0",
)

# Load the custom-trained smile detection model
MODEL_PATH = 'trained_models/smile_cnn_model.h5'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
smile_model = load_model(MODEL_PATH)
IMAGE_SIZE = (64, 64) # Must match the size used during training

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# --- Pydantic Models for API ---
class Face(BaseModel):
    box: list[int]
    encoding: list[float]
    is_smiling: bool
    
class DetectionResponse(BaseModel):
    faces: list[Face]

@app.post("/detect", response_model=DetectionResponse)
async def detect_smiles(file: UploadFile = File(...)):
    """
    Endpoint to detect faces and smiles in an uploaded image.
    Uses MediaPipe for detection and a custom CNN for classification.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = rgb_img.shape

    detected_faces = []

    # --- Face Detection with MediaPipe ---
    results = face_detector.process(rgb_img)

    if results.detections:
        for detection in results.detections:
            # Extract bounding box
            bboxC = detection.location_data.relative_bounding_box
            box = [
                int(bboxC.xmin * img_w),
                int(bboxC.ymin * img_h),
                int(bboxC.width * img_w),
                int(bboxC.height * img_h)
            ]
            # Ensure box coordinates are within image bounds
            left, top, width, height = box
            right, bottom = left + width, top + height
            left, top = max(0, left), max(0, top)
            right, bottom = min(img_w, right), min(img_h, bottom)
            
            # --- Smile Prediction with Custom CNN ---
            # Crop face from the image
            face_img = rgb_img[top:bottom, left:right]
            if face_img.size == 0:
                continue

            # Preprocess the face for the model
            face_resized = cv2.resize(face_img, IMAGE_SIZE)
            face_array = np.asarray(face_resized) / 255.0
            face_expanded = np.expand_dims(face_array, axis=0)
            
            # Predict smile
            smile_prediction = smile_model.predict(face_expanded)[0][0]
            is_smiling = smile_prediction > 0.5 # Threshold for classification

            # --- Face Encoding for Duplicate Checking ---
            # Use face_recognition to get the encoding for this specific face
            # The box format for face_recognition is (top, right, bottom, left)
            fr_box = [(top, right, bottom, left)]
            encodings = face_recognition.face_encodings(rgb_img, known_face_locations=fr_box)
            
            if encodings:
                encoding = encodings[0].tolist()
                api_box = [left, top, right, bottom]
                detected_faces.append(Face(box=api_box, encoding=encoding, is_smiling=is_smiling))

    return DetectionResponse(faces=detected_faces)

