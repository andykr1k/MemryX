import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import mediapipe as mp
import hashlib

app = FastAPI()

MODEL_PATH = 'smile_cnn_model.h5'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please train the model first.")
smile_model = load_model(MODEL_PATH, compile=False)
IMAGE_SIZE = (64, 64)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

class Face(BaseModel):
    box: list[int]
    encoding: list[float]
    is_smiling: bool

class DetectionResponse(BaseModel):
    faces: list[Face]

@app.post("/detect", response_model=DetectionResponse)
async def detect_smiles(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = rgb_img.shape
    results = face_detector.process(rgb_img)

    faces = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            box = [
                int(bboxC.xmin * img_w),
                int(bboxC.ymin * img_h),
                int(bboxC.width * img_w),
                int(bboxC.height * img_h)
            ]
            left, top, width, height = box
            right, bottom = left + width, top + height
            face_img = rgb_img[max(0, top):min(img_h, bottom), max(0, left):min(img_w, right)]

            if face_img.size == 0:
                continue

            resized = cv2.resize(face_img, IMAGE_SIZE)
            norm = resized / 255.0
            pred = smile_model.predict(np.expand_dims(norm, 0))[0][0]
            is_smiling = pred > 0.5

            # Fake "encoding" as a hash of pixel values
            encoding = [float(b) for b in hashlib.sha256(norm.tobytes()).digest()[:32]]

            faces.append(Face(
                box=[left, top, right, bottom],
                encoding=encoding,
                is_smiling=is_smiling
            ))

    return DetectionResponse(faces=faces)
