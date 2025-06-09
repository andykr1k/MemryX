import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import mediapipe as mp
# from insightface.app import FaceAnalysis

# Initialize FastAPI app
app = FastAPI()

# Load smile detection model
MODEL_PATH = 'trained_models/smile_cnn_model.h5'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Please train the model first.")
smile_model = load_model(MODEL_PATH)
IMAGE_SIZE = (64, 64)

# Initialize MediaPipe and InsightFace
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
# face_analyzer = FaceAnalysis(name='buffalo_l')
# face_analyzer.prepare(ctx_id=0, det_size=(640, 640))


class Face(BaseModel):
    box: list[int]
    encoding: list[float]
    is_smiling: bool


class DetectionResponse(BaseModel):
    faces: list[Face]


@app.post("/detect", response_model=DetectionResponse)
async def detect_smiles(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = rgb_img.shape

    results = face_detector.process(rgb_img)
    detected_faces = []

    if results.detections:
        # faces = face_analyzer.get(rgb_img)

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
            left, top = max(0, left), max(0, top)
            right, bottom = min(img_w, right), min(img_h, bottom)
            face_img = rgb_img[top:bottom, left:right]
            if face_img.size == 0:
                continue

            resized = cv2.resize(face_img, IMAGE_SIZE)
            norm = resized / 255.0
            pred = smile_model.predict(np.expand_dims(norm, 0))[0][0]
            is_smiling = pred > 0.5

            encoding = None
            # for f in faces:
            #     x1, y1, x2, y2 = f.bbox
            #     if left < x1 < right and top < y1 < bottom:
            #         encoding = f.embedding.tolist()
            #         break

            if encoding:
                detected_faces.append(Face(
                    box=[left, top, right, bottom],
                    encoding=encoding,
                    is_smiling=is_smiling
                ))

    return DetectionResponse(faces=detected_faces)
