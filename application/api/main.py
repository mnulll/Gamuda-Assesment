# FastAPI service for defect segmentation.
# Takes an image, checks its quality, runs the YOLOv8 model, returns the result.

import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Settings (can be changed with environment variables).
MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
BRIGHTNESS_MIN = float(os.getenv("BRIGHTNESS_MIN", "40"))
BRIGHTNESS_MAX = float(os.getenv("BRIGHTNESS_MAX", "220"))
BLUR_MIN = float(os.getenv("BLUR_MIN", "100"))

app = FastAPI()

# Allow the Streamlit app to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the app starts.
model = YOLO(MODEL_PATH)


def check_quality(image):
    # Measure brightness and blur. Returns the numbers and any warnings.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []
    if brightness < BRIGHTNESS_MIN:
        warnings.append("Image is too dark; results may be unreliable.")
    elif brightness > BRIGHTNESS_MAX:
        warnings.append("Image is too bright; results may be unreliable.")
    if blur_score < BLUR_MIN:
        warnings.append("Image is blurry; small defects may be missed.")

    return {
        "brightness": round(brightness, 1),
        "blur_score": round(blur_score, 1),
        "passed": len(warnings) == 0,
        "warnings": warnings,
    }


@app.get("/health")
def health():
    # Simple check that the service is running.
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image.
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    # Check quality (this does not stop inference).
    quality = check_quality(image)

    # Run the model.
    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    # Draw the masks and labels on the image.
    annotated = result.plot()
    ok, buffer = cv2.imencode(".png", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    # List what was found.
    detections = []
    if result.boxes is not None:
        for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            detections.append({
                "class_name": model.names[int(cls_id)],
                "confidence": round(float(conf), 3),
            })

    return {
        "annotated_image": annotated_b64,
        "detections": detections,
        "quality": quality,
    }
