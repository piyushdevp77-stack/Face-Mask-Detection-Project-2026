
from fastapi import FastAPI, Request, HTTPException
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

print("Loading face mask model...")

model = tf.keras.models.load_model("face_mask_model.keras")

print("Face mask model loaded successfully!")

@app.get("/")
def home():
    return {"message": "Face Mask Prediction Server is running"}

@app.post("/predict")
async def predict(request: Request):

    image_bytes = await request.body()

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = float(model.predict(image, verbose=0)[0][0])

    if prediction < 0.5:
        label = "With Mask"
        confidence = (1 - prediction) * 100
    else:
        label = "Without Mask"
        confidence = prediction * 100

    return {
        "prediction": label,
        "confidence": round(float(confidence), 2)
    }
