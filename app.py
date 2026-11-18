
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_model.keras")

model = load_model()

# Load OpenCV face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("😷 Face Mask Detection")
st.write("Upload an image to detect faces and mask status.")

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    # Convert RGB image to grayscale for face detection
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    st.write("Faces detected:", len(faces))

    for (x, y, w, h) in faces:
        face = image_array[y:y+h, x:x+w]

        # Prepare face for CNN
        face_image = Image.fromarray(face).resize((128, 128))
        face_array = np.array(face_image) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        prediction = float(model.predict(face_array, verbose=0)[0][0])

        if prediction < 0.5:
            label = "With Mask"
            color = (0, 255, 0)
            confidence = (1 - prediction) * 100
        else:
            label = "Without Mask"
            color = (255, 0, 0)
            confidence = prediction * 100

        # Draw rectangle and prediction
        cv2.rectangle(
            image_array,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        cv2.putText(
            image_array,
            f"{label}: {confidence:.1f}%",
            (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    st.image(
        image_array,
        caption="Face Detection Results",
        width="stretch"
    )
