
import streamlit as st
import cv2
import numpy as np
import av
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Page configuration
st.set_page_config(
    page_title="Real-Time Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

st.title("😷 Real-Time Face Mask Detection")
st.write("Live webcam detection using a trained CNN model.")

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_model.keras")

model = load_model()

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Could not load the OpenCV face detector.")
    st.stop()

# Prediction and drawing function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = float(model.predict(face, verbose=0)[0][0])

        if prediction < 0.5:
            label = "With Mask"
            confidence = (1 - prediction) * 100
            color = (0, 200, 0)
        else:
            label = "Without Mask"
            confidence = prediction * 100
            color = (0, 0, 255)

        text = f"{label}: {confidence:.1f}%"

        cv2.rectangle(
            img, (x, y), (x+w, y+h), color, 2
        )

        cv2.rectangle(
            img, (x, y-30), (x+w, y), color, -1
        )

        cv2.putText(
            img,
            text,
            (x+5, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.subheader("Live Webcam")

st.info(
    "Click START to enable your camera. "
    "Allow camera permission in your browser."
)

webrtc_streamer(
    key="face-mask-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)

st.caption(
    "Green box: With Mask | Red box: Without Mask"
)

