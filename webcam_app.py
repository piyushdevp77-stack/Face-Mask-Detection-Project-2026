
import streamlit as st
import cv2
import av
import requests

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Face Mask Detection", layout="wide")

st.title("Real-Time Face Mask Detection")
st.write("AI-powered face mask recognition using your trained CNN model.")

API_URL = "http://127.0.0.1:8000/predict"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


class FaceMaskDetector(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_label = "Checking..."
        self.last_confidence = 0
        self.last_error = ""

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        self.frame_count += 1

        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]

            # Send one face image to the separate AI server
            if self.frame_count % 5 == 0:
                success, encoded = cv2.imencode(".jpg", face)

                if success:
                    try:
                        response = requests.post(
                            API_URL,
                            data=encoded.tobytes(),
                            headers={"Content-Type": "image/jpeg"},
                            timeout=10
                        )

                        response.raise_for_status()
                        result = response.json()

                        self.last_label = result["prediction"]
                        self.last_confidence = result["confidence"]
                        self.last_error = ""

                    except requests.RequestException as e:
                        self.last_error = str(e)
                        self.last_label = "Server Error"

            if self.last_label == "With Mask":
                color = (0, 255, 0)
            elif self.last_label == "Without Mask":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            cv2.rectangle(
                img, (x, y), (x+w, y+h), color, 3
            )

            text = f"{self.last_label}: {self.last_confidence:.1f}%"

            cv2.putText(
                img,
                text,
                (x, max(y-10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        return av.VideoFrame.from_ndarray(
            img, format="bgr24"
        )


webrtc_streamer(
    key="face-mask-detection",
    video_processor_factory=FaceMaskDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=False
)

st.info("Click Start to activate your webcam.")
