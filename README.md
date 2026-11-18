# Face Mask Detection Using CNN

## Project Overview
A deep learning project that detects whether a person is wearing a face mask using a Convolutional Neural Network (CNN).

## Features
- Image upload mask detection
- Real-time webcam mask detection
- Face detection using OpenCV
- FastAPI prediction backend
- Streamlit interface

## Dataset
Kaggle Face Mask Dataset:
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

- With Mask: 3,725 images
- Without Mask: 3,828 images
- Total: 7,553 images

## CNN Architecture
- Conv2D: 32 filters
- Conv2D: 64 filters
- Conv2D: 128 filters
- MaxPooling
- Flatten
- Dense: 128 neurons
- Dropout: 0.5
- Output: 1 sigmoid neuron

Input size: 128 x 128
Optimizer: Adam
Loss: Binary Crossentropy

## Model Performance
Validation Accuracy: 96.09%

See accuracy.png for the accuracy graph.

## Technologies
Python, TensorFlow, Keras, CNN, OpenCV, Streamlit, FastAPI.

## Installation
1. Clone this repository.
2. Install dependencies using pip install -r requirements.txt.
3. Run streamlit run app.py.

## Webcam Application
Start the FastAPI server:
python -m uvicorn predict_server:app --host 0.0.0.0 --port 8000

In another terminal:
streamlit run webcam_app.py

Open http://localhost:8501

## Author
Parminder
