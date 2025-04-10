import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load model
model = load_model('model.keras')  # Ganti nama file sesuai modelmu
st.title("Real-Time Detection with Camera")

# Fungsi untuk prediksi dari gambar
def predict_image(image):
    img = cv2.resize(image, (224, 224))  # Sesuaikan ukuran input model
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

# Aktifkan kamera
camera = st.camera_input("Take a picture using the camera")

if camera is not None:
    img = Image.open(camera)
    img_np = np.array(img)

    st.image(img_np, caption="Captured Image", use_container_width=True)
    pred = predict_image(img_np)

    # Misalnya model output sigmoid
    label = "Detected" if pred[0][0] > 0.5 else "No Pothole"
    st.subheader(f"Prediction Result: {label} ({pred[0][0]:.2f})")
