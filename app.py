import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load the trained model (.h5 file)
@st.cache_resource
def load_cnn_model():
    model_path = "C:/Users/bened/Documents/Alyster Coding/PROJECTS/CV/resnet50_bbox_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check the path.")
        return None
    return load_model(model_path)

model = load_cnn_model()

# Image preprocessing and prediction
def predict_bounding_box(img):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        prediction = model.predict(img_expanded)[0] if model else None
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Visualize bounding box on the image
def draw_bbox(image, bbox):
    if bbox is None:
        return image
    
    h, w = image.shape[:2]
    x_min = max(int(bbox[0] * w), 0)
    y_min = max(int(bbox[1] * h), 0)
    x_max = min(int(bbox[2] * w), w)
    y_max = min(int(bbox[3] * h), h)

    # Draw green box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

# Streamlit UI
st.title("🚗 Traffic Sign/Car Bounding Box Detector")
st.write("Upload an image and the model will predict the bounding box.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)

        st.image(image_np, caption='Uploaded Image', use_column_width=True)

        with st.spinner("Predicting..."):
            bbox = predict_bounding_box(image_np)
            if bbox is None:
                st.error("Prediction failed. Please check your model and input.")
            else:
                result_image = draw_bbox(image_np.copy(), bbox)
                st.success("Prediction done!")
                st.image(result_image, caption='Detected Bounding Box', use_column_width=True)
    except Exception as e:
        st.error(f"Error processing the image: {e}")