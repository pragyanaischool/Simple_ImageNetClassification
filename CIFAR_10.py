import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Preprocess image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to fit ResNet input shape
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Load pre-trained ResNet model
model = ResNet50(weights='imagenet')

# Function to classify image
def classify_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    class_name = decoded_predictions[0][1]
    return class_name

# Streamlit UI
st.title("CIFAR-10 Image Classification Demo")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Classify the image
    predicted_class = classify_image(uploaded_file)

    st.write(f"Prediction: {predicted_class}")

