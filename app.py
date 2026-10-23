import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
IMG_SIZE = 160
@st.cache_resource
def load_model():
return tf.keras.models.load_model("model.keras")
with open("labels.json", "r", encoding="utf-8") as f:
LABELS = json.load(f)
st.set_page_config(page_title="PetVision AI", page_icon="■")
st.title("■ PetVision AI")
st.write("Upload a picture to classify it as cat, dog, or neither.")
uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded image", use_container_width=True)
resized = image.resize((IMG_SIZE, IMG_SIZE))
arr = np.asarray(resized, dtype="float32") / 255.0
arr = np.expand_dims(arr, axis=0)
with st.spinner("Classifying image..."):
model = load_model()
scores = model.predict(arr, verbose=0)[0]
idx = int(np.argmax(scores))
st.subheader(f"Prediction: {LABELS[idx].title()}")
st.write("Class scores")
for i, label in enumerate(LABELS):
st.write(f"{label.title()}: {scores[i] * 100:.1f}%")
st.caption("Scores are estimates, not guarantees. Results may be unreliable for unfamiliar images.")
