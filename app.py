import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 160
MODEL_PATH =
Path(__file__).with_name("model.keras")
LABELS_PATH =
Path(__file__).with_name("labels.json")

st.set_page_config(
    page_title="PetVision AI",
    page_icon="🐾",
    layout="centered"
)


@st.cache_resource
def load_model():
    """Load the trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.keras is missing. Put it in the same folder as app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)


def load_labels():
    """Load the class names."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            "labels.json is missing. Put it in the same folder as app.py."
        )

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(labels, list) or not labels:
        raise ValueError(
            "labels.json must contain a list of class names."
        )

    return labels


def prepare_image(image):
    """Resize and prepare an image for prediction."""
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return np.expand_dims(image_array, axis=0)


st.title("🐾 PetVision AI")

st.write(
    "Upload a picture to classify it as "
    "cat, dog, or neither."
)

st.caption(
    "Predictions are estimates, not guarantees. "
    "The model may make mistakes on unfamiliar images."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        with st.spinner("Classifying your image..."):
            model = load_model()
            labels = load_labels()

            image_array = prepare_image(image)

            scores = model.predict(
                image_array,
                verbose=0
            )[0]

        if len(scores) != len(labels):
            st.error(
                "The model output does not match labels.json. "
                "Check that both files belong to the same model."
            )

        else:
            best_index = int(np.argmax(scores))

            st.subheader(
                f"Prediction: {str(labels[best_index]).title()}"
            )

            st.write("Class scores:")

            for index, label in enumerate(labels):
                percentage = float(scores[index]) * 100

                st.write(
                    f"{str(label).title()}: "
                    f"{percentage:.1f}%"
                )

    except FileNotFoundError as error:
        st.error(str(error))

    except ValueError as error:
        st.error(str(error))

    except UnidentifiedImageError:
        st.error(
            "The uploaded file is not a readable image. "
            "Please choose a JPG or PNG file."
        )

    except Exception as error:
        st.error(
            "The app could not complete the prediction. "
            "See the error details below."
        )
        st.exception(error)

else:
    st.info(
        "Choose a JPG or PNG image above to get a prediction."
    )
