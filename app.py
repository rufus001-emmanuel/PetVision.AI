[24/10/2026 01:49] REG: ["cat", "dog", "neither"]
[24/10/2026 01:56] REG: import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="PetVision AI | Intelligent Image Analysis",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
LABELS_PATH = BASE_DIR / "labels.json"
IMG_SIZE = 160

# --------------------------------------------------
# PROFESSIONAL INTERFACE
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: #f5f7fb;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #102b26, #176b52);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 2.4rem;
    margin-bottom: 8px;
}
.hero p {
    color: #e0f4eb;
    font-size: 1rem;
}
.section-title {
    color: #153e32;
    font-size: 1.35rem;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 17px;
    border-radius: 12px;
}
div[data-testid="stFileUploader"] {
    background: white;
    padding: 14px;
    border-radius: 12px;
}
.small-note {
    color: #526173;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🐾 PetVision AI</h1>
    <p>Intelligent animal recognition, object detection,
    and visual image analytics.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EXISTING CLASSIFIER AND NEW OBJECT DETECTOR
# --------------------------------------------------
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_detector():
    return YOLO("yolov8n.pt")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🐾 PetVision AI")
    st.caption("Computer Vision Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Image Analysis", "About the Project"],
    )

    st.divider()
    st.markdown("Current capabilities")
    st.write("✓ Cat / dog / neither classification")
    st.write("✓ Multiple-object detection")
    st.write("✓ Animal and object counting")
    st.write("✓ Detection confidence scores")
    st.write("✓ Annotated image results")

    st.caption("Breed identification is not yet included.")

if page == "About the Project":
    st.header("About PetVision AI")
    st.write(
        "PetVision AI is a computer vision project combining "
        "a custom three-class image classifier with a pretrained "
        "object detector."
    )

    st.subheader("Models")
    st.write(
        "- Custom classifier: predicts cat, dog, or neither.\n"
        "- Object detector: identifies individual objects "
        "from its supported pretrained categories."
    )

    st.subheader("Current limitations")
    st.write(
        "The three-class model does not identify breeds. "
        "Object detection depends on the detector's supported "
        "categories and may miss small, hidden, or unusual objects. "
        "Confidence scores are not guarantees of correctness."
    )
    st.stop()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Upload an image for analysis</div>',
    unsafe_allow_html=True,
)
st.write(
    "Choose a JPG, JPEG, or PNG image. "
    "Images containing multiple animals or objects are supported."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)
[24/10/2026 01:56] REG: confidence_threshold = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help="Lower values may detect more objects but can also add false detections.",
)

if uploaded_file is None:
    st.info("Upload an image to begin your analysis.")

    st.markdown("### What you will receive")
    c1, c2, c3 = st.columns(3)
    c1.metric("Image input", "JPG / PNG")
    c2.metric("Analysis", "AI-powered")
    c3.metric("Results", "Visual + numeric")
    st.stop()

# Read the uploaded image safely.
try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception:
    st.error("This image could not be opened. Please upload a valid JPG or PNG.")
    st.stop()

st.markdown("### Your image")
st.image(image, use_container_width=True)

# --------------------------------------------------
# RUN BOTH MODELS
# --------------------------------------------------
if st.button("Analyze Image", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading AI models and analyzing your image..."):
            classifier = load_classifier()
            detector = load_detector()

            # Existing three-class classifier
            resized = image.resize((IMG_SIZE, IMG_SIZE))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            array = np.expand_dims(array, axis=0)

            with open(LABELS_PATH, "r", encoding="utf-8") as file:
                labels = json.load(file)

            scores = classifier.predict(array, verbose=0)[0]

            if len(scores) != len(labels):
                raise ValueError(
                    "The number of model outputs does not match labels.json."
                )

            best_index = int(np.argmax(scores))
            best_label = labels[best_index]
            best_score = float(scores[best_index]) * 100

            # Pretrained object detector
            detections = detector.predict(
                source=image,
                conf=confidence_threshold,
                verbose=False,
            )[0]

            # Draw boxes around detected objects
            annotated_bgr = detections.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1].copy()

            # Collect object detections
            rows = []
            if detections.boxes is not None:
                for box in detections.boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    coords = box.xyxy[0].tolist()
                    object_name = detections.names[class_id]

                    rows.append({
                        "Object": object_name.title(),
                        "Confidence (%)": round(score * 100, 1),
                        "X1": round(coords[0]),
                        "Y1": round(coords[1]),
                        "X2": round(coords[2]),
                        "Y2": round(coords[3]),
                    })

        st.success("Image analysis completed.")

        # ------------------------------------------
        # CLASSIFIER RESULTS
        # ------------------------------------------
        st.markdown("## 1. Image classification")

        c1, c2 = st.columns(2)
        c1.metric("Predicted class", best_label.title())
        c2.metric("Classifier score", f"{best_score:.1f}%")

        st.caption(
            "This prediction classifies the whole image into one of "
            "your existing three classes. It does not count animals."
        )

        st.markdown("Scores for all three classes")
        class_scores = pd.DataFrame({
            "Class": [name.title() for name in labels],
            "Score (%)": [
                round(float(value) * 100, 1) for value in scores
            ],
        })
        st.bar_chart(class_scores.set_index("Class"))
[24/10/2026 01:56] REG: # ------------------------------------------
        # OBJECT DETECTION RESULTS
        # ------------------------------------------
        st.markdown("## 2. Object detection")
        left, right = st.columns([1.25, 1])

        with left:
            st.image(
                annotated_rgb,
                caption="Detected objects outlined by the AI detector",
                use_container_width=True,
            )

        with right:
            total = len(rows)
            counts = Counter(row["Object"] for row in rows)

            # These are broad object categories, not breed predictions.
            animal_names = {
                "Cat", "Dog", "Bird", "Horse", "Sheep",
                "Cow", "Elephant", "Bear", "Zebra", "Giraffe",
            }
            animal_total = sum(
                count for name, count in counts.items()
                if name in animal_names
            )

            st.metric("Objects detected", total)
            st.metric("Animals detected", animal_total)
            st.metric(
                "Other detected objects",
                total - animal_total,
            )

            if total == 0:
                st.warning(
                    "No objects passed the confidence threshold. "
                    "Try lowering the threshold or using a clearer image."
                )
            else:
                st.markdown("Detected object counts")
                count_df = pd.DataFrame(
                    counts.items(), columns=["Object", "Count"]
                )
                st.dataframe(
                    count_df,
                    use_container_width=True,
                    hide_index=True,
                )

        # ------------------------------------------
        # PERCENTAGES
        # ------------------------------------------
        st.markdown("## 3. Image statistics")

        if total > 0:
            st.caption(
                "Percentages below mean each detected object's share "
                "of all detected objects. They do not represent the "
                "percentage of image area occupied."
            )

            percentage_df = pd.DataFrame([
                {
                    "Object": name,
                    "Count": count,
                    "Percentage of detections (%)":
                        round(count / total * 100, 1),
                }
                for name, count in counts.items()
            ]).sort_values("Count", ascending=False)

            st.dataframe(
                percentage_df,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                percentage_df.set_index("Object")[
                    "Percentage of detections (%)"
                ]
            )

            if animal_total > 0:
                st.markdown("Animal distribution")
                animal_counts = {
                    name: count for name, count in counts.items()
                    if name in animal_names
                }

                animal_df = pd.DataFrame([
                    {
                        "Animal": name,
                        "Count": count,
                        "Share of detected animals (%)":
                            round(count / animal_total * 100, 1),
                    }
                    for name, count in animal_counts.items()
                ])

                st.dataframe(
                    animal_df,
                    use_container_width=True,
                    hide_index=True,
                )

        # ------------------------------------------
        # DETAILED DETECTIONS + DOWNLOAD
        # ------------------------------------------
        st.markdown("## 4. Detection details")

        if rows:
            details = pd.DataFrame(rows)
            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True,
            )
[24/10/2026 01:56] REG: csv = details.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download detection results (CSV)",
                data=csv,
                file_name="petvision_detection_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("There are no individual detections to display.")

        # ------------------------------------------
        # LIMITATIONS
        # ------------------------------------------
        st.markdown("## 5. Interpretation and limitations")
        st.info(
            "Breed recognition and coat-colour analysis are not "
            "implemented in this version. The current classifier only "
            "predicts cat, dog, or neither. The pretrained detector "
            "recognizes supported object categories, not specific "
            "cat or dog breeds. Objects outside its categories, "
            "including grass as a scene/background, may not be "
            "identified correctly."
        )

    except FileNotFoundError as error:
        st.error(
            f"A required project file is missing: {error.filename}. "
            "Check that model.keras and labels.json are in the "
            "same repository folder as app.py."
        )
    except Exception as error:
        st.error(
            "Analysis failed. Check the deployment logs for details."
        )
        st.exception(error)

st.divider()
st.caption(
    "PetVision AI | Educational computer vision project. "
    "Predictions may be incorrect and should be independently checked."
)
