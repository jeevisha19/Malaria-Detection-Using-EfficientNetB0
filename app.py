import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Malaria Detection using EfficientNetB0",
    page_icon="🦠",
    layout="centered"
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_malaria_model.keras")
    return model

model = load_model()

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.title("🦠 Malaria Detection using EfficientNetB0")

st.markdown(
"""
## AI-Based Malaria Cell Detection System

This application uses **EfficientNetB0** and **Deep Learning**
to classify microscopic blood smear images into:

- 🦠 **Parasitized**
- ✅ **Uninfected**
"""
)

st.markdown("---")

# ----------------------------------------------------
# IMAGE UPLOAD
# ----------------------------------------------------
st.subheader("📤 Upload Blood Smear Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("🔍 Analyzing Blood Smear Image..."):

        # Resize image
        img = image.resize((224, 224))

        # Convert to numpy
        img = np.array(img).astype(np.float32)

        # SAME preprocessing used during training
        img = tf.keras.applications.efficientnet.preprocess_input(img)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img, verbose=0)

        probability = float(prediction[0][0])

        if probability < 0.5:
            predicted = "Parasitized"
            confidence = (1 - probability) * 100
        else:
            predicted = "Uninfected"
            confidence = probability * 100

    st.markdown("---")

    if predicted == "Parasitized":
        st.error(f"🦠 Prediction: {predicted}")
    else:
        st.success(f"✅ Prediction: {predicted}")

    st.markdown(
        f"### Confidence Score: **{confidence:.2f}%**"
    )

    st.progress(confidence / 100)

# ----------------------------------------------------
# ABOUT PROJECT
# ----------------------------------------------------
st.markdown("---")

st.info(
"""
### 📖 About this Project

This application is powered by **EfficientNetB0**, a state-of-the-art
Convolutional Neural Network (CNN) trained using **Transfer Learning**
on microscopic blood smear images.

The model classifies red blood cells into:

- 🦠 Parasitized
- ✅ Uninfected

**Dataset Used:** NIH Malaria Cell Images Dataset

**Technologies Used**

- Python
- TensorFlow / Keras
- EfficientNetB0
- Streamlit
- NumPy
- Pillow
"""
)

# ----------------------------------------------------
# DISCLAIMER
# ----------------------------------------------------
st.warning(
"""
### ⚠ Disclaimer

This application is intended **only for educational and research purposes**.

It should **NOT** be used as a substitute for professional
medical diagnosis or clinical decision making.
"""
)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")

st.caption(
"""
Developed by **Jeevisha Satish**

B.Tech Computer Science & Engineering (AI & ML)

Amity University, Lucknow Campus
"""
)