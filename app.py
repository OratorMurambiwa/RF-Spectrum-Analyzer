import json
import tempfile
import streamlit as st
import tensorflow as tf
from PIL import Image

from rf_utils import analyze_signal

st.set_page_config(page_title="RF Signal Analyzer", layout="centered")

MODEL_PATH = "rf_classifier.keras"
IMAGE_SIZE = (224, 224)

with open("kept_classes.json", "r") as f:
    CLASS_NAMES = json.load(f)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("RF Signal Analyzer")
st.write("Upload a spectrogram image to classify the signal and extract band parameters.")

uploaded = st.file_uploader("Upload a spectrogram (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        img_path = tmp.name

    result = analyze_signal(model, CLASS_NAMES, img_path, IMAGE_SIZE)

    st.subheader("Classification")
    st.write(f"Prediction: **{result['predicted_class']}**")
    st.write(f"Accuracy: **{result['confidence'] * 100:.2f}%**")

    st.subheader("Extracted Parameters")
    band = result["band_params"]
    if band is None:
        st.write("No clear band detected.")
    else:
        st.write(f"Center frequency (normalized): **{band['center_freq_norm']:.3f}**")
        st.write(f"Bandwidth (normalized): **{band['bandwidth_norm']:.3f}**")
        st.write(f"SNR-like score: **{band['snr_like']:.3f}**")
        st.write(f"Band columns: **{band['start_col']} → {band['end_col']}**")

    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2),
        file_name="rf_analysis.json",
        mime="application/json"
    )
