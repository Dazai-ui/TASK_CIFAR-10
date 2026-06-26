import sys

import streamlit as st
import torch

sys.path.append(".")
from app.predict import load_model, predict_image

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")
st.title("🖼️ CIFAR-10 Image Classifier")
st.write(
    "Upload an image and the trained CNN will predict which of the 10 "
    "CIFAR-10 categories it most resembles: airplane, automobile, bird, cat, "
    "deer, dog, frog, horse, ship, truck."
)


@st.cache_resource
def get_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(checkpoint_path, device), device


checkpoint_path = st.sidebar.text_input("Checkpoint path", "./models/best_model.pt")

try:
    model, device = get_model(checkpoint_path)
except FileNotFoundError:
    st.error(f"Checkpoint not found at `{checkpoint_path}`. Train a model first with `python -m src.train`.")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", width=256)

    with open("_tmp_upload.png", "wb") as f:
        f.write(uploaded.getbuffer())

    results = predict_image(model, "_tmp_upload.png", device, top_k=5)

    st.subheader("Predictions")
    for label, prob in results:
        st.write(f"**{label}** — {prob * 100:.2f}%")
        st.progress(min(prob, 1.0))
