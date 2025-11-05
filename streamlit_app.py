"""Streamlit application for COVID-19 chest X-ray classification."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from covid_xray.inference import (
    CLASS_NAMES,
    build_inference_transform,
    load_trained_model,
    predict_image,
)

BEST_MODEL_PATH = Path("best_model.pth")

st.set_page_config(page_title="COVID-19 X-ray Classifier", layout="centered")


@st.cache_resource
def load_cached_model() -> Tuple[torch.nn.Module, torch.device]:
    """Load the pretrained model and cache it for reuse.

    Returns:
        Tuple containing the pretrained model and associated device.
    """

    model, device = load_trained_model(BEST_MODEL_PATH)
    return model, device


def main() -> None:
    """Render the Streamlit user interface."""

    st.title("COVID-19 Chest X-ray Classifier")
    st.write(
        "Upload a chest X-ray image to receive a COVID-positive or COVID-negative "
        "prediction along with class probabilities."
    )

    if not BEST_MODEL_PATH.exists():
        st.error(
            "Model weights were not found. Ensure `best_model.pth` is located in the "
            "project root before launching the app."
        )
        return

    model, device = load_cached_model()
    transform = build_inference_transform()

    uploader = st.file_uploader(
        "Upload an X-ray image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    if uploader is None:
        st.info("Awaiting image upload.")
        return

    image = Image.open(uploader).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference..."):
        label, confidence, probabilities = predict_image(
            image_input=image,
            model=model,
            device=device,
            transform=transform,
        )

    st.success(f"Prediction: {label}")
    st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

    probability_df = pd.DataFrame(
        {
            "Class": list(CLASS_NAMES),
            "Probability": [value * 100 for value in probabilities.tolist()],
        }
    )
    probability_df.set_index("Class", inplace=True)
    st.bar_chart(probability_df)


if __name__ == "__main__":
    main()
