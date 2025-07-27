# app.py

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# App UI
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")
st.write("Upload an image to generate a caption using a Vision-Language Model (BLIP).")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.success("Caption generated!")
    st.subheader("📝 Caption:")
    st.write(caption)
