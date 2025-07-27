import streamlit as st
from PIL import Image
import requests

HF_API_TOKEN = "YOUR_HF_API_TOKEN"

st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator (HuggingFace API)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            img_bytes = uploaded_file.getvalue()
            response = requests.post(
                "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
                headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                data=img_bytes,
            )
            if response.status_code == 200:
                caption = response.json()[0]["generated_text"]
                st.success("Caption:")
                st.write(caption)
            else:
                st.error(f"Error: {response.status_code} - {response.json()}")
