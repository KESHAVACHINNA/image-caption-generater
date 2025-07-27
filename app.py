import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set Streamlit page config
st.set_page_config(page_title="🖼️ BLIP Image Captioning", layout="centered")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app uses [Salesforce's BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) model 
to generate a caption for any image you upload. It works best with natural scenes and objects.
""")

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    model.eval()
    return processor, model

processor, model = load_model()

# Main UI
st.title("📸 Image Caption Generator (BLIP)")
st.write("Upload an image and get a caption generated using a powerful vision-language model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("❌ Image too large! Please upload a file smaller than 5MB.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((384, 384))  # BLIP input size
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🧠 Generate Caption"):
            with st.spinner("Generating caption..."):
                try:
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    st.success(f"📌 **Caption**: {caption}")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")

# Optional styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
