import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import qrcode
from fpdf import FPDF
from wordcloud import WordCloud
import requests
import io # For handling in-memory files

# --- Global/Cached Resources (IMPORTANT for Streamlit) ---
@st.cache_resource # Cache the model to avoid re-loading on every interaction
def load_blip_model():
    # Use st.spinner to show loading feedback
    with st.spinner("Loading BLIP model... (This may take a moment the first time)"):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load BLIP model once for the app
blip_processor, blip_model = load_blip_model()

# --- Streamlit App Structure ---
st.set_page_config(
    page_title="All-In-One Utility App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✨ All-In-One Python Utility App ✨")
st.markdown("---")

# Use a sidebar for navigation (like your Tkinter tabs)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a Utility",
    ["📊 Data Visualization", "🖼️ Image Processing", "🌐 Web Tools", "🧠 NLP & AI", "📄 PDF & QR", " Reddit"]
)

if app_mode == "📊 Data Visualization":
    st.header("📊 Data Visualization")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully!")
            st.dataframe(df.head()) # Display first few rows

            st.subheader("Static Graphs (Matplotlib/Seaborn)")
            if 'Month' in df.columns and 'Sales' in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.lineplot(x="Month", y="Sales", data=df, marker='o', ax=axes[0])
                axes[0].set_title("Monthly Sales (Static)")
                axes[0].tick_params(axis='x', rotation=45)

                if 'Profit' in df.columns:
                    sns.scatterplot(x="Sales", y="Profit", data=df, ax=axes[1])
                    axes[1].set_title("Sales vs Profit (Static)")
                plt.tight_layout()
                st.pyplot(fig) # Display the Matplotlib figure
            else:
                st.warning("For static plots, please ensure your CSV has 'Month' and 'Sales' (and optionally 'Profit') columns.")

            st.subheader("Interactive Dashboard (Plotly)")
            if 'Sales' in df.columns and 'Month' in df.columns:
                if 'Profit' in df.columns:
                    fig_px = px.line(df, x="Month", y=["Sales", "Profit"], title="Interactive Sales & Profit Dashboard")
                else:
                    fig_px = px.line(df, x="Month", y="Sales", title="Interactive Sales Dashboard")
                st.plotly_chart(fig_px, use_container_width=True) # Display the Plotly figure
            else:
                st.warning("For interactive plots, please ensure your CSV has 'Month' and 'Sales' (and optionally 'Profit') columns.")

        except Exception as e:
            st.error(f"Error loading or plotting data: {e}")

elif app_mode == "🧠 NLP & AI":
    st.header("🧠 NLP & AI")

    st.subheader("WordCloud Generator")
    wordcloud_text = st.text_area("Enter text for WordCloud:", height=150)
    if st.button("Generate WordCloud"):
        if wordcloud_text:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
                fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title("Generated WordCloud")
                st.pyplot(fig_wc)
            except Exception as e:
                st.error(f"Error generating WordCloud: {e}")
        else:
            st.warning("Please enter some text for the WordCloud.")

    st.subheader("Image Captioning (BLIP Model)")
    st.info(f"BLIP Model is running on: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

    uploaded_image = st.file_uploader("Upload an image for captioning", type=["png", "jpg", "jpeg", "gif", "bmp"])

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Generating caption..."):
                inputs = blip_processor(image, return_tensors="pt") # No .to(device) needed, Streamlit handles it
                # Ensure model is on the correct device if running on GPU
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    blip_model.to("cuda") # Ensure model is on GPU for inference

                out = blip_model.generate(**inputs)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)
            st.success("Caption Generated!")
            st.write(f"**Caption:** {caption}")
        except Exception as e:
            st.error(f"Error generating image caption: {e}")
            st.info("Ensure the BLIP model is fully loaded. This can take a while the first time.")

# Add similar elif blocks for other tabs:
# elif app_mode == "🌐 Web Tools":
#     st.header("🌐 Web Tools")
#     # Implement weather and other web tools here
#     city = st.text_input("Enter City for Weather:", "Bengaluru")
#     if st.button("Get Weather"):
#         # ... call weather API ...
#         st.write("Weather info here...")

# elif app_mode == "📄 PDF & QR":
#     st.header("📄 PDF & QR")
#     # Implement QR and PDF generation here
#     qr_data = st.text_input("Enter text/URL for QR Code:")
#     if st.button("Generate QR Code"):
#         if qr_data:
#             img = qrcode.make(qr_data)
#             buf = io.BytesIO()
#             img.save(buf, format="PNG")
#             byte_im = buf.getvalue()
#             st.image(byte_im, caption="Generated QR Code")
#             st.download_button(label="Download QR Code", data=byte_im, file_name="qrcode.png", mime="image/png")
#         else:
#             st.warning("Please enter data for QR code.")
#     # ... PDF generation ...

# elif app_mode == " Reddit":
#     st.header(" Reddit")
#     # Implement Reddit features here
#     # Remember to use st.secrets for API keys
#     # st.text_input("Subreddit:")
#     # st.button("Fetch Posts")
