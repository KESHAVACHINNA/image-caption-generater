import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import numpy as np
from PIL import Image
from wordcloud import WordCloud
from fpdf import FPDF
from transformers import BlipProcessor, BlipForConditionalGeneration
import qrcode
import praw
import requests
import io

st.set_page_config(page_title="All-in-One Dashboard", layout="wide")
st.title("🧠 All-in-One Utility Dashboard")

# Sidebar Navigation
pages = ["Data Visualization", "Face Detection", "Word Cloud", "Image Caption", "Invoice Generator", "QR Generator", "Reddit Bot", "Weather"]
choice = st.sidebar.radio("Navigate", pages)

# Reddit API Setup
reddit = praw.Reddit(client_id='XklaFB2839fVxoJ7jrOlIQ',
                     client_secret='WyTB6nDQUvrEB0KiDnw-eOTMqzMZcg',
                     user_agent='my_bot_v1_pristine123')

# Image Captioning Setup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --- Pages --- #
if choice == "Data Visualization":
    st.subheader("📊 Monthly Sales Analysis")
    data = {
        "Month": ["January", "February", "March", "April", "May", "June"],
        "Sales": [200, 220, 250, 280, 300, 320],
        "Profit": [50, 70, 80, 90, 100, 110],
        "Customer_Count": [30, 35, 40, 50, 55, 60],
    }
    df = pd.DataFrame(data)

    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df, x="Month", y="Sales", marker='o', ax=ax1)
    st.pyplot(fig1)

    fig2 = px.line(df, x="Month", y=["Sales", "Profit"], title="Interactive Sales vs Profit")
    st.plotly_chart(fig2)

elif choice == "Face Detection":
    st.subheader("😊 Face Detection with OpenCV")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = np.array(Image.open(uploaded_file).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        st.image(img, caption="Detected Faces", use_column_width=True)

elif choice == "Word Cloud":
    st.subheader("☁️ Word Cloud Generator")
    text_input = st.text_area("Enter your text")
    if st.button("Generate Word Cloud") and text_input:
        wc = WordCloud(width=800, height=400, background_color='white').generate(text_input)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

elif choice == "Image Caption":
    st.subheader("🖼️ Image Caption Generator (BLIP)")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.success(f"Caption: {caption}")

elif choice == "Invoice Generator":
    st.subheader("📄 Invoice Generator")
    customer = st.text_input("Customer Name", value="John Cena")
    items = {"Laptop": 100, "Headphones": 10}
    if st.button("Generate Invoice"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(40, 10, 'Invoice')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(40, 10, f'Customer: {customer}')
        pdf.ln(10)
        total = 0
        for item, price in items.items():
            pdf.cell(40, 10, f'{item}: ${price}')
            pdf.ln(8)
            total += price
        pdf.ln(10)
        pdf.cell(40, 10, f'Total: ${total}')
        output = io.BytesIO()
        pdf.output(output)
        st.download_button("Download Invoice", output.getvalue(), file_name="invoice.pdf")

elif choice == "QR Generator":
    st.subheader("🔳 Artistic QR Code Generator")
    data = st.text_input("Enter Data to Encode")
    if st.button("Generate QR") and data:
        qr = qrcode.make(data)
        st.image(qr, caption="Generated QR", use_column_width=False)
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        st.download_button("Download QR", buf.getvalue(), file_name="qr.png")

elif choice == "Reddit Bot":
    st.subheader("🔥 Reddit Top Post Scraper")
    subreddit_name = st.text_input("Enter Subreddit", value="Python")
    if st.button("Fetch Top Posts"):
        subreddit = reddit.subreddit(subreddit_name)
        posts = [[post.title, str(post.author), post.score] for post in subreddit.top(limit=10)]
        df = pd.DataFrame(posts, columns=["Title", "Author", "Upvotes"])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"{subreddit_name}_top_posts.csv")

elif choice == "Weather":
    st.subheader("🌦️ Live Weather Info")
    city = st.text_input("Enter City")
    api_key = '7e1f6dcba3b83a0b8c3554d32e1b6e1c'
    if st.button("Get Weather") and city:
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description'].capitalize()
            st.success(f"{city.title()}: {temp}°C, {desc}")
        else:
            st.error("City not found or API error")
