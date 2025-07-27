from flask import Flask, render_template, request
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return render_template("index.html", caption="No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template("index.html", caption="No file selected.")

    image = Image.open(file.stream).convert('RGB')

    # Preprocess and generate caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return render_template("index.html", caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
