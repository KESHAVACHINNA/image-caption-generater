from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

app = Flask(__name__)

# Load model only once and force CPU
device = torch.device("cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')

    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
