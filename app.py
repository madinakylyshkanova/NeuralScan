"""
app.py - file for Ui,  Flask web server for project
open: http://localhost:5000
"""

import os
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
from io import BytesIO
import base64

from config import *
from models.crnn import CRNN
from utils.dataset import ResizeAndPad
from utils.decode import decode_prediction

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    ResizeAndPad(IMG_H, IMG_W),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model once at startup
MODEL_PATH = "models/saved_models/crnn_best.pth"
model = None

def load_model():
    global model
    m = CRNN(len(ALPHABET) + 1).to(device)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    model = m
    print(f"Model loaded from {MODEL_PATH}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(BytesIO(file.read())).convert("L")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            prediction = decode_prediction(output, ALPHABET)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Make sure crnn_best.pth is in models/saved_models/")
    else:
        load_model()
        print("Starting server at http://localhost:5000")
        app.run(debug=False, port=5000)
