"""
infer.py — Run inference on a single handwritten word image.
how to run: python infer.py path/to/image.png
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from config import *
from models.crnn import CRNN
from utils.dataset import ResizeAndPad
from utils.decode import decode_prediction


def load_model(model_path, device):
    model = CRNN(len(ALPHABET) + 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(img_path, model, transform, device):
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img)
        return decode_prediction(preds, ALPHABET)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten Azerbaijani word from image")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="models/saved_models/crnn_best.pth",
                        help="Path to model weights (default: crnn_best.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        ResizeAndPad(IMG_H, IMG_W),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = load_model(args.model, device)
    result = predict(args.image, model, transform, device)
    print(f"Predicted: {result}")
