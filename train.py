import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import random
import os

from config import *
from models.crnn import CRNN
from utils.dataset import WordDataset, ResizeAndPad, collate_fn
from utils.decode import decode_prediction
from utils.metrics import average_cer, word_accuracy
from utils.split_data import create_splits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# augmentation
class HandwritingAugment:
    """Light augmentations suited for small handwriting datasets."""
    def __call__(self, img):
        # Random slight rotation
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle, fill=255)

        # Random horizontal stretch
        if random.random() < 0.4:
            w, h = img.size
            new_w = int(w * random.uniform(0.85, 1.15))
            new_w = max(1, new_w)
            img = img.resize((new_w, h), Image.BILINEAR)

        # Random brightness/contrast jitter
        if random.random() < 0.4:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        return img

# Transforms
train_transform = transforms.Compose([
    HandwritingAugment(),
    ResizeAndPad(IMG_H, IMG_W),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    ResizeAndPad(IMG_H, IMG_W),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# keep 'transform' alias for backward compatibility (infer/evaluate scripts)
transform = val_transform

#folders
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

os.makedirs("models/saved_models", exist_ok=True)

# DATA 
# Build two dataset views: one with augmentation (train), one without (val/test)
train_dataset = WordDataset("data/Dataset", "Labels.txt", train_transform, ALPHABET)
val_dataset   = WordDataset("data/Dataset", "Labels.txt", val_transform,   ALPHABET)

train_ds, _, _       = create_splits(train_dataset)
_, val_ds, test_ds   = create_splits(val_dataset)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# model
model = CRNN(len(ALPHABET) + 1).to(device)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Halve LR when CER stops improving for 10 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=10, factor=0.5
)

best_cer = float("inf")

# training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, targets, lengths, _, _ in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        preds = model(imgs)  # (T, B, C)
        preds = preds.log_softmax(2)

        T = preds.size(0)
        B = imgs.size(0)

        input_lengths = torch.full(
            (B,),
            T,
            dtype=torch.long,
            device=device
        )

        loss = criterion(preds, targets, input_lengths, lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # validation
    model.eval()
    preds_all, gt_all = [], []

    with torch.no_grad():
        for imgs, _, _, words, _ in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            decoded = decode_prediction(preds, ALPHABET)

            preds_all.extend(decoded)
            gt_all.extend(words)

    cer_val = average_cer(preds_all, gt_all)
    acc = word_accuracy(preds_all, gt_all)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1} | Loss {avg_loss:.4f} | CER {cer_val:.4f} | Acc {acc:.4f} | LR {current_lr:.6f}")

    scheduler.step(cer_val)

    # saving the best model
    if cer_val < best_cer:
        best_cer = cer_val
        torch.save(model.state_dict(), "models/saved_models/crnn_best.pth")
        print("🔥 Best model saved!")

    # save every 5th epoch's result
    if (epoch + 1) % 5 == 0:
        epoch_folder = os.path.join(results_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_folder, exist_ok=True)

        # save model
        torch.save(model.state_dict(), os.path.join(epoch_folder, "model.pth"))

        # save ALL predictions
        with open(os.path.join(epoch_folder, "predictions.txt"), "w", encoding="utf-8") as f:
            for pred, gt in zip(preds_all, gt_all):
                f.write(f"GT: {gt} | Pred: {pred}\n")