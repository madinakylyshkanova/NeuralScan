import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import *
from models.crnn import CRNN
from utils.dataset import WordDataset, ResizeAndPad, collate_fn
from utils.decode import decode_prediction
from utils.metrics import average_cer, word_accuracy
from utils.split_data import create_splits


def evaluate(model_path, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        ResizeAndPad(IMG_H, IMG_W),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Use same seed as training so test split is identical
    dataset = WordDataset("data/Dataset", "Labels.txt", transform, ALPHABET)
    _, _, test_ds = create_splits(dataset)

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CRNN(len(ALPHABET) + 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_truths, all_names = [], [], []

    with torch.no_grad():
        for imgs, _, _, words, names in loader:
            imgs = imgs.to(device)
            decoded = decode_prediction(model(imgs), ALPHABET)
            all_preds.extend(decoded)
            all_truths.extend(words)
            all_names.extend(names)

    acc = word_accuracy(all_preds, all_truths)
    cer = average_cer(all_preds, all_truths)

    #  Print summary
    print("=" * 50)
    print("        TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Total samples : {len(all_truths)}")
    print(f"  Word Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  CER           : {cer:.4f}")
    print("=" * 50)

    #  Sample predictions (first 20) 
    print("\nSample predictions (first 20):")
    print(f"{'GT':<20} {'Pred':<20} {'Match'}")
    print("-" * 45)
    for gt, pred in list(zip(all_truths, all_preds))[:20]:
        match = "✓" if gt == pred else "✗"
        print(f"{gt:<20} {pred:<20} {match}")

    #  Save full results 
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Word Accuracy: {acc:.4f}\n")
            f.write(f"CER: {cer:.4f}\n")
            f.write(f"Total samples: {len(all_truths)}\n\n")
            for name, gt, pred in zip(all_names, all_truths, all_preds):
                f.write(f"{name} | GT: {gt} | Pred: {pred}\n")
        print(f"\nFull results saved to: {output_path}")

    return acc, cer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CRNN on test set")
    parser.add_argument("--model", default="models/saved_models/crnn_best.pth",
                        help="Path to model weights")
    parser.add_argument("--output", default="results/test_results.txt",
                        help="Path to save full prediction results")
    args = parser.parse_args()

    evaluate(args.model, args.output)
