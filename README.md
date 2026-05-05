# NeuralScan — Azerbaijani Handwritten Word Recognition

**Course:** CSCI 4701 — Deep Learning, Spring 2026  
**Student:** Madina Kylyshkanova  

---

## Project Goal

Build and evaluate a deep learning system that recognizes handwritten Azerbaijani words from images. The core research question: **can simple data augmentation techniques significantly improve recognition accuracy when training data is limited?**

We compare three experimental settings on the same architecture and held-out test set:
1. Baseline CRNN — no augmentation
2. Augmented CRNN — handwriting-specific augmentations
3. Augmented CRNN + Synthetic Data — augmentation + 987 synthetically generated images

---

## Team Responsibilities

| Member | Contributions |
|---|---|
| Madina Kylyshkanova | Dataset collection and labeling, model architecture design, training pipeline, data augmentation, synthetic data generation, evaluation, web demo UI, analysis and report |

---

## Project Structure

```
NeuralScan/
├── notebook.ipynb             # Main graded notebook (run on Colab)
├── app.py                     # Flask web demo
├── train.py                   # Training script
├── evaluate.py                # Test set evaluation
├── infer.py                   # Single-image CLI inference
├── generate_synthetic.py      # Synthetic data generator
├── config.py                  # Hyperparameters and alphabet
├── models/
│   ├── crnn.py                # CRNN architecture
│   └── saved_models/
│       └── crnn_best.pth      # Best trained model weights
├── utils/
│   ├── dataset.py             # Dataset loader + transforms
│   ├── decode.py              # CTC greedy decoder
│   ├── metrics.py             # CER and word accuracy
│   └── split_data.py          # Train/val/test split (80/10/10)
├── data/
│   └── Dataset/               # Images + Labels.txt
└── results/
    └── test_results.txt       # Final test set predictions
```

---

## Setup

```bash
pip install -r requirements.txt
pip install flask   # only needed for web demo
```

---

## Running on Google Colab

1. Open `notebook.ipynb` in Google Colab
2. Update the `git clone` URL in cell 1 to your repository
3. Run all cells top to bottom
4. The notebook reproduces all experiments and plots without modification

---

## Scripts

**Train:**
```bash
python train.py
```

**Evaluate on test set:**
```bash
python evaluate.py
```

**Single image inference:**
```bash
python infer.py path/to/image.png
```

**Web demo:**
```bash
python app.py
# open http://localhost:5000
```

---

## Results

| Model | Dataset Size | Word Accuracy | CER |
|---|---|---|---|
| Baseline (no augmentation) | 1,149 | 20.00% | ~0.80 |
| Augmented CRNN | 1,149 | 44.35% | 0.174 |
| Augmented CRNN + Synthetic Data | 2,136 | **53.27%** | **0.174** |

---

## Analysis of Findings

### What Worked

**Data augmentation** was the most impactful change — doubling accuracy from 20% to 44%. Random rotation, horizontal stretch, and brightness jitter simulate natural handwriting variation and help the model generalize despite the small dataset.

**BatchNorm on all conv layers** stabilized training. The original model only had it on 1 of 7 layers, causing unstable gradients.

**ReduceLROnPlateau scheduler** prevented early plateau. The LR dropped automatically (3e-4 → ~5e-6) allowing continued improvement past epoch 150.

**Synthetic data** gave an additional +9% by increasing dataset diversity from 1,149 to 2,136 images using a cursive handwriting font with varied size, rotation, noise, and blur.

### What Did Not Work / Limitations

1. **Mixed-language dataset**: The dataset contains both Azerbaijani and English words. With ~1,100 real images split across two languages, the model lacks sufficient examples of either to learn both reliably. English loanwords like *compensation* and *headband* are predicted poorly.

2. **Long words**: The model struggles with words longer than 8 characters. CTC with a small dataset doesn't generalize well to long sequences.

3. **Single synthetic font**: Synthetic images use one font (Coal-Hand-Luke), introducing a distribution gap with real handwriting. More font diversity or elastic distortion would likely help.

4. **Dataset ceiling**: 2,136 total images is fundamentally small. Models like this typically need 10,000+ samples for reliable generalization — the 53% ceiling reflects this hard constraint.

### Conclusion

Clear evidence that data augmentation significantly improves handwriting recognition in low-resource settings. Accuracy improved from 20% to 53% (2.65× increase), demonstrating that targeted augmentation strategies can partially compensate for small dataset size — a practical finding for low-resource language OCR where large annotated datasets are unavailable.

---

## Model Architecture

- **CNN**: 7-layer convolutional feature extractor with BatchNorm after every layer
- **RNN**: 2-layer Bidirectional LSTM (hidden size 256)
- **Loss**: CTC (Connectionist Temporal Classification)
- **Decoder**: Greedy CTC decode
- **Dropout**: 0.1

## Hyperparameters

| Parameter | Value |
|---|---|
| Image height | 32 |
| Image width | 128 |
| Batch size | 16 |
| Epochs | 200 |
| Learning rate | 3e-4 (ReduceLROnPlateau, patience=10, factor=0.5) |
| Optimizer | Adam |

## Dataset

- **1,149 real handwritten images** — collected from multiple writers covering the full Azerbaijani Latin alphabet
- **987 synthetic images** — generated using `generate_synthetic.py` with Coal-Hand-Luke handwriting font, randomized size, rotation, noise, and blur
- **Split**: 80% train / 10% val / 10% test (fixed seed for reproducibility)
