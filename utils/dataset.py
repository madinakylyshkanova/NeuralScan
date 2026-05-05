import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ResizeAndPad:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        w, h = img.size
        scale = self.height / h
        new_w = int(w * scale)

        img = img.resize((new_w, self.height), Image.BILINEAR)

        if new_w > self.width:
            img = img.crop((0, 0, self.width, self.height))
        else:
            new_img = Image.new("L", (self.width, self.height), color=255)
            new_img.paste(img, (0, 0))
            img = new_img

        return img


class WordDataset(Dataset):
    def __init__(self, dataset_dir, labels_file, transform, alphabet):
        self.samples = []
        self.transform = transform
        self.alphabet = alphabet
        self.char2idx = {c: i + 1 for i, c in enumerate(alphabet)}

        label_path = Path(dataset_dir) / labels_file

        with open(label_path, encoding="utf-8") as f:
            for line in f:
                name, word = line.strip().split(maxsplit=1)

                if any(c not in self.char2idx for c in word):
                    continue

                self.samples.append((name, Path(dataset_dir) / name, word))

        print("Total valid samples:", len(self.samples))

    def encode(self, word):
        return torch.tensor(
            [self.char2idx[c] for c in word],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, path, word = self.samples[idx]

        img = Image.open(path).convert("L")
        img = self.transform(img)

        label = self.encode(word)

        return img, label, word, name


def collate_fn(batch):
    imgs, labels, words, names = zip(*batch)

    imgs = torch.stack(imgs)

    labels = [l.long() for l in labels]
    targets = torch.cat(labels)

    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    return imgs, targets, lengths, words, names