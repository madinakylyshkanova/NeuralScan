"""
Generates synthetic Azerbaijani handwriting-style word images using
the Coal-Hand-Luke.ttf handwriting font
How to use:python generate_synthetic.py --count 1500
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter


AZ_WORDS = [
    "adam","qadın","uşaq","ev","iş","vaxt","il","gün","yol","su",
    "baş","əl","göz","qol","dil","səs","söz","ad","yer","hava",
    "od","torpaq","dağ","dəniz","çay","meşə","kənd","şəhər","ölkə","dünya",
    "həyat","ölüm","sevgi","dostluq","ailə","ana","ata","qardaş","bacı","oğul",
    "qız","nənə","baba","əmi","xala","dayı","bibi","körpə","gənc","qoca",
    "getmək","gəlmək","görmək","bilmək","olmaq","vermək","almaq","demək","etmək","açmaq",
    "baxmaq","oxumaq","yazmaq","işləmək","sevmək","gülmək","ağlamaq","oynamaq","yemək","içmək",
    "yatmaq","durmaq","gəzmək","danışmaq","dinləmək","düşünmək","anlamaq","başlamaq","bitirmək","tapmaq",
    "böyük","kiçik","yaxşı","pis","yeni","köhnə","gözəl","çirkin","isti","soyuq",
    "uzun","qısa","ağır","yüngül","sürətli","yavaş","güclü","zəif","ağıllı","axmaq",
    "xoşbəxt","kədərli","sağlam","xəstə","dolu","boş","həqiqi","vacib","maraqlı","çətin",
    "əlifba","şəkil","çiçək","göyərçin","üzüm","ördək","işıq","özüm","öyrən","üzgün",
    "əsas","şair","şənlik","ərazi","şimal","cənub","qərb","şərq","ətraf","özəl",
    "ünsür","əmək","şərt","əkin","öküz","üzər","çörək","əhval","öhdəlik","şəxs",
    "bir","iki","üç","dörd","beş","altı","yeddi","səkkiz","doqquz","on",
    "iyirmi","otuz","əlli","yüz","min","sıfır","birinci","ikinci","üçüncü","axırıncı",
    "sabah","dünən","həftə","ay","saat","dəqiqə","saniyə",
    "Azərbaycan","Bakı","Türkiyə","Rusiya","Xəzər","Araz","Kür","Qarabağ","Naxçıvan","Gəncə",
    "Sumqayıt","Şirvan","Lənkəran","Mingəçevir",
    "kompüter","internet","telefon","proqram","sistem","texnologiya","universitet","məktəb",
    "müəllim","tələbə","kitab","dəftər","qələm","sinif","imtahan","bilik","təhsil","diplom",
    "ürək","sümük","əzələ","damar","qan",
    "balıq","toyuq","yumurta","pendir","süd","düyü","kartof",
    "pomidor","xiyar","soğan","sarımsaq","limon","alma","armud","üzüm","ərik",
    "rəng","musiqi","rəqs","tarix","coğrafiya","riyaziyyat","fizika","kimya","biologiya",
]


def generate_word_image(word, font_path, img_h=32, img_w=128):
    font_size = random.randint(22, 32)
    font = ImageFont.truetype(font_path, font_size)

    bg  = random.randint(245, 255)
    ink = random.randint(0, 20)

    canvas = Image.new("L", (800, 80), color=bg)
    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = 50
    y = (80 - th) // 2 + random.randint(-3, 3)
    draw.text((x, y), word, font=font, fill=ink)

    bbox2 = canvas.getbbox()
    if bbox2 is None:
        bbox2 = (0, 0, 800, 80)
    pad = random.randint(3, 8)
    canvas = canvas.crop((
        max(0, bbox2[0] - pad), max(0, bbox2[1] - pad),
        min(800, bbox2[2] + pad), min(80, bbox2[3] + pad)
    ))

    if random.random() < 0.6:
        canvas = canvas.rotate(random.uniform(-5, 5), fillcolor=bg, expand=True)

    w, h = canvas.size
    scale = img_h / max(h, 1)
    new_w = int(w * scale)
    canvas = canvas.resize((max(1, new_w), img_h), Image.BILINEAR)

    if new_w > img_w:
        canvas = canvas.crop((0, 0, img_w, img_h))
    else:
        padded = Image.new("L", (img_w, img_h), color=255)
        padded.paste(canvas, (0, 0))
        canvas = padded

    # verify not blank
    arr = np.array(canvas, dtype=np.float32)
    if arr.min() > 200:
        return generate_word_image(word, font_path, img_h, img_w)

    arr += np.random.normal(0, random.uniform(2, 7), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(arr)

    if random.random() < 0.3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",     type=int, default=1500,
                        help="Number of images to generate (default: 1500)")
    parser.add_argument("--output",    default="data/Dataset",
                        help="Output folder (default: data/Dataset)")
    parser.add_argument("--labels",    default="data/Dataset/Labels.txt",
                        help="Labels file to append to (default: data/Dataset/Labels.txt)")
    parser.add_argument("--start_idx", type=int, default=2000,
                        help="Starting index for filenames (default: 2000)")
    parser.add_argument("--img_h",     type=int, default=32)
    parser.add_argument("--img_w",     type=int, default=128)
    args = parser.parse_args()

    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Coal-Hand-Luke.ttf")
    if not os.path.exists(font_path):
        print(f"ERROR: Coal-Hand-Luke.ttf not found in {os.path.dirname(font_path)}")
        return

    os.makedirs(args.output, exist_ok=True)

    print(f"Font            : Coal-Hand-Luke.ttf ✓")
    print(f"Words in list   : {len(AZ_WORDS)}")
    print(f"Generating {args.count} images → {args.output}")
    print(f"Labels file     → {args.labels}\n")

    new_labels = []
    for i in range(args.count):
        word = random.choice(AZ_WORDS)
        filename = f"img_{args.start_idx + i:05d}.png"
        filepath = os.path.join(args.output, filename)

        img = generate_word_image(word, font_path, args.img_h, args.img_w)
        img.save(filepath)
        new_labels.append(f"{filename} {word}")

        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{args.count} done...")

    with open(args.labels, "a", encoding="utf-8") as f:
        for entry in new_labels:
            f.write(entry + "\n")

    print(f"\nDone! Generated {args.count} images.")
    print(f"Labels appended to: {args.labels}")
    print(f"\nNow retrain with: python train.py")


if __name__ == "__main__":
    main()