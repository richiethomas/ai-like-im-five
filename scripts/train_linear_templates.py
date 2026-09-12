#!/usr/bin/env python3
"""Generate the CIFAR-10 linear-classifier weight-template montage used in
Part 5 (paper-1-part-5-giving-the-computer-a-scorecard-linear-classifiers).

A multinomial logistic regression is a linear classifier: one weight per input
value (32x32x3 = 3,072 numbers) per class. After training, reshaping a class's
3,072 weights back into a 32x32x3 image shows the "template" that class has
learned. The templates come out blurry (each is an average over thousands of
photos), which is the point Part 5 makes.

Requirements: numpy, scikit-learn, pillow. And CIFAR-10 (Python version) from
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz, extracted so that
--data points at the cifar-10-batches-py directory.

Run:
    python scripts/train_linear_templates.py \
        --data /path/to/cifar-10-batches-py \
        --out public/experiments/linear-classifiers/cifar10-linear-weights.png
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression


def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"], np.array(d[b"labels"])


def weights_to_image(w):
    """Reshape 3,072 weights to 32x32x3 and normalize to 0-255 for display."""
    img = w.reshape(3, 32, 32).transpose(1, 2, 0)  # CIFAR layout is 3 x 32 x 32
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo)
    return (img * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cifar-10-batches-py directory")
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args()

    data = Path(args.data)
    X = np.concatenate([load_batch(data / f"data_batch_{i}")[0] for i in range(1, 6)])
    y = np.concatenate([load_batch(data / f"data_batch_{i}")[1] for i in range(1, 6)])
    X = X.astype(np.float64) / 255.0

    meta = pickle.load(open(data / "batches.meta", "rb"), encoding="bytes")
    names = [n.decode() for n in meta[b"label_names"]]

    # Light regularization (small C) keeps the templates clean rather than noisy.
    clf = LogisticRegression(max_iter=200, C=0.01, tol=1e-3)
    clf.fit(X, y)
    print("train accuracy:", round(clf.score(X, y), 3))
    W = clf.coef_  # (10, 3072)

    SCALE, PAD, LABEL_H, COLS, ROWS = 6, 8, 22, 5, 2
    TILE = 32 * SCALE
    cell_w, cell_h = TILE + PAD, TILE + PAD + LABEL_H
    montage = Image.new("RGB", (COLS * cell_w + PAD, ROWS * cell_h + PAD),
                        (255, 255, 255))
    draw = ImageDraw.Draw(montage)
    for idx in range(10):
        tile = Image.fromarray(weights_to_image(W[idx])).resize(
            (TILE, TILE), Image.NEAREST)
        r, c = divmod(idx, COLS)
        x, ytop = PAD + c * cell_w, PAD + r * cell_h
        montage.paste(tile, (x, ytop))
        draw.text((x + 2, ytop + TILE + 4), names[idx], fill=(30, 30, 30))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out)
    print("saved:", out, montage.size)


if __name__ == "__main__":
    main()
