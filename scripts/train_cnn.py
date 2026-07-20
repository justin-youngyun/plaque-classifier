#!/usr/bin/env python3
"""
Train the binary plaque / not-plaque ResNet50.

This is the not-plaque filter that runs before morphology subtyping. It needs
PyTorch (pip install torch torchvision). Everything else in this repo runs
without it, so this script is separate.

Reads the same labeled tiles as train.py: labels.csv with columns tile, label
(and optionally image for honest splits). Any label in the not-plaque alias set
becomes class 0, every other label becomes class 1.

Run:
    python scripts/train_cnn.py --data data --out models/binary_cnn_resnet50.pth --epochs 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data', help='folder with tiles/ and labels.csv')
    ap.add_argument('--out', default='models/binary_cnn_resnet50.pth')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from plaque_classifier.cnn import train_binary_cnn

    tiles_dir = os.path.join(args.data, 'tiles')
    labels_csv = os.path.join(args.data, 'labels.csv')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    train_binary_cnn(tiles_dir, labels_csv, epochs=args.epochs, batch=args.batch,
                     lr=args.lr, seed=args.seed, out_path=args.out)


if __name__ == '__main__':
    main()
