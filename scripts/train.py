#!/usr/bin/env python3
"""
Train the morphology classifier: engineered features -> LDA.

Reads the labeled tiles produced by make_synthetic_data.py (or your own tiles
plus a labels.csv with columns tile, label), keeps the three plaque classes,
computes the features, fits the scaler + LDA, reports cross-validated accuracy,
and saves the model bundle.

Run:
    python scripts/train.py --data data --out models/morphology_lda.pkl
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plaque_classifier.features import compute_features_batch
from plaque_classifier.classifier import train_lda, save_bundle, PLAQUE_CLASSES


def load_tile(path):
    return np.array(Image.open(path).convert('L')).astype(np.float64) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data', help='folder with tiles/ and labels.csv')
    ap.add_argument('--out', default='models/morphology_lda.pkl')
    ap.add_argument('--n_folds', type=int, default=10)
    args = ap.parse_args()

    tiles_dir = os.path.join(args.data, 'tiles')
    labels_csv = os.path.join(args.data, 'labels.csv')
    df = pd.read_csv(labels_csv)
    df = df[df['label'].isin(PLAQUE_CLASSES)].reset_index(drop=True)
    print(f"Plaque tiles: {len(df)}")
    for c in PLAQUE_CLASSES:
        print(f"    {c:11s}: {int((df['label'] == c).sum())}")

    print("\nExtracting features...")
    tiles = [load_tile(os.path.join(tiles_dir, fn)) for fn in df['tile']]
    X = compute_features_batch(tiles)
    y = df['label'].values

    print("\nTraining LDA...")
    bundle = train_lda(X, y, n_folds=args.n_folds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_bundle(bundle, args.out)
    print(f"\nSaved model bundle: {args.out}")
    print(f"  Silhouette (2D LDA): {bundle['silhouette']:.4f}")


if __name__ == '__main__':
    main()
