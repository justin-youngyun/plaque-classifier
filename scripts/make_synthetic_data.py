#!/usr/bin/env python3
"""
Generate synthetic training tiles and whole images so the pipeline can run
without any real microscopy.

Writes, under --out (default ./data):
  tiles/                 PNG tiles, one object each
  labels.csv             columns: tile, label, image
  images/                whole .tif images with several plaques each
  images_truth.csv       ground-truth object locations and classes

Run:
    python scripts/make_synthetic_data.py --out data --n_per_class 150 --n_images 6
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plaque_classifier.synthetic import make_tile, make_image, CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data')
    ap.add_argument('--n_per_class', type=int, default=150,
                    help='training tiles per plaque class')
    ap.add_argument('--n_notplaque', type=int, default=150,
                    help='not-plaque tiles (used only by the CNN stage)')
    ap.add_argument('--n_images', type=int, default=6, help='whole images to synthesize')
    ap.add_argument('--img_size', type=int, default=1024)
    ap.add_argument('--px_um', type=float, default=0.3788)
    ap.add_argument('--tiles_per_image', type=int, default=30,
                    help='pseudo source-image grouping so CNN splits stay honest')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tiles_dir = os.path.join(args.out, 'tiles')
    images_dir = os.path.join(args.out, 'images')
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # ---- Training tiles ----
    rows = []
    class_plan = [(c, args.n_per_class) for c in CLASSES] + [('NotPlaque', args.n_notplaque)]
    idx = 0
    for cls, n in class_plan:
        for _ in range(n):
            tile = make_tile(cls, rng, px_um=args.px_um)
            fname = f"{cls}_{idx:05d}.png"
            Image.fromarray((tile * 255).astype(np.uint8)).save(os.path.join(tiles_dir, fname))
            rows.append({'tile': fname, 'label': cls,
                         'image': f"synthetic_{idx // args.tiles_per_image:03d}"})
            idx += 1
    labels_df = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    labels_path = os.path.join(args.out, 'labels.csv')
    labels_df.to_csv(labels_path, index=False)
    print(f"Tiles:  {idx} written to {tiles_dir}")
    for cls, n in class_plan:
        print(f"    {cls:11s}: {n}")
    print(f"Labels: {labels_path}")

    # ---- Whole images ----
    import tifffile
    truth_rows = []
    for i in range(args.n_images):
        img, truth = make_image(rng, size=args.img_size, px_um=args.px_um)
        name = f"synthetic_slide_{i:03d}.tif"
        tifffile.imwrite(os.path.join(images_dir, name), img)
        for t in truth:
            t['Image'] = name
            truth_rows.append(t)
    truth_path = os.path.join(args.out, 'images_truth.csv')
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    print(f"\nImages: {args.n_images} written to {images_dir}")
    print(f"Truth:  {truth_path} ({len(truth_rows)} objects)")


if __name__ == '__main__':
    main()
