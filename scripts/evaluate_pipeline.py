#!/usr/bin/env python3
"""
Score the whole pipeline against ground truth on the synthetic images.

The tile-level cross validation in train.py measures the classifier in isolation.
This measures the whole thing end to end, segmentation and feature extraction and
classification together, which is the number that actually matters. It runs
prediction on the synthetic images, matches each predicted object to the nearest
ground-truth object by centroid, and reports detection recall plus the
morphology confusion matrix on the matched objects.

Run:
    python scripts/evaluate_pipeline.py --images data/images --model models/morphology_lda.pkl \
        --truth data/images_truth.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plaque_classifier.classifier import load_bundle, PLAQUE_CLASSES
from plaque_classifier.pipeline import predict_directory
from plaque_classifier.segmentation import DEFAULT_PX_UM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='data/images')
    ap.add_argument('--model', default='models/morphology_lda.pkl')
    ap.add_argument('--truth', default='data/images_truth.csv')
    ap.add_argument('--out', default='predictions')
    ap.add_argument('--px_um', type=float, default=DEFAULT_PX_UM)
    ap.add_argument('--match_um', type=float, default=12.0,
                    help='max centroid distance to call a prediction the same object')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bundle = load_bundle(args.model)
    pred = predict_directory(args.images, bundle, args.out, args.px_um)
    truth = pd.read_csv(args.truth)

    # Match each ground-truth object to the nearest prediction in the same image.
    matched = []
    n_detected = 0
    for img in truth['Image'].unique():
        t = truth[truth['Image'] == img]
        p = pred[pred['Image'] == img] if len(pred) else pred
        for _, tr in t.iterrows():
            if len(p) == 0:
                continue
            d = np.hypot(p['Cx_px'] - tr['Cx_px'], p['Cy_px'] - tr['Cy_px']) * args.px_um
            j = d.idxmin()
            if d.loc[j] < args.match_um:
                n_detected += 1
                matched.append((tr['TrueClass'], p.loc[j, 'PredictedClass']))

    print("\n=== END TO END EVALUATION ===")
    print(f"Ground-truth objects: {len(truth)}")
    print(f"Detected (recall):    {n_detected} ({100 * n_detected / max(len(truth), 1):.1f}%)")

    if not matched:
        print("No matched objects to score.")
        return
    m = pd.DataFrame(matched, columns=['true', 'pred'])
    conf = pd.crosstab(m['true'], m['pred']).reindex(index=PLAQUE_CLASSES, columns=PLAQUE_CLASSES, fill_value=0)
    print("\nMorphology confusion (rows true, columns predicted):")
    print(conf)
    acc = float((m['true'] == m['pred']).mean())
    print(f"\nMorphology accuracy on matched objects: {acc:.3f}")


if __name__ == '__main__':
    main()
