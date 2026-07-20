#!/usr/bin/env python3
"""
Predict plaque morphology on whole images.

Segments candidate objects, optionally filters them with the CNN, computes the
engineered features, and classifies each object with the trained LDA. Writes a
per-object CSV, a per-image summary, an overlay per image, and a scatter of the
LD1 / LD2 morphology continuum.

Run (feature + LDA only, no PyTorch needed):
    python scripts/predict.py --images data/images --model models/morphology_lda.pkl --out predictions

Add the CNN not-plaque filter (needs PyTorch and a trained checkpoint):
    python scripts/predict.py --images data/images --model models/morphology_lda.pkl \
        --out predictions --cnn models/binary_cnn_resnet50.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plaque_classifier.classifier import load_bundle, PLAQUE_CLASSES
from plaque_classifier.pipeline import predict_directory
from plaque_classifier.segmentation import DEFAULT_PX_UM


def make_continuum_plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'Diffuse': '#2ecc71', 'DenseCore': '#e74c3c', 'Compact': '#3498db'}
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in PLAQUE_CLASSES:
        sub = result[result['PredictedClass'] == cls]
        ax.scatter(sub['LD1_maturity'], sub['LD2_coredness'],
                   c=colors[cls], alpha=0.6, s=25, label=f'{cls} (n={len(sub)})')
    ax.set_xlabel('LD1: maturity axis  (Diffuse -> Compact)')
    ax.set_ylabel('LD2: coredness axis  (more DenseCore-like)')
    ax.set_title('Plaque morphology continuum')
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='data/images')
    ap.add_argument('--model', default='models/morphology_lda.pkl')
    ap.add_argument('--out', default='predictions')
    ap.add_argument('--px_um', type=float, default=DEFAULT_PX_UM)
    ap.add_argument('--cnn', default=None, help='optional trained binary CNN checkpoint')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bundle = load_bundle(args.model)
    print(f"Model: {args.model} ({bundle['n_features']} features)")

    cnn, cnn_n_classes, device = None, None, 'cpu'
    if args.cnn:
        from plaque_classifier.cnn import load_cnn
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cnn, cnn_n_classes = load_cnn(args.cnn, device)
        print(f"CNN:   {args.cnn} ({cnn_n_classes} classes) on {device}")
    else:
        print("CNN:   disabled (segmentation + features + LDA only)")

    result = predict_directory(args.images, bundle, args.out, args.px_um,
                               cnn, cnn_n_classes, device)
    if len(result) == 0:
        return

    print("\n=== SUMMARY ===")
    print(f"Total plaques: {len(result)}")
    for cls in PLAQUE_CLASSES:
        n = int((result['PredictedClass'] == cls).sum())
        print(f"    {cls:11s}: {n:4d} ({100 * n / len(result):.1f}%)")

    plot_path = os.path.join(args.out, 'continuum.png')
    make_continuum_plot(result, plot_path)
    print(f"\nContinuum plot: {plot_path}")


if __name__ == '__main__':
    main()
