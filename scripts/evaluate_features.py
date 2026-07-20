#!/usr/bin/env python3
"""
Rank and select features for the LDA.

This is the methodology I used to decide which engineered features to keep. It
computes every feature on the labeled tiles, then:
  1. Ranks features by ANOVA F-statistic and by mutual information.
  2. Runs greedy forward selection under image-level k-fold cross validation,
     adding the feature that most improves macro F1 until it stalls.

Cross validation is grouped by source image so tiles from one image never sit in
both the train and test fold, which is what keeps the reported number honest.

Run:
    python scripts/evaluate_features.py --data data --n_folds 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plaque_classifier.features import compute_features_batch, FEATURE_NAMES
from plaque_classifier.classifier import PLAQUE_CLASSES


def load_tile(path):
    return np.array(Image.open(path).convert('L')).astype(np.float64) / 255.0


def image_fold_groups(images, n_folds, seed=42):
    """Deterministic image-level fold assignment (same seed -> same partition)."""
    uniq = sorted(set(images))
    rng = np.random.RandomState(seed)
    idx = np.arange(len(uniq))
    rng.shuffle(idx)
    fold_of = {uniq[idx[i]]: i % n_folds for i in range(len(uniq))}
    return np.array([fold_of[im] for im in images])


def evaluate_subset(X, y, fold_groups, indices, n_folds):
    """Image-level k-fold CV macro F1 for an LDA on the chosen feature columns."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    f1s = []
    for fold in range(n_folds):
        tr = fold_groups != fold
        te = fold_groups == fold
        if not tr.any() or not te.any():
            continue
        Xtr, Xte = X[tr][:, indices], X[te][:, indices]
        sc = StandardScaler().fit(Xtr)
        try:
            lda = LinearDiscriminantAnalysis().fit(sc.transform(Xtr), y[tr])
            preds = lda.predict(sc.transform(Xte))
            f1s.append(f1_score(y[te], preds, average='macro', zero_division=0))
        except Exception:
            pass
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--n_folds', type=int, default=10)
    ap.add_argument('--max_forward', type=int, default=25)
    args = ap.parse_args()

    tiles_dir = os.path.join(args.data, 'tiles')
    df = pd.read_csv(os.path.join(args.data, 'labels.csv'))
    df = df[df['label'].isin(PLAQUE_CLASSES)].reset_index(drop=True)
    print(f"Plaque tiles: {len(df)}")

    tiles = [load_tile(os.path.join(tiles_dir, fn)) for fn in df['tile']]
    X = compute_features_batch(tiles)
    y = df['label'].values
    images = df['image'].values if 'image' in df.columns else df['tile'].values

    n_folds = min(args.n_folds, int(min((y == c).sum() for c in PLAQUE_CLASSES)))
    n_folds = max(2, n_folds)
    fold_groups = image_fold_groups(images, n_folds)

    # ---- Baseline: all features ----
    all_idx = list(range(X.shape[1]))
    base_f1 = evaluate_subset(X, y, fold_groups, all_idx, n_folds)
    print(f"\nAll {X.shape[1]} features: macro F1 ({n_folds}-fold image-level CV) = {base_f1:.4f}")

    # ---- Univariate rankings ----
    from sklearn.feature_selection import f_classif, mutual_info_classif
    f_stats, f_pvals = f_classif(X, y)
    mi = mutual_info_classif(X, y, random_state=42)
    print("\nTop 10 by ANOVA F-statistic:")
    for i in np.argsort(-f_stats)[:10]:
        print(f"    {FEATURE_NAMES[i]:26s} F={f_stats[i]:8.2f}  p={f_pvals[i]:.2e}")
    print("\nTop 10 by mutual information:")
    for i in np.argsort(-mi)[:10]:
        print(f"    {FEATURE_NAMES[i]:26s} MI={mi[i]:.4f}")

    # ---- Forward selection ----
    print(f"\nForward selection (greedy add, max {args.max_forward}):")
    selected, remaining, best_f1 = [], list(range(X.shape[1])), 0.0
    stall = 0
    while len(selected) < args.max_forward and remaining:
        best_new, best_new_f1 = None, best_f1
        for idx in remaining:
            f1 = evaluate_subset(X, y, fold_groups, selected + [idx], n_folds)
            if f1 > best_new_f1:
                best_new_f1, best_new = f1, idx
        if best_new is None:
            print("    (no single addition improves, stopping)")
            break
        delta = best_new_f1 - best_f1
        selected.append(best_new)
        remaining.remove(best_new)
        print(f"    + {FEATURE_NAMES[best_new]:26s} F1={best_new_f1:.4f} (delta {delta:+.4f})")
        best_f1 = best_new_f1
        stall = stall + 1 if delta < 0.001 else 0
        if stall >= 3:
            print("    (stalled, stopping)")
            break

    print(f"\nForward-selected subset ({len(selected)} features): F1={best_f1:.4f}")
    print(f"Features: {[FEATURE_NAMES[i] for i in selected]}")

    out_csv = os.path.join(args.data, 'feature_eval_results.csv')
    pd.DataFrame({
        'feature': FEATURE_NAMES,
        'anova_F': f_stats,
        'anova_p': f_pvals,
        'mutual_info': mi,
        'forward_selected': [i in selected for i in range(len(FEATURE_NAMES))],
    }).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
