#!/usr/bin/env python3
"""
End to end prediction on a whole image.

Segment candidate objects, size-filter them, optionally pass each through the CNN
to drop non-plaques, compute the engineered features, and classify morphology
with the LDA. Returns one row per accepted plaque with its class, class
probabilities, the LD1 / LD2 continuum coordinates, size, and every raw feature.
"""

import os
import numpy as np
import pandas as pd

from .features import compute_features
from .segmentation import (read_image, segment_candidates, extract_tile,
                           render_overlay, DEFAULT_PX_UM)
from .classifier import classify_features


def process_image(img_path, bundle, px_um=DEFAULT_PX_UM, cnn=None, cnn_n_classes=None,
                  device='cpu', min_diam_um=10, max_diam_um=200):
    """Classify every plaque in one image.

    Pass cnn (a loaded model) to enable the not-plaque filter; leave it None to
    run segmentation + features + LDA only. Returns (dataframe, image, labeled_mask).
    """
    from skimage.measure import regionprops, label as sk_label

    I = read_image(img_path)
    img_name = os.path.basename(img_path)

    bw = segment_candidates(I, px_um)
    if not bw.any():
        return pd.DataFrame(), I, np.zeros(I.shape[:2], dtype=int)

    labeled = sk_label(bw)
    props = regionprops(labeled, intensity_image=I)
    feature_names = bundle['feature_names']

    results = []
    for prop in props:
        area_um2 = prop.area * px_um ** 2
        eqd = 2 * np.sqrt(area_um2 / np.pi)
        if eqd < min_diam_um or eqd > max_diam_um:
            continue
        if prop.solidity < 0.20 or prop.eccentricity > 0.98:
            continue
        circ = 4 * np.pi * prop.area / max(prop.perimeter ** 2, 1)
        if circ < 0.15:
            continue

        cy, cx = prop.centroid
        tile = extract_tile(I, cy, cx, px_um)

        if cnn is not None:
            from .cnn import cnn_is_plaque
            if not cnn_is_plaque(cnn, tile, cnn_n_classes, device):
                continue

        feats = compute_features(tile)
        pred, proba, (ld1, ld2) = classify_features(feats, bundle)

        row = {
            'Image': img_name,
            'ObjID': int(prop.label),
            'Cx_px': float(cx), 'Cy_px': float(cy),
            'EquivDiam_um': float(eqd),
            'Area_um2': float(area_um2),
            'PredictedClass': pred,
            'LD1_maturity': ld1,
            'LD2_coredness': ld2,
        }
        for cls, p in proba.items():
            row[f'P_{cls}'] = p
        for fi, fn in enumerate(feature_names):
            row[fn] = feats[fi]
        results.append(row)

    return pd.DataFrame(results), I, labeled


def predict_directory(input_dir, bundle, out_dir, px_um=DEFAULT_PX_UM, cnn=None,
                      cnn_n_classes=None, device='cpu', logger=print):
    """Run process_image over every .tif / .tiff / .png in input_dir.

    Writes per-object predictions, a per-image summary, and an overlay per image.
    Returns the combined predictions dataframe.
    """
    from pathlib import Path
    overlay_dir = os.path.join(out_dir, 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)

    paths = sorted(list(Path(input_dir).glob('*.tif')) +
                   list(Path(input_dir).glob('*.tiff')) +
                   list(Path(input_dir).glob('*.png')))
    logger(f"  Images: {len(paths)}")

    all_dfs = []
    for i, p in enumerate(paths):
        df, I, labeled = process_image(str(p), bundle, px_um, cnn, cnn_n_classes, device)
        if len(df) > 0:
            all_dfs.append(df)
            render_overlay(I, df, labeled, os.path.join(overlay_dir, f"{p.stem}_overlay.png"))
            counts = df['PredictedClass'].value_counts()
            logger(f"  [{i + 1}/{len(paths)}] {p.name}: "
                   f"Diffuse={counts.get('Diffuse', 0)} "
                   f"DenseCore={counts.get('DenseCore', 0)} "
                   f"Compact={counts.get('Compact', 0)}")
        else:
            logger(f"  [{i + 1}/{len(paths)}] {p.name}: no plaques")

    if not all_dfs:
        logger("  No plaques found.")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    pred_path = os.path.join(out_dir, 'predictions.csv')
    result.to_csv(pred_path, index=False)

    summary = []
    for img in result['Image'].unique():
        sub = result[result['Image'] == img]
        summary.append({
            'Image': img,
            'Total_plaques': len(sub),
            'N_Diffuse': int((sub['PredictedClass'] == 'Diffuse').sum()),
            'N_DenseCore': int((sub['PredictedClass'] == 'DenseCore').sum()),
            'N_Compact': int((sub['PredictedClass'] == 'Compact').sum()),
            'Mean_LD1': float(sub['LD1_maturity'].mean()),
            'Mean_LD2': float(sub['LD2_coredness'].mean()),
            'Total_area_um2': float(sub['Area_um2'].sum()),
        })
    summary_path = os.path.join(out_dir, 'image_summary.csv')
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    logger(f"\n  Predictions: {pred_path} ({len(result)} plaques)")
    logger(f"  Summary:     {summary_path}")
    logger(f"  Overlays:    {overlay_dir}")
    return result
