#!/usr/bin/env python3
"""
Morphology classifier: Linear Discriminant Analysis on the engineered features.

I standardize the features and fit an LDA. The LDA does two jobs at once:
  1. It assigns one of the three classes (Diffuse, DenseCore, Compact).
  2. Its 2D transform gives a morphology continuum I read as LD1 = maturity
     (Diffuse to Compact) and LD2 = coredness (how DenseCore-like an object is).

I report honest performance with 10-fold cross validation using a Pipeline so the
scaler is refit inside every fold, plus the silhouette score of the 2D projection
to show the classes actually separate. The saved bundle is everything predict
needs: the fitted scaler, the fitted LDA, the class order, and the feature names.
"""

import pickle
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, silhouette_score

from .features import FEATURE_NAMES, N_FEATURES

PLAQUE_CLASSES = ['Diffuse', 'DenseCore', 'Compact']
# Maturity order used for the continuum reading: immature/fuzzy -> intermediate -> mature/tight.
ORDINAL_ORDER = ['Diffuse', 'DenseCore', 'Compact']


def train_lda(X, y, n_folds=10, logger=print):
    """Fit the scaler + LDA on feature matrix X and string labels y.

    Returns a bundle dict ready to pickle. X is (n_tiles, N_FEATURES), y holds
    class-name strings from PLAQUE_CLASSES.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    logger(f"  Feature matrix: {X.shape}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xs, y)

    # 2D projection: class means along the two discriminant axes.
    X2d = lda.transform(Xs)
    for cls in PLAQUE_CLASSES:
        mask = y == cls
        if mask.any():
            logger(f"    {cls:12s}: LD1={X2d[mask, 0].mean():+.3f} LD2={X2d[mask, 1].mean():+.3f} n={int(mask.sum())}")

    # Honest cross validation: the scaler is refit inside each fold via the pipeline.
    cv_pipe = Pipeline([('scaler', StandardScaler()),
                        ('clf', LinearDiscriminantAnalysis())])
    n_folds = min(n_folds, int(np.min([np.sum(y == c) for c in PLAQUE_CLASSES])))
    n_folds = max(2, n_folds)
    cv_pred = cross_val_predict(cv_pipe, X, y, cv=n_folds)

    try:
        sil = silhouette_score(X2d, y)
    except Exception:
        sil = float('nan')
    logger(f"\n  Silhouette (2D LDA projection): {sil:.4f}")
    logger(f"  {n_folds}-fold CV report:")
    logger(classification_report(y, cv_pred, target_names=PLAQUE_CLASSES, digits=3, zero_division=0))

    return {
        'lda': lda,
        'scaler': scaler,
        'class_names': PLAQUE_CLASSES,
        'feature_names': list(FEATURE_NAMES),
        'n_features': N_FEATURES,
        'silhouette': float(sil),
    }


def save_bundle(bundle, path):
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)


def load_bundle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def classify_features(feat_vec, bundle):
    """Classify one feature vector. Returns (predicted_class, proba_dict, (LD1, LD2)).

    Probabilities are keyed by the LDA's own class order (lda.classes_), which
    sklearn sorts alphabetically and which is not the same as PLAQUE_CLASSES, so
    the class name is taken from lda.predict rather than by indexing a fixed list.
    """
    lda = bundle['lda']
    scaler = bundle['scaler']
    x = np.asarray(feat_vec, dtype=np.float64).reshape(1, -1)
    xs = scaler.transform(x)
    proba = lda.predict_proba(xs)[0]
    ld = lda.transform(xs)[0]
    pred = str(lda.predict(xs)[0])
    proba_dict = {str(c): float(proba[i]) for i, c in enumerate(lda.classes_)}
    ld1 = float(ld[0])
    ld2 = float(ld[1]) if len(ld) > 1 else 0.0
    return pred, proba_dict, (ld1, ld2)

