#!/usr/bin/env python3
"""
Synthetic plaque generator.

This repository ships no real microscopy. To make the pipeline runnable I
generate synthetic amyloid-like objects whose morphology matches the three
classes the classifier targets, drawn on a faint tissue-like background:

  Diffuse   - large, low intensity, granular, irregular boundary, no core
  DenseCore - a diffuse halo with a small bright compact core in the middle
  Compact   - small, bright, near-uniform, sharp round edge

The point is not photorealism, it is that the same intensity, radial-profile,
texture, and shape structure the real features key on is present here, so the
feature extraction, the LDA, and the segmentation all run and report a real
(cross-validated) number on data that is entirely made up.

make_tile produces one centered 256 tile for training. make_image scatters
objects on a larger canvas with a ground-truth table for the prediction demo.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .segmentation import DEFAULT_PX_UM

CLASSES = ['Diffuse', 'DenseCore', 'Compact']

# Field of view of one tile, in microns. Training tiles and the tiles the
# pipeline crops around each detected object share this scale, so the features
# match between training and prediction. This is the crop size, not a plaque
# size. It matches the default tile_um in segmentation.extract_tile.
TILE_UM = 64.0

# Physical object radius per class, in microns (halo radius for DenseCore).
PHYS_RADIUS_UM = {
    'Diffuse': (14.0, 22.0),
    'DenseCore': (10.0, 18.0),
    'Compact': (6.0, 12.0),
    'NotPlaque': (8.0, 20.0),
}


def _tissue_background(shape, rng, level=0.0025, grain=0.0018):
    """Near-black background with faint low-frequency tissue texture.

    Kept well below plaque intensity on purpose: the features threshold the
    foreground relative to the brightest pixels, the same way real immunostain
    sits on dark tissue.
    """
    low = gaussian_filter(rng.standard_normal(shape), sigma=max(shape) / 12.0)
    low = (low - low.min()) / max(np.ptp(low), 1e-9)
    bg = level + grain * low + rng.normal(0, grain * 0.4, shape)
    return np.clip(bg, 0, None)


def _radial(shape, cy, cx):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2), yy, xx


def draw_plaque(canvas, cy, cx, cls, rng, radius, blend_prob=0.0):
    """Add one plaque of class cls to canvas in place, centered on (cy, cx).

    radius is the object radius in pixels (the halo radius for DenseCore). With
    probability blend_prob the object is nudged toward a neighboring class, which
    produces the genuinely ambiguous boundary cases a real dataset has.
    """
    shape = canvas.shape
    r, yy, xx = _radial(shape, cy, cx)
    theta = np.arctan2(yy - cy, xx - cx)

    if cls == 'Diffuse':
        # Irregular, lobed boundary and granular fill, no bright center.
        wobble = (1.0 + 0.16 * np.sin(3 * theta + rng.uniform(0, 2 * np.pi))
                  + 0.11 * np.sin(5 * theta + rng.uniform(0, 2 * np.pi)))
        r_eff = r / np.clip(wobble, 0.4, None)
        peak = rng.uniform(0.26, 0.50)
        body = peak * np.exp(-(r_eff / radius) ** 2 * 1.3)
        grain = gaussian_filter(rng.standard_normal(shape), sigma=1.4)
        grain = 1.0 + 0.60 * (grain - grain.mean()) / max(grain.std(), 1e-9)
        body = body * np.clip(grain, 0.2, 2.1)
        canvas += np.clip(body, 0, None)

    elif cls == 'DenseCore':
        # Diffuse halo plus a small, sharp, bright core. The ranges overlap the
        # other two classes on purpose (weak cores look diffuse, big bright cores
        # look compact) so the classifier faces realistic confusion.
        peak_halo = rng.uniform(0.20, 0.38)
        halo = peak_halo * np.exp(-(r / radius) ** 2)
        grain = gaussian_filter(rng.standard_normal(shape), sigma=1.6)
        grain = 1.0 + 0.35 * (grain - grain.mean()) / max(grain.std(), 1e-9)
        halo = halo * np.clip(grain, 0.3, 1.9)
        r_core = radius * rng.uniform(0.20, 0.40)
        peak_core = rng.uniform(0.55, 1.0)
        core = peak_core * np.exp(-(r / max(r_core, 1.0)) ** 2 * 1.5)
        canvas += np.clip(halo, 0, None) + core

    elif cls == 'Compact':
        # Flat-topped, sharp-edged, round, near-uniform bright disk. Softer edges
        # and dimmer disks at the low end overlap the DenseCore class.
        p = rng.uniform(3.0, 8.0)
        peak = rng.uniform(0.52, 0.95)
        disk = peak * np.exp(-(r / radius) ** p)
        disk += 0.02 * rng.standard_normal(shape) * (disk > 0.05)
        canvas += np.clip(disk, 0, None)

    elif cls == 'NotPlaque':
        # Distractors the CNN should reject: an elongated streak (vessel-like)
        # or a faint shapeless smudge.
        if rng.random() < 0.5:
            ang = rng.uniform(0, np.pi)
            xr = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
            yr = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
            streak = rng.uniform(0.2, 0.4) * np.exp(-(xr / (radius * 2.2)) ** 2
                                                    - (yr / max(radius * 0.28, 1.0)) ** 2)
            canvas += np.clip(streak, 0, None)
        else:
            smudge = rng.uniform(0.12, 0.22) * np.exp(-(r / (radius * 1.4)) ** 2)
            smudge *= 1.0 + 0.8 * gaussian_filter(rng.standard_normal(shape), 2.0)
            canvas += np.clip(smudge, 0, None)

    # Push a fraction of objects toward a neighboring class so the dataset has
    # real boundary cases and the classifier does not score a suspicious 100%.
    if blend_prob and cls in CLASSES and rng.random() < blend_prob:
        if cls == 'Diffuse':
            # A faint compact core makes a diffuse plaque read as borderline DenseCore.
            r_core = radius * rng.uniform(0.16, 0.26)
            canvas += rng.uniform(0.30, 0.52) * np.exp(-(r / max(r_core, 1.0)) ** 2 * 1.5)
        elif cls == 'DenseCore':
            # Fill in the core region so the halo fades and it reads as Compact.
            canvas += rng.uniform(0.24, 0.40) * np.exp(-(r / (radius * 0.55)) ** 4)
        elif cls == 'Compact':
            # A faint broad halo makes a compact plaque read as borderline DenseCore.
            canvas += rng.uniform(0.14, 0.24) * np.exp(-(r / (radius * 2.0)) ** 2)


def make_tile(cls, rng, out_size=256, px_um=DEFAULT_PX_UM, blend_prob=0.18):
    """One synthetic tile with the object centered (slight jitter), values 0-1.

    The object is drawn at the same native pixel scale a real object has in a
    whole image, inside a TILE_UM window, then resized to out_size the same way
    segmentation.extract_tile resizes a cropped object. Training tiles and the
    tiles the pipeline crops at prediction time therefore go through the identical
    crop-and-resize path, so their features line up.
    """
    crop_px = int(round(TILE_UM / px_um))
    canvas = _tissue_background((crop_px, crop_px), rng)
    jitter = crop_px * 0.04
    cy = (crop_px - 1) / 2 + rng.uniform(-jitter, jitter)
    cx = (crop_px - 1) / 2 + rng.uniform(-jitter, jitter)
    radius_px = rng.uniform(*PHYS_RADIUS_UM[cls]) / px_um
    draw_plaque(canvas, cy, cx, cls, rng, radius_px, blend_prob=blend_prob)
    canvas += rng.normal(0, 0.006, (crop_px, crop_px))
    canvas = np.clip(canvas, 0, 1)
    tile = np.array(Image.fromarray((canvas * 255).astype(np.uint8)).resize((out_size, out_size))) / 255.0
    return tile.astype(np.float32)


def make_image(rng, size=1024, px_um=0.3788, n_range=(10, 16), min_gap_um=40):
    """A larger canvas with several scattered plaques.

    Returns (image 0-1, list of ground-truth dicts with Cy_px, Cx_px, TrueClass,
    Radius_um). Object sizes are set in microns and converted with px_um so the
    physical-scale segmentation finds them.
    """
    canvas = _tissue_background((size, size), rng)
    n = rng.integers(n_range[0], n_range[1] + 1)
    min_gap_px = min_gap_um / px_um
    placed = []
    truth = []
    attempts = 0
    while len(placed) < n and attempts < n * 40:
        attempts += 1
        cls = CLASSES[rng.integers(0, len(CLASSES))]
        radius_um = rng.uniform(*PHYS_RADIUS_UM[cls])
        radius_px = radius_um / px_um
        margin = radius_px + 6
        cy = rng.uniform(margin, size - margin)
        cx = rng.uniform(margin, size - margin)
        if any(np.hypot(cy - py, cx - px) < min_gap_px for py, px in placed):
            continue
        draw_plaque(canvas, cy, cx, cls, rng, radius_px)
        placed.append((cy, cx))
        truth.append({'Cy_px': float(cy), 'Cx_px': float(cx),
                      'TrueClass': cls, 'Radius_um': float(radius_um)})
    canvas += rng.normal(0, 0.004, (size, size))
    return np.clip(canvas, 0, 1).astype(np.float32), truth
