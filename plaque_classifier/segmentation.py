#!/usr/bin/env python3
"""
Candidate segmentation for whole slide images.

Full images hold many objects. Before classifying morphology I have to find the
candidate plaques and cut a fixed physical-size tile around each one. The
detector is a double band-pass filter (difference of Gaussians at two scales)
to reject slow tissue-level intensity variation, an adaptive local threshold,
morphological cleanup, and a watershed to split plaques that touch.

Physical scale matters, so every size is written in microns and converted with
px_um (microns per pixel). The default matches the confocal objective I used,
but pass your own with --px_um.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_fill_holes

DEFAULT_PX_UM = 0.3788


def read_image(path):
    """Read a .tif / .tiff (max-projecting a z-stack) or a normal image to 0-1 grayscale."""
    try:
        from tifffile import TiffFile
        with TiffFile(str(path)) as tif:
            pages = tif.pages
            if len(pages) > 1:
                I = np.stack([p.asarray() for p in pages]).astype(np.float32).max(axis=0)
            else:
                I = pages[0].asarray().astype(np.float32)
    except Exception:
        I = np.array(Image.open(path).convert('L')).astype(np.float32)
    if I.max() > 1:
        I = I / I.max()
    return I


def segment_candidates(I, px_um=DEFAULT_PX_UM, min_diam_um=5):
    """Return a boolean mask of candidate objects.

    Double band-pass (2-10 um and 5-30 um) locks onto sharp plaque-sized objects
    and rejects tissue-level intensity gradients. The distance transform plus
    watershed then splits merged objects at their narrowest points.
    """
    sig_s1 = max(1, round(2 / px_um))
    sig_l1 = max(sig_s1 + 1, round(10 / px_um))
    sig_s2 = max(1, round(5 / px_um))
    sig_l2 = max(sig_s2 + 1, round(30 / px_um))
    bp1 = gaussian_filter(I, sig_s1) - gaussian_filter(I, sig_l1)
    bp2 = gaussian_filter(I, sig_s2) - gaussian_filter(I, sig_l2)
    bp = np.maximum(bp1, bp2)
    bpn = bp - bp.min()
    if bpn.max() > 0:
        bpn = bpn / bpn.max()

    from skimage.filters import threshold_local
    from skimage.morphology import opening, closing, disk
    from skimage.measure import label as sk_label
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import distance_transform_edt

    ns = max(3, int(2 * round((30 / px_um) / 2) + 1))
    if ns % 2 == 0:
        ns += 1
    bw = bpn > threshold_local(bpn, ns, method='mean', offset=-0.10)
    # Drop objects below the minimum plaque area.
    min_area = int(np.pi * ((min_diam_um / 2) / px_um) ** 2)
    lab = sk_label(bw)
    if lab.max() > 0:
        areas = np.bincount(lab.ravel())
        keep = areas >= max(1, min_area)
        keep[0] = False
        bw = keep[lab]
    bw = opening(bw, disk(max(1, round(2 / px_um))))
    bw = closing(bw, disk(max(1, round(1 / px_um))))
    bw = binary_fill_holes(bw)

    # Watershed splits touching plaques. Distance transform finds the centers,
    # then the basins grow out from those seeds.
    dist = distance_transform_edt(bw)
    min_dist_px = max(5, int(min_diam_um / px_um))
    local_max = peak_local_max(dist, min_distance=min_dist_px, labels=bw)
    markers = np.zeros_like(bw, dtype=int)
    for i, (rr, cc) in enumerate(local_max):
        markers[rr, cc] = i + 1
    ws = watershed(-dist, markers, mask=bw)
    return ws > 0


def extract_tile(I, cy, cx, px_um=DEFAULT_PX_UM, tile_um=64, out_size=256):
    """Cut a fixed physical-size tile centered on (cy, cx) and resize to out_size."""
    tile_px = int(2 * round((tile_um / px_um) / 2))
    half = tile_px // 2
    padI = np.pad(I, half, mode='reflect')
    r1 = int(round(cy))
    c1 = int(round(cx))
    tile = padI[r1:r1 + tile_px, c1:c1 + tile_px]
    tile_out = np.array(Image.fromarray(
        (np.clip(tile, 0, 1) * 255).astype(np.uint8)
    ).resize((out_size, out_size))) / 255.0
    return tile_out


# Fixed display colors per class (RGB).
CLASS_COLORS = {'Diffuse': (0, 255, 0), 'DenseCore': (255, 0, 0), 'Compact': (0, 160, 255)}


def render_overlay(I, plaques_df, labeled_mask, save_path):
    """Draw the classified objects back onto the image as a fluorescence-style overlay."""
    from skimage.segmentation import find_boundaries
    from skimage.morphology import dilation, disk
    from PIL import ImageDraw

    p_lo = np.percentile(I, 1)
    p_hi = np.percentile(I, 99.5)
    I_adj = np.clip((I - p_lo) / max(p_hi - p_lo, 1e-10), 0, 1)
    base = (I_adj * 255).astype(np.uint8)
    rgb = np.zeros((*base.shape, 3), dtype=np.uint8)
    rgb[:, :, 1] = base  # green channel base

    for _, row in plaques_df.iterrows():
        cls = row['PredictedClass']
        obj_id = row['ObjID']
        if cls not in CLASS_COLORS:
            continue
        color = CLASS_COLORS[cls]
        mask = labeled_mask == obj_id
        if not mask.any():
            continue
        for c in range(3):
            rgb[:, :, c][mask] = np.clip(0.55 * rgb[:, :, c][mask] + 0.45 * color[c], 0, 255).astype(np.uint8)
        boundary = find_boundaries(mask, mode='outer')
        boundary_thick = dilation(boundary, disk(2))
        for c in range(3):
            rgb[:, :, c][boundary_thick] = color[c]

    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    for i, (label, color) in enumerate(CLASS_COLORS.items()):
        y_pos = 10 + i * 26
        draw.rectangle([10, y_pos, 180, y_pos + 22], fill=(0, 0, 0))
        draw.rectangle([14, y_pos + 4, 30, y_pos + 18], fill=color)
        draw.text((35, y_pos + 3), label, fill=(255, 255, 255))
    pil_img.save(save_path)
