#!/usr/bin/env python3
"""
Engineered morphology features for amyloid plaque tiles.

Each tile is a grayscale image of a single segmented object, intensity scaled
to 0-1, centered so the object centroid sits at the tile center. compute_features
returns N_FEATURES (47) numbers describing intensity, texture, the radial
intensity profile, object shape, and a handful of compound features I added to
give a linear classifier the core-vs-halo signal it cannot learn on its own.

The three morphology classes these features target:
  Diffuse   - fuzzy, low contrast, no defined bright core, irregular boundary
  DenseCore - a bright compact core sitting inside a diffuse halo
  Compact   - small, tight, uniformly bright, sharp round boundary
"""

import numpy as np
from scipy import stats as sp_stats

# 47 features: 37 morphology/intensity/radial/texture + 6 DenseCore-targeted
# compound features + 4 features I kept after a forward-selection / ANOVA / MI
# sweep (see scripts/evaluate_features.py).
N_FEATURES = 47
FEATURE_NAMES = [
    'mean', 'std', 'p2', 'p50', 'p98', 'skewness', 'kurtosis', 'cv', 'intensity_range',
    'peak_to_mean', 'core_halo_ratio', 'entropy_mean', 'entropy_std',
    'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity',
    'radial_auc', 'abs_radial_slope', 'peak_position', 'fwhm', 'cum_r50', 'cum_r80',
    'circularity', 'solidity', 'eccentricity', 'extent', 'aspect_ratio', 'boundary_roughness',
    'gradient_mean', 'gradient_std', 'gradient_boundary',
    'object_area_frac', 'core_sharpness', 'n_peaks',
    'annularity', 'fractal_dimension',
    # DenseCore-targeted compound features
    'core_peak_compactness',   # circularity of the brightest 20% pixel cluster
    'radial_bimodality',       # depth of the valley between core peak and edge
    'core_halo_gradient',      # max radial gradient at the core to halo transition
    'edge_sharpness_asymmetry',  # inner-edge gradient / outer-edge gradient
    'dc_signature',            # interaction: core_sharpness * annularity
    'mature_signature',        # interaction: peak_to_mean * (1 - object_area_frac)
    # Kept after feature selection (top ANOVA / mutual-information winners)
    'radial_skew',             # skewness of the radial intensity profile
    'radial_kurtosis',         # kurtosis of the radial intensity profile
    'glcm_contrast_d3',        # long-range (distance 3) texture contrast
    'centroid_intensity_norm',  # brightness of the center pixel / tile median
]


def compute_features(img):
    """Compute the N_FEATURES morphology features on a full 0-1 grayscale tile.

    The radial profile is centered on the tile center, which is where the object
    centroid sits after tile extraction, so context (the halo) is preserved.
    """
    vals = img.ravel()
    t10 = np.mean(np.sort(vals)[-min(10, len(vals)):])
    thr_fg = max(1 / 65535, 0.01 * t10)
    M = vals >= thr_fg
    if not M.any():
        M = np.ones(len(vals), dtype=bool)
    fg = vals[M]
    if len(fg) == 0:
        return list(np.zeros(N_FEATURES))

    # Intensity statistics
    m1, s1 = np.mean(fg), np.std(fg)
    p2, p50, p98 = np.percentile(fg, [2, 50, 98])
    sk = sp_stats.skew(fg) if len(fg) > 2 else 0
    ku = sp_stats.kurtosis(fg) if len(fg) > 2 else 0
    cv = s1 / max(m1, 1e-10)
    ir = p98 - p2
    ptm = np.max(fg) / max(m1, 1e-10)
    ct = m1 + 2 * s1
    hi, lo = fg[fg > ct], fg[fg <= ct]
    chr_v = np.mean(hi) / max(np.mean(lo), 1e-10) if len(hi) > 0 and len(lo) > 0 else 1.0

    M2d = img >= thr_fg

    # Local entropy texture
    try:
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        I8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        E = entropy(I8, disk(4))
        En = (E - E.min()) / max(E.max() - E.min(), 1e-10)
        eM, eS = np.mean(En[M2d]), np.std(En[M2d])
    except Exception:
        eM, eS = 0, 0

    # Grey-level co-occurrence texture (distance 1)
    try:
        from skimage.feature import graycomatrix, graycoprops
        I8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        gl = graycomatrix(I8, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, True, True)
        gc, gcor = np.mean(graycoprops(gl, 'contrast')), np.mean(graycoprops(gl, 'correlation'))
        gen, gh = np.mean(graycoprops(gl, 'energy')), np.mean(graycoprops(gl, 'homogeneity'))
    except Exception:
        gc = gcor = gen = gh = 0

    # Radial intensity profile centered on the tile center
    H, W = img.shape
    yy, xx = np.mgrid[:H, :W]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = max(2, int(r[M2d].max()))
    edges = np.arange(0, rmax + 1)
    bins = np.digitize(r, edges) - 1
    prof = [np.mean(img[(bins == k) & M2d]) if ((bins == k) & M2d).any() else np.nan
            for k in range(len(edges) - 1)]
    prof = np.array(prof)
    pm = np.nanmax(prof)
    pn = prof / pm if pm > 0 else prof
    pn[np.isnan(pn)] = 0
    auc = np.nansum(pn)
    sl = np.nanmean(np.diff(pn[:min(10, len(pn))])) if len(pn) > 1 else 0
    if not np.isfinite(sl):
        sl = 0
    pp = np.nanargmax(pn) if not np.all(pn == 0) else 0
    va = np.where(pn >= 0.5)[0]
    fw = (va[-1] - va[0]) if len(va) >= 2 else 0
    tot = np.nansum(img[M2d])
    cf = [np.nansum(img[(r <= edges[k + 1]) & M2d]) / max(tot, 1e-10) for k in range(len(edges) - 1)]
    cf = np.array(cf)
    r50i = np.where(cf >= 0.5)[0]
    r50 = r50i[0] if len(r50i) else 0
    r80i = np.where(cf >= 0.8)[0]
    r80 = r80i[0] if len(r80i) else 0

    # Shape of the largest connected object under an adaptive threshold
    try:
        from skimage.filters import threshold_local
        from skimage.measure import regionprops, label as sk_label
        bw = (img > threshold_local(img, 51, method='mean', offset=-0.05)) & M2d
        lb = sk_label(bw)
        rps = regionprops(lb)
        if rps:
            lg = max(rps, key=lambda x: x.area)
            a, p = lg.area, lg.perimeter
            tc = 4 * np.pi * a / max(p ** 2, 1)
            ts = lg.solidity
            te = lg.eccentricity
            tx = lg.extent
            ta = lg.axis_major_length / max(lg.axis_minor_length, 1)
            br = p / max(np.pi * 2 * np.sqrt(a / np.pi), 1)
        else:
            tc = ts = te = tx = 0
            ta = 1
            br = 1
    except Exception:
        tc = ts = te = tx = 0
        ta = 1
        br = 1
        bw = M2d

    # Gradient magnitude across the object and at its boundary
    gy, gx = np.gradient(img)
    gm2 = np.sqrt(gx ** 2 + gy ** 2)
    gm, gs2 = np.mean(gm2[M2d]), np.std(gm2[M2d])
    try:
        from skimage.segmentation import find_boundaries
        bnd = find_boundaries(bw, mode='inner') if bw.any() else np.zeros_like(M2d)
        gb = np.mean(gm2[bnd]) if bnd.any() else 0
    except Exception:
        gb = 0

    oaf = np.sum(M2d) / M2d.size
    d2 = np.abs(np.diff(pn, n=2))
    cs = np.max(d2) if len(d2) > 0 else 0
    nk = sum(1 for j in range(1, len(pn) - 1)
             if pn[j] > pn[j - 1] and pn[j] > pn[j + 1] and pn[j] > 0.3)

    # Annularity: peripheral ring intensity / central core intensity.
    # High for DenseCore (bright corona around the dense amyloid core).
    rmax_annul = min(H, W) // 2
    r_inner = max(1, rmax_annul // 3)
    r_outer_lo = max(2, rmax_annul // 3)
    r_outer_hi = rmax_annul
    core_mask = r <= r_inner
    ring_mask = (r > r_outer_lo) & (r <= r_outer_hi)
    core_int = np.mean(img[core_mask & M2d]) if (core_mask & M2d).any() else 0
    ring_int = np.mean(img[ring_mask & M2d]) if (ring_mask & M2d).any() else 0
    annularity = ring_int / max(core_int, 1e-10)

    # Fractal dimension by box counting on the thresholded foreground.
    # High for Diffuse (irregular, fuzzy boundary), low for Compact (smooth, round).
    try:
        bw_frac = M2d.astype(np.uint8)
        sizes = [2, 4, 8, 16, 32]
        counts = []
        for s in sizes:
            n_boxes = 0
            for i_r in range(0, bw_frac.shape[0], s):
                for i_c in range(0, bw_frac.shape[1], s):
                    if bw_frac[i_r:i_r + s, i_c:i_c + s].any():
                        n_boxes += 1
            if n_boxes > 0:
                counts.append((np.log(1.0 / s), np.log(n_boxes)))
        if len(counts) >= 2:
            log_inv_s, log_n = zip(*counts)
            fractal_dim = np.polyfit(log_inv_s, log_n, 1)[0]
        else:
            fractal_dim = 1.5
    except Exception:
        fractal_dim = 1.5

    # ---- DenseCore-specific compound features ----
    # The features above capture general morphology, but a linear classifier
    # cannot learn interactions. DenseCore's signature is a bright compact core
    # plus a diffuse halo, so I compute that interaction explicitly.

    # core_peak_compactness: circularity of the brightest 20% pixel cluster.
    try:
        pk_thr = np.percentile(img[M2d], 80) if M2d.any() else 0
        pk_mask = (img >= pk_thr) & M2d
        if pk_mask.any():
            from skimage.measure import regionprops as _rp, label as _sk_label
            lb_pk = _sk_label(pk_mask)
            rps_pk = _rp(lb_pk)
            if rps_pk:
                lg_pk = max(rps_pk, key=lambda p: p.area)
                a_pk, p_pk = lg_pk.area, lg_pk.perimeter
                core_peak_compactness = 4 * np.pi * a_pk / max(p_pk ** 2, 1)
            else:
                core_peak_compactness = 0
        else:
            core_peak_compactness = 0
    except Exception:
        core_peak_compactness = 0

    # radial_bimodality: depth of the valley between the core peak and the edge.
    # DenseCore's core-halo structure creates a shoulder; Diffuse and Compact
    # profiles are monotonic.
    try:
        pn_smooth = pn.copy()
        if len(pn_smooth) >= 5:
            kernel = np.ones(3) / 3.0
            pn_smooth = np.convolve(pn_smooth, kernel, mode='same')
        peak_idx = int(np.nanargmax(pn_smooth)) if len(pn_smooth) > 0 else 0
        if peak_idx < len(pn_smooth) - 2 and pn_smooth[peak_idx] > 0:
            tail = pn_smooth[peak_idx + 1:]
            tail_min = np.nanmin(tail) if len(tail) > 0 else pn_smooth[peak_idx]
            radial_bimodality = (pn_smooth[peak_idx] - tail_min) / max(pn_smooth[peak_idx], 1e-10)
            if len(tail) >= 2:
                post_valley = tail[np.nanargmin(tail):]
                if len(post_valley) >= 2 and np.nanmean(post_valley[1:]) > tail_min:
                    pass  # true bimodal signature, keep the value
                else:
                    radial_bimodality *= 0.5  # monotonic, weaker signal
        else:
            radial_bimodality = 0
    except Exception:
        radial_bimodality = 0

    # core_halo_gradient: max radial gradient at the core to halo transition
    # (inner third to middle third of the radius). DenseCore drops sharply here.
    try:
        if len(pn) >= 6:
            inner_third = len(pn) // 3
            mid_third = 2 * len(pn) // 3
            if mid_third > inner_third + 1:
                transition_region = pn[inner_third:mid_third]
                core_halo_gradient = float(np.max(np.abs(np.diff(transition_region))) if len(transition_region) > 1 else 0)
            else:
                core_halo_gradient = 0
        else:
            core_halo_gradient = 0
    except Exception:
        core_halo_gradient = 0

    # edge_sharpness_asymmetry: inner-edge gradient / outer-edge gradient.
    # DenseCore has a sharp inner (core) edge and a fuzzy outer (halo) edge, so
    # the ratio is high; Compact and Diffuse are roughly symmetric (ratio ~1).
    try:
        if len(pn) >= 6:
            inner_q = max(1, len(pn) // 4)
            outer_q = max(inner_q + 1, 3 * len(pn) // 4)
            inner_grad = float(np.mean(np.abs(np.diff(pn[:inner_q * 2]))) if inner_q * 2 > 1 else 0)
            outer_grad = float(np.mean(np.abs(np.diff(pn[outer_q:]))) if len(pn) - outer_q > 1 else 1e-6)
            edge_sharpness_asymmetry = inner_grad / max(outer_grad, 1e-6)
        else:
            edge_sharpness_asymmetry = 1.0
    except Exception:
        edge_sharpness_asymmetry = 1.0

    # Interaction terms the linear classifier cannot form itself.
    dc_signature = cs * annularity                 # tight core AND diffuse halo
    mature_signature = ptm * (1.0 - oaf)           # bright peak AND compact fill

    # ---- Features kept after the feature-selection sweep ----

    # radial_skew / radial_kurtosis: shape of the radial intensity profile over
    # fixed bins spanning half the tile. Top ANOVA and mutual-information winners.
    try:
        rmax_rs = min(H, W) // 2
        n_bins_rs = 30
        r_edges = np.linspace(0, rmax_rs, n_bins_rs + 1)
        prof_rs = []
        for k_rs in range(n_bins_rs):
            mask_rs = (r >= r_edges[k_rs]) & (r < r_edges[k_rs + 1]) & M2d
            prof_rs.append(float(np.mean(img[mask_rs])) if mask_rs.any() else np.nan)
        prof_rs = np.array(prof_rs)
        prof_rs = prof_rs[np.isfinite(prof_rs)]
        if len(prof_rs) >= 3:
            mu_p = float(np.mean(prof_rs))
            sd_p = float(np.std(prof_rs))
            radial_skew = float(np.mean(((prof_rs - mu_p) / max(sd_p, 1e-10)) ** 3))
            radial_kurtosis = float(np.mean(((prof_rs - mu_p) / max(sd_p, 1e-10)) ** 4) - 3)
        else:
            radial_skew = 0.0
            radial_kurtosis = 0.0
    except Exception:
        radial_skew = 0.0
        radial_kurtosis = 0.0

    # glcm_contrast_d3: GLCM contrast at distance 3 (pattern spacing, e.g. the
    # granular spacing inside Diffuse plaques) vs the distance-1 baseline above.
    try:
        from skimage.feature import graycomatrix as _glcm3, graycoprops as _glcp3
        I8_d3 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        gl3 = _glcm3(I8_d3, [3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, True, True)
        glcm_contrast_d3 = float(np.mean(_glcp3(gl3, 'contrast')))
    except Exception:
        glcm_contrast_d3 = 0.0

    # centroid_intensity_norm: center pixel intensity / median tile intensity.
    # DenseCore has a bright center (high ratio); Diffuse has no defined center.
    try:
        cy_i = int(round((H - 1) / 2.0))
        cx_i = int(round((W - 1) / 2.0))
        med_fg = float(np.median(img[M2d])) if M2d.any() else 1e-10
        centroid_intensity_norm = float(img[cy_i, cx_i]) / max(med_fg, 1e-10)
    except Exception:
        centroid_intensity_norm = 1.0

    f = [m1, s1, p2, p50, p98, sk, ku, cv, ir, ptm, chr_v, eM, eS, gc, gcor, gen, gh,
         auc, abs(sl), pp, fw, r50, r80, tc, ts, te, tx, ta, br, gm, gs2, gb, oaf, cs, nk,
         annularity, fractal_dim,
         core_peak_compactness, radial_bimodality, core_halo_gradient,
         edge_sharpness_asymmetry, dc_signature, mature_signature,
         radial_skew, radial_kurtosis, glcm_contrast_d3, centroid_intensity_norm]
    return [0 if not np.isfinite(x) else x for x in f]


def compute_features_batch(tiles, log_every=100, logger=print):
    """Stack compute_features over a list of 0-1 grayscale tiles into an array."""
    out = np.zeros((len(tiles), N_FEATURES), dtype=np.float64)
    for i, t in enumerate(tiles):
        out[i] = compute_features(t)
        if log_every and (i + 1) % log_every == 0:
            logger(f"    features: {i + 1}/{len(tiles)}")
    return out
