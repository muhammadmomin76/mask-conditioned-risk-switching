#!/usr/bin/env python3
"""
H3 / D0  --  SSIM + IEF for every existing comparison, on the paper's exact grid,
plus a diagnostic re-run of MRS with the D(0)=0 convention in Eqn (6).

SPEC
----
Grid          : 6 images x 9 densities (0.1 .. 0.9) x 3 seeds (1,2,3)  = 162 runs
Noise         : numpy default_rng(seed); u<d marks corruption, v>=0.5 -> 255 else 0
Methods       : median3, median7, amf(S_max=7), nvbmf(global 45%),
                las2(local 5x5 45% OR global 45%), mrs(hybrid: V2<V1 OR global>45%),
                mrs_d0 (same but with D(0)=0 in the pairwise term), oracle
Structure fn  : GLOBAL estimator (Dcurve over 22 lags, isotropic bins in r^2)
Metrics       : PSNR (dB), SSIM_win (Wang 2004, 11x11 Gaussian sigma=1.5, mean of map),
                SSIM_glob (the whole-image form already in nvbmf.py), IEF, RIR (%)
Output        : h3_rows.json   (one record per image/density/seed)

Every filter output is clipped and rounded to uint8 before scoring, exactly as in
mrs_paper_data.py, so psnr_* here must reproduce the published table to 2 dp.
"""
import json, time
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np


from mrs_core import (stage1_with_sources, structure_function_global, isotropic,
                      Dcurve, LAGS, risk_stage1, risk_stage2, _box)
from mrs_run import load, add_spn, box3_nonzero_mean, psnr, IMAGES

DENS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [1, 2, 3]
CUTOFF = 0.45
AMF_SMAX = 7


# ----------------------------------------------------------------- baselines
def _stack(a, k):
    r = k // 2
    H, W = a.shape
    p = np.pad(a.astype(np.float64), r, mode='edge')
    return np.stack([p[i:i + H, j:j + W] for i in range(k) for j in range(k)], 0)


def median_filter(a, k):
    return np.sort(_stack(a, k), axis=0)[(k * k) // 2]


def adaptive_median(a, kmax=AMF_SMAX):
    x = a.astype(np.float64)
    out = x.copy()
    undecided = np.ones(x.shape, bool)
    for k in range(3, kmax + 1, 2):
        w = _stack(x, k)
        o = np.sort(w, axis=0)
        zmin, zmax, zmed = o[0], o[-1], np.median(w, 0)
        A = (zmed > zmin) & (zmed < zmax)
        take = undecided & A
        B = (x > zmin) & (x < zmax)
        out[take] = np.where(B, x, zmed)[take]
        undecided &= ~A
    if undecided.any():
        out[undecided] = np.median(_stack(x, kmax), 0)[undecided]
    return out


# ----------------------------------------------------------------- metrics
def _f(a):
    return np.asarray(a, np.float64)


def ssim_global(X, Y):
    """Whole-image SSIM, the form already used in nvbmf.py."""
    X, Y = _f(X), _f(Y)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = X.mean(), Y.mean()
    vx = ((X - mx) ** 2).mean()
    vy = ((Y - my) ** 2).mean()
    cxy = ((X - mx) * (Y - my)).mean()
    return float(((2 * mx * my + C1) * (2 * cxy + C2)) /
                 ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2)))


def _gauss_kernel(size=11, sigma=1.5):
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return g


_G = _gauss_kernel()


def _sep_filter(a, g):
    """Separable Gaussian, 'valid' support, array arithmetic only."""
    k = g.size
    H, W = a.shape
    t = np.zeros((H - k + 1, W), np.float64)
    for i in range(k):
        t += g[i] * a[i:i + H - k + 1, :]
    out = np.zeros((H - k + 1, W - k + 1), np.float64)
    for j in range(k):
        out += g[j] * t[:, j:j + W - k + 1]
    return out


def ssim_windowed(X, Y):
    """Wang et al. 2004 mean SSIM: 11x11 Gaussian sigma=1.5, valid windows."""
    X, Y = _f(X), _f(Y)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx = _sep_filter(X, _G)
    my = _sep_filter(Y, _G)
    mxx = _sep_filter(X * X, _G)
    myy = _sep_filter(Y * Y, _G)
    mxy = _sep_filter(X * Y, _G)
    vx = mxx - mx * mx
    vy = myy - my * my
    cxy = mxy - mx * my
    s = ((2 * mx * my + C1) * (2 * cxy + C2)) / \
        ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    return float(s.mean())


def ief(orig, noisy, restored):
    o, n, r = _f(orig), _f(noisy), _f(restored)
    den = ((r - o) ** 2).sum()
    return float('inf') if den == 0 else float(((n - o) ** 2).sum() / den)


def rir(a):
    a = np.clip(np.round(a), 0, 255)
    return float(((a == 0) | (a == 255)).mean()) * 100.0


# ----------------------------------------------------------------- D(0)=0 wrapper
class DZero:
    """Same isotropic curve, but D is exactly zero at lag zero.

    The published implementation interpolates with np.interp, which clamps below
    the smallest measured lag, so the i=j terms of the ordered double sum in
    Eqn (6) are charged D(1) instead of 0. This wrapper restores the convention
    that Eqn (6) states, so the two can be compared directly.
    """

    def __init__(self, base):
        self.base = base

    def __call__(self, r):
        r = np.asarray(r, np.float64)
        return np.where(r <= 0.0, 0.0, self.base(r))


# ----------------------------------------------------------------- one condition
def run_one(name, path, d, seed):
    img = load(path)
    noisy, mask = add_spn(img, d, seed)
    surv = ~mask
    gt = img.astype(np.float64)

    s1, sdi, sdj, rid, rm = stage1_with_sources(noisy, mask)
    s2 = box3_nonzero_mean(s1)
    gd = float(mask.mean())

    Dg = Dcurve(isotropic(structure_function_global(noisy.astype(np.float64), surv, LAGS)))
    V1 = risk_stage1(mask, rid, rm, Dg)
    V2 = risk_stage2(mask, sdi, sdj, Dg)

    D0 = DZero(Dg)
    V1z = risk_stage1(mask, rid, rm, D0)
    V2z = risk_stage2(mask, sdi, sdj, D0)

    e1 = (s1 - gt) ** 2
    e2 = (s2 - gt) ** 2
    orc = (e2 < e1)

    ld = _box(mask.astype(np.float64), 2) / _box(np.ones_like(s1), 2)

    outs = {
        'median3': median_filter(noisy, 3),
        'median7': median_filter(noisy, 7),
        'amf': adaptive_median(noisy, AMF_SMAX),
        'nvbmf': np.where(mask & (gd > CUTOFF), s2, s1),
        'las2': np.where(mask & ((ld > CUTOFF) | (gd > CUTOFF)), s2, s1),
        'mrs': np.where(mask & ((V2 < V1) | (gd > CUTOFF)), s2, s1),
        'mrs_d0': np.where(mask & ((V2z < V1z) | (gd > CUTOFF)), s2, s1),
        'oracle': np.where(mask & orc, s2, s1),
        'stage1': s1,
        'stage2': s2,
    }

    r = {'image': name, 'd': d, 'seed': seed}
    for k, v in outs.items():
        v8 = np.clip(np.round(v), 0, 255).astype(np.uint8)
        r['psnr_' + k] = psnr(img, v8)
        r['ssimw_' + k] = ssim_windowed(img, v8)
        r['ssimg_' + k] = ssim_global(img, v8)
        r['ief_' + k] = ief(img, noisy, v8)
        r['rir_' + k] = rir(v8)
    r['agree'] = float(((V2 < V1)[mask] == orc[mask]).mean())
    r['agree_d0'] = float(((V2z < V1z)[mask] == orc[mask]).mean())
    r['frac_mrs'] = float((mask & (V2 < V1)).sum()) / max(int(mask.sum()), 1)
    r['frac_mrs_d0'] = float((mask & (V2z < V1z)).sum()) / max(int(mask.sum()), 1)
    r['frac_oracle'] = float(orc[mask].mean())
    return r


if __name__ == '__main__':
    t0 = time.time()
    rows = []
    for name, path in IMAGES.items():
        for d in DENS:
            for s in SEEDS:
                rows.append(run_one(name, path, d, s))
                print(f"{name:20s} d={d:.1f} s={s} "
                      f"psnr_mrs={rows[-1]['psnr_mrs']:6.2f} "
                      f"psnr_mrs_d0={rows[-1]['psnr_mrs_d0']:6.2f} "
                      f"ssimw_mrs={rows[-1]['ssimw_mrs']:.4f} "
                      f"ief_mrs={rows[-1]['ief_mrs']:8.1f} "
                      f"[{time.time()-t0:6.1f}s]", flush=True)
                json.dump(rows, open(os.path.join(ROOT,'results','h3_rows.json'), 'w'), indent=1)
    print('elapsed', time.time() - t0)
