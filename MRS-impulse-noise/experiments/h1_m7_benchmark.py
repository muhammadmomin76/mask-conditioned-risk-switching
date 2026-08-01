#!/usr/bin/env python3
"""
H1 + M7  --  MRS grafted onto NVBMF, benchmarked against the anchor paper's
comparison set on the anchor paper's datasets, with Wilcoxon signed-rank tests.

WHAT THIS ANSWERS
-----------------
H1 : the reviewer's objection that six images cannot support the claim, and that
     the anchor [1] is weakest in the 10-40% band, which is exactly the band MRS
     acts on. Running on BSDS200 + TESTIMAGES40 + MATLAB20 puts MRS on the anchor's
     own ground.
M7 : scale beyond six images, re-estimate the roughness-gain correlation on a set
     large enough to carry it, and test significance the way the anchor does.

DATASETS (you must supply these; see PREPARE below)
---------------------------------------------------
    data/bsds200/*.png        200 images, UC-Berkeley BSDS train split, grayscale
    data/testimages40/*.png    40 images, TESTIMAGES 1200x1200 subset, grayscale
    data/matlab20/*.png        20 images, the MATLAB sample-image set, grayscale

PREPARE
-------
Convert every source image to 8-bit grayscale PNG. Do NOT resize BSDS (481x321 is
the native size the anchor uses). Record the exact file list; the script writes it
into the output so the run is reproducible.

DENSITIES
---------
0.10 .. 0.40 in steps of 0.05 (the band where MRS is active), plus 0.50 and 0.60 as
a control band where MRS defers to the global rule and must be identical to NVBMF.

BASELINES
---------
NVBMF [1] and MRS are implemented here. ARmF, IAWMF, IMF, DAMF and SFT_lp are NOT
reimplemented: reimplementing five filters from paywalled papers would put unverified
constants into the comparison. Two honest options, in order of preference:
  (a) run the authors' released code (Erkan/Enginoglu publish MATLAB for DAMF, IMF,
      AFMF, ARmF, IAWMF) on the same corrupted images this script writes out, then
      paste the PSNR/SSIM tables back;
  (b) quote the anchor's published numbers for those filters and state in the caption
      that they are quoted and not re-measured. Only valid if the image set, the
      density grid and the noise generator match exactly.
Set DUMP_NOISY = True to write the corrupted images for option (a).

OUTPUT
------
    h1_rows.json     one record per image/density/seed
    h1_summary.json  per-dataset means, per-density means, Wilcoxon results,
                     roughness-gain correlation over the whole set
"""
import json, glob, time, math
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np


from mrs_core import (stage1_with_sources, structure_function_global, isotropic,
                      Dcurve, LAGS, risk_stage1, risk_stage2, _box)

from PIL import Image

# ----------------------------------------------------------------- configuration
DATASETS = {
    'bsds200': 'data/bsds200/*.png',
    'testimages40': 'data/testimages40/*.png',
    'matlab20': 'data/matlab20/*.png',
}
ACTIVE_DENS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CONTROL_DENS = [0.50, 0.60]
SEEDS = [1, 2, 3]
CUTOFF = 0.45
DUMP_NOISY = False
FIX_D0 = True          # see the D0 node: use the convention Eqn (6) states


# ----------------------------------------------------------------- helpers
def load(p):
    return np.array(Image.open(p).convert('L')).astype(np.uint8)


def add_spn(img, d, seed):
    """Identical generator to the six-image study: same rng, same draw order."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape)
    corrupt = r < d
    salt = rng.random(img.shape) < 0.5
    out[corrupt & salt] = 255
    out[corrupt & ~salt] = 0
    return out, corrupt


def box3_nonzero_mean(a):
    a = a.astype(np.float64)
    H, W = a.shape
    ap = np.pad(a, 1)
    zp = np.pad((a != 0.0).astype(np.float64), 1)
    ws = np.zeros_like(a)
    wc = np.zeros_like(a)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            ws += ap[di:di + H, dj:dj + W] * zp[di:di + H, dj:dj + W]
            wc += zp[di:di + H, dj:dj + W]
    out = a.copy()
    has = wc > 0
    out[has] = ws[has] / wc[has]
    return out


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def _gauss(size=11, sigma=1.5):
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


_G = _gauss()


def _sep(a, g):
    k = g.size
    H, W = a.shape
    t = np.zeros((H - k + 1, W), np.float64)
    for i in range(k):
        t += g[i] * a[i:i + H - k + 1, :]
    o = np.zeros((H - k + 1, W - k + 1), np.float64)
    for j in range(k):
        o += g[j] * t[:, j:j + W - k + 1]
    return o


def ssim_windowed(X, Y):
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _sep(X, _G), _sep(Y, _G)
    vx = _sep(X * X, _G) - mx * mx
    vy = _sep(Y * Y, _G) - my * my
    cxy = _sep(X * Y, _G) - mx * my
    return float((((2 * mx * my + C1) * (2 * cxy + C2)) /
                  ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))).mean())


def ief(o, n, r):
    o, n, r = (np.asarray(z, np.float64) for z in (o, n, r))
    den = ((r - o) ** 2).sum()
    return float('inf') if den == 0 else float(((n - o) ** 2).sum() / den)


class DZero:
    def __init__(self, base):
        self.base = base

    def __call__(self, r):
        r = np.asarray(r, np.float64)
        return np.where(r <= 0.0, 0.0, self.base(r))


# ----------------------------------------------------------------- one condition
def run_one(img, d, seed, dump_path=None):
    noisy, mask = add_spn(img, d, seed)
    if dump_path:
        Image.fromarray(noisy).save(dump_path)
    surv = ~mask
    gt = img.astype(np.float64)

    s1, sdi, sdj, rid, rm = stage1_with_sources(noisy, mask)
    s2 = box3_nonzero_mean(s1)
    gd = float(mask.mean())

    Dg = Dcurve(isotropic(structure_function_global(noisy.astype(np.float64), surv, LAGS)))
    Dfun = DZero(Dg) if FIX_D0 else Dg
    V1 = risk_stage1(mask, rid, rm, Dfun)
    V2 = risk_stage2(mask, sdi, sdj, Dfun)

    e1 = (s1 - gt) ** 2
    e2 = (s2 - gt) ** 2
    orc = e2 < e1
    ld = _box(mask.astype(np.float64), 2) / _box(np.ones_like(s1), 2)

    outs = {
        'nvbmf': np.where(mask & (gd > CUTOFF), s2, s1),
        'las2': np.where(mask & ((ld > CUTOFF) | (gd > CUTOFF)), s2, s1),
        'mrs': np.where(mask & ((V2 < V1) | (gd > CUTOFF)), s2, s1),
        'oracle': np.where(mask & orc, s2, s1),
    }
    r = {'d': d, 'seed': seed}
    for k, v in outs.items():
        v8 = np.clip(np.round(v), 0, 255).astype(np.uint8)
        r['psnr_' + k] = psnr(img, v8)
        r['ssim_' + k] = ssim_windowed(img, v8)
        r['ief_' + k] = ief(img, noisy, v8)
    r['agree'] = float(((V2 < V1)[mask] == orc[mask]).mean())
    # roughness at unit lag, normalised by variance, measured on the CLEAN image
    dh = isotropic(structure_function_global(gt, np.ones_like(mask, bool), [(0, 1), (1, 0)]))
    r['D1_over_var'] = float(dh[1] / max(gt.var(), 1e-9))
    return r


# ----------------------------------------------------------------- statistics
def wilcoxon_signed_rank(x, y):
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie and
    continuity correction. Written out so the result does not depend on a library
    version. Returns (W, z, p, n_used)."""
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    a = np.abs(d)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(n, np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    wp = ranks[d > 0].sum()
    wm = ranks[d < 0].sum()
    W = min(wp, wm)
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(a, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    z = (W - mu + 0.5) / sd if sd > 0 else 0.0
    p = 1.0 + math.erf(-abs(z) / math.sqrt(2.0))
    return float(W), float(z), float(p), int(n)


def spearman(x, y):
    def rk(v):
        v = np.asarray(v, np.float64)
        o = np.argsort(v, kind='mergesort')
        r = np.empty(v.size, np.float64)
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and v[o[j + 1]] == v[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    return pearson(rx, ry)


def pearson(x, y):
    x = np.asarray(x, np.float64) - np.mean(x)
    y = np.asarray(y, np.float64) - np.mean(y)
    return float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))


# ----------------------------------------------------------------- main
if __name__ == '__main__':
    t0 = time.time()
    rows = []
    manifest = {}
    for ds, pat in DATASETS.items():
        files = sorted(glob.glob(pat))
        manifest[ds] = [os.path.basename(f) for f in files]
        if not files:
            print(f"!! {ds}: no files matched {pat}", flush=True)
            continue
        for fp in files:
            img = load(fp)
            for d in ACTIVE_DENS + CONTROL_DENS:
                for s in SEEDS:
                    r = run_one(img, d, s)
                    r['dataset'] = ds
                    r['image'] = os.path.basename(fp)
                    rows.append(r)
            print(f"{ds}/{os.path.basename(fp):30s} [{time.time()-t0:7.1f}s]", flush=True)
            json.dump(rows, open('h1_rows.json', 'w'))

    # ---- summary
    summ = {'manifest': manifest, 'n_rows': len(rows), 'fix_d0': FIX_D0}
    act = [r for r in rows if r['d'] in ACTIVE_DENS]
    ctl = [r for r in rows if r['d'] in CONTROL_DENS]

    summ['control_identical'] = all(
        abs(r['psnr_mrs'] - r['psnr_nvbmf']) < 1e-9 for r in ctl)

    for ds in DATASETS:
        a = [r for r in act if r['dataset'] == ds]
        if not a:
            continue
        gm = np.array([r['psnr_mrs'] for r in a])
        gn = np.array([r['psnr_nvbmf'] for r in a])
        gl = np.array([r['psnr_las2'] for r in a])
        go = np.array([r['psnr_oracle'] for r in a])
        W, z, p, n = wilcoxon_signed_rank(gm, gn)
        summ[ds] = {
            'n_conditions': len(a),
            'mean_gain_mrs_db': float((gm - gn).mean()),
            'mean_gain_local_db': float((gl - gn).mean()),
            'mean_gain_oracle_db': float((go - gn).mean()),
            'better': int((gm > gn).sum()), 'worse': int((gm < gn).sum()),
            'mean_ssim_mrs': float(np.mean([r['ssim_mrs'] for r in a])),
            'mean_ssim_nvbmf': float(np.mean([r['ssim_nvbmf'] for r in a])),
            'mean_ief_mrs': float(np.mean([r['ief_mrs'] for r in a])),
            'mean_ief_nvbmf': float(np.mean([r['ief_nvbmf'] for r in a])),
            'wilcoxon_W': W, 'wilcoxon_z': z, 'wilcoxon_p': p, 'wilcoxon_n': n,
        }
        summ[ds]['per_density'] = {
            f"{d:.2f}": float(np.mean([r['psnr_mrs'] - r['psnr_nvbmf']
                                       for r in a if r['d'] == d]))
            for d in ACTIVE_DENS}

    # ---- roughness / gain correlation over every image in the whole set
    per_img = {}
    for r in act:
        if r['d'] != 0.40:
            continue
        per_img.setdefault((r['dataset'], r['image']), []).append(r)
    rough, gain = [], []
    for k, v in per_img.items():
        rough.append(np.mean([x['D1_over_var'] for x in v]))
        gain.append(np.mean([x['psnr_mrs'] - x['psnr_nvbmf'] for x in v]))
    if len(rough) > 2:
        summ['roughness_gain'] = {
            'n_images': len(rough),
            'spearman': spearman(rough, gain),
            'pearson_log': pearson(np.log(np.maximum(rough, 1e-12)), gain),
        }

    json.dump(summ, open('h1_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1)[:4000])
    print('elapsed', time.time() - t0)
