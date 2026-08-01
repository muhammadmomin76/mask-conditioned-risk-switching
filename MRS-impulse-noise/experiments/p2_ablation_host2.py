#!/usr/bin/env python3
"""
Pass 2, job A
=============
(1) Re-runs the Section V-B2 ablation with the corrected D(0)=0 convention.
    The geometry-only rules do not involve D at all, so they are unchanged by
    construction; re-running them is the check that this is true.
(2) M6: grafts MRS onto a SECOND host filter and reports it on the same grid.

SECOND HOST -- read this before citing it
-----------------------------------------
The review proposed DAMF or MDBUTMF. Neither is admissible without changing the
derivation: DAMF replaces a corrupted pixel by the MEDIAN of the surviving
neighbours of an adaptive window, and MDBUTMF by a trimmed median. Eqn (6) is the
extension variance of an UNWEIGHTED MEAN, and an order statistic has no extension
variance in this framework.

The host used here is therefore the adaptive-window mean of the surviving pixels,
which is the mean-based member of the same 2014-2020 family and is the filter [8]
belongs to, taken in its unweighted form so that Eqn (6) applies exactly:

    Stage 1  grow a (2r+1)^2 window from r=1 until it contains a survivor, then
             replace the corrupted pixel by the unweighted mean of every survivor
             in that window.  S1 = those survivors, read from M.
    Stage 2  the 3x3 non-extreme mean of the Stage-1 output, as in [1].
             S2 = each window offset traced through its donor centroid.

This differs from [1] in exactly the way that matters for the claim: [1] copies
the nearest ring, this host averages a growing neighbourhood, so the two have
different S1 geometry and different V1. If MRS improves both, the rule is not a
property of [1].

Grid: 6 images x {0.1,0.2,0.3,0.4} x seeds {1,2,3} = 72 conditions.
Out : p2_ablation_host2.json
"""
import json, time
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np

from mrs_core import (stage1_with_sources, structure_function_global, isotropic,
                      Dcurve, LAGS, risk_stage1, risk_stage2, shift, RINGS, _box)
from mrs_run import load, add_spn, box3_nonzero_mean, psnr, IMAGES

DENS = [0.1, 0.2, 0.3, 0.4]
SEEDS = [1, 2, 3]
MAXR = 3


# ------------------------------------------------------------------ host 2
def host2_stage1(noisy, mask):
    """Adaptive-window unweighted mean of survivors.

    Returns (s1, cdi, cdj, radius, m) with cdi/cdj the centroid offset of the
    source set and m its size, so the pair term can be built per radius.
    """
    x = noisy.astype(np.float64)
    surv = (~mask).astype(np.float64)
    H, W = x.shape
    val = np.zeros((H, W))
    done = ~mask
    rad = np.zeros((H, W), np.int16)
    msz = np.zeros((H, W))
    cdi = np.zeros((H, W))
    cdj = np.zeros((H, W))

    for r in range(1, MAXR + 1):
        todo = ~done
        if not todo.any():
            break
        s = np.zeros((H, W)); c = np.zeros((H, W))
        adi = np.zeros((H, W)); adj = np.zeros((H, W))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di == 0 and dj == 0:
                    continue
                sm = shift(surv, di, dj)
                s += shift(x, di, dj) * sm
                c += sm
                adi += di * sm
                adj += dj * sm
        hit = todo & (c > 0)
        if hit.any():
            val[hit] = s[hit] / c[hit]
            rad[hit] = r
            msz[hit] = c[hit]
            cdi[hit] = adi[hit] / c[hit]
            cdj[hit] = adj[hit] / c[hit]
            done |= hit

    left = ~done
    if left.any():
        cm = x[~mask].mean() if (~mask).any() else x.mean()
        val[left] = cm
        msz[left] = 1.0
        rad[left] = MAXR
    return np.where(mask, val, x), cdi, cdj, rad, msz


def host2_risk1(mask, rad, msz, Dfun):
    """V1 for an unweighted mean over every survivor of a (2r+1)^2 window."""
    H, W = mask.shape
    surv = (~mask).astype(np.float64)
    first = np.zeros((H, W))
    pair = np.zeros((H, W))
    for r in range(1, MAXR + 1):
        sel = (rad == r) & mask
        if not sel.any():
            continue
        offs = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                if not (di == 0 and dj == 0)]
        sm = {o: shift(surv, o[0], o[1]) for o in offs}
        t1 = np.zeros((H, W))
        for (ai, aj) in offs:
            t1 += sm[(ai, aj)] * float(Dfun(np.hypot(ai, aj)))
        t2 = np.zeros((H, W))
        for (ai, aj) in offs:
            for (bi, bj) in offs:
                h = np.hypot(ai - bi, aj - bj)
                if h == 0.0:
                    continue                       # D(0) = 0, ordered diagonal
                t2 += sm[(ai, aj)] * sm[(bi, bj)] * float(Dfun(h))
        m = np.maximum(msz, 1.0)
        first[sel] = (t1 / m)[sel]
        pair[sel] = (t2 / (2.0 * m * m))[sel]
    return np.maximum(first - pair, 1e-9)


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    t0 = time.time()
    rows = []
    for nm, path in IMAGES.items():
        img = load(path)
        gt = img.astype(np.float64)
        for d in DENS:
            for seed in SEEDS:
                noisy, mask = add_spn(img, d, seed)
                surv = ~mask
                D = Dcurve(isotropic(structure_function_global(
                    noisy.astype(np.float64), surv, LAGS)))

                # ---------- host 1 = NVBMF [1]
                s1, sdi, sdj, rid, rm = stage1_with_sources(noisy, mask)
                s2 = box3_nonzero_mean(s1)
                q = lambda sel, a=s1, b=s2: psnr(
                    img, np.clip(np.round(np.where(sel, b, a)), 0, 255).astype(np.uint8))
                V1 = risk_stage1(mask, rid, rm, D)
                V2 = risk_stage2(mask, sdi, sdj, D)

                rr = np.zeros_like(s1)
                for k, (d2, offs) in enumerate(RINGS):
                    m = rid == k
                    if m.any():
                        rr[m] = np.sqrt(d2)

                out = {'image': nm, 'd': d, 'seed': seed,
                       'fixed': q(np.zeros_like(mask)),
                       'mrs': q(mask & (V2 < V1))}
                for t in [1.0, 1.42, 2.0, 2.24, 2.83]:
                    out[f'geom>{t}'] = q(mask & (rr > t))
                out['ties<2'] = q(mask & (rm < 2))
                e1 = (s1 - gt) ** 2
                e2 = (s2 - gt) ** 2
                out['oracle'] = q(mask & (e2 < e1))

                # ---------- host 2 = adaptive-window mean
                h1, cdi, cdj, rad, msz = host2_stage1(noisy, mask)
                h2 = box3_nonzero_mean(h1)
                qh = lambda sel: psnr(
                    img, np.clip(np.round(np.where(sel, h2, h1)), 0, 255).astype(np.uint8))
                W1 = host2_risk1(mask, rad, msz, D)
                W2 = risk_stage2(mask, cdi, cdj, D)
                he1 = (h1 - gt) ** 2
                he2 = (h2 - gt) ** 2
                out['h2_fixed'] = qh(np.zeros_like(mask))       # never refine, as [1] below cutoff
                out['h2_always'] = qh(mask)                     # always refine
                out['h2_mrs'] = qh(mask & (W2 < W1))
                out['h2_local'] = qh(mask & (
                    _box(mask.astype(np.float64), 2) /
                    _box(np.ones_like(h1), 2) > 0.45))
                out['h2_oracle'] = qh(mask & (he2 < he1))
                out['h2_agree'] = float(((W2 < W1)[mask] == (he2 < he1)[mask]).mean())
                out['h2_frac'] = float((mask & (W2 < W1)).sum()) / max(int(mask.sum()), 1)
                out['h2_frac_oracle'] = float((he2 < he1)[mask].mean())

                rows.append(out)
                print(f"{nm:20s} d={d} s={seed} mrs={out['mrs']:.3f} "
                      f"h2_fixed={out['h2_fixed']:.3f} h2_mrs={out['h2_mrs']:.3f} "
                      f"h2_oracle={out['h2_oracle']:.3f} [{time.time()-t0:6.1f}s]", flush=True)
                json.dump(rows, open(os.path.join(ROOT,'results','p2_ablation_host2.json'), 'w'), indent=1)
    print('elapsed', time.time() - t0)
