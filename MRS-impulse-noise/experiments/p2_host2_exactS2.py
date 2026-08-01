#!/usr/bin/env python3
"""
Pass 2, job C -- diagnostic for the second-host failure.

Job A showed that MRS transfers poorly to the adaptive-window mean host: it agrees
with that host's oracle on only 0.42 of repaired pixels, which is BELOW chance, and
it refines 0.63 of them against the oracle's 0.37. The suspect is the first
approximation of Eqn (8): a repaired pixel is represented by ONE source at the
centroid of its donors. On host [1] the donors are a tied ring and sit tight, so the
collapse is harmless. On the adaptive-window host they are spread over the whole
window, and their centroid sits almost on the pixel itself, so S2 looks far tighter
than it is, V2 is underestimated, and the rule over-refines.

This job recomputes V2 for host 2 WITHOUT the collapse: every window position is
expanded into its full surviving source multiset, and Eqn (6) is applied to the
union. If the agreement recovers, the centroid collapse is the cause and the paper
can say so; if it does not, the transfer fails for a deeper reason.

Grid: 6 images x {0.1,0.2,0.3,0.4} x seeds {1,2,3}.
Out : p2_host2_exact.json
"""
import sys, json, time
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np

from mrs_core import (structure_function_global, isotropic, Dcurve, LAGS, shift)
from mrs_run import load, add_spn, box3_nonzero_mean, psnr, IMAGES
from p2_ablation_host2 import host2_stage1, host2_risk1, MAXR

DENS = [0.1, 0.2, 0.3, 0.4]
SEEDS = [1, 2, 3]
WIN3 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]


def host2_risk2_exact(mask, rad, Dfun):
    """V2 with every window position expanded into its true source multiset.

    Source multiset of p:
        for each u in the 3x3 window,
            if p+u survived            -> the single offset u
            if p+u was repaired at r   -> the offsets u+v for every survivor v
                                          of its (2r+1)^2 window
    Weights are the survival indicators, so the multiset is read from M alone.
    """
    H, W = mask.shape
    surv = (~mask).astype(np.float64)
    rep = mask.astype(np.float64)

    # (offset, weight-array) pairs making up the union
    terms = []
    for (ui, uj) in WIN3:
        s_here = shift(surv, ui, uj)          # p+u survived
        terms.append(((float(ui), float(uj)), s_here))
        for r in range(1, MAXR + 1):
            sel_r = shift(((rad == r) * rep).astype(np.float64), ui, uj)
            if not sel_r.any():
                continue
            for vi in range(-r, r + 1):
                for vj in range(-r, r + 1):
                    if vi == 0 and vj == 0:
                        continue
                    w = sel_r * shift(surv, ui + vi, uj + vj)
                    terms.append(((float(ui + vi), float(uj + vj)), w))

    m = np.zeros((H, W))
    for _, w in terms:
        m += w
    m = np.maximum(m, 1.0)

    t1 = np.zeros((H, W))
    for (ai, aj), w in terms:
        t1 += w * float(Dfun(np.hypot(ai, aj)))
    t1 /= m

    t2 = np.zeros((H, W))
    for (ai, aj), wa in terms:
        for (bi, bj), wb in terms:
            h = np.hypot(ai - bi, aj - bj)
            if h == 0.0:
                continue
            t2 += wa * wb * float(Dfun(h))
    t2 /= (2.0 * m * m)
    return np.maximum(t1 - t2, 1e-9)


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
                h1, cdi, cdj, rad, msz = host2_stage1(noisy, mask)
                h2 = box3_nonzero_mean(h1)
                W1 = host2_risk1(mask, rad, msz, D)
                W2e = host2_risk2_exact(mask, rad, D)
                he1 = (h1 - gt) ** 2
                he2 = (h2 - gt) ** 2
                orc = he2 < he1
                sel = mask & (W2e < W1)
                q = lambda s: psnr(img, np.clip(np.round(np.where(s, h2, h1)), 0, 255).astype(np.uint8))
                rows.append({'image': nm, 'd': d, 'seed': seed,
                             'h2_fixed': q(np.zeros_like(mask)),
                             'h2_mrs_exact': q(sel),
                             'h2_oracle': q(mask & orc),
                             'agree_exact': float((sel[mask] == orc[mask]).mean()),
                             'frac_exact': float(sel.sum()) / max(int(mask.sum()), 1),
                             'frac_oracle': float(orc[mask].mean())})
                print(f"{nm:20s} d={d} s={seed} fixed={rows[-1]['h2_fixed']:.3f} "
                      f"exact={rows[-1]['h2_mrs_exact']:.3f} oracle={rows[-1]['h2_oracle']:.3f} "
                      f"agree={rows[-1]['agree_exact']:.3f} frac={rows[-1]['frac_exact']:.3f}"
                      f" [{time.time()-t0:7.1f}s]", flush=True)
                json.dump(rows, open(os.path.join(ROOT,'results','p2_host2_exact.json'), 'w'), indent=1)
    print('elapsed', time.time() - t0)
