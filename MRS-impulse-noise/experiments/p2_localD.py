#!/usr/bin/env python3
"""
Pass 2, job B -- Table VI redone with the corrected D(0)=0 convention.

Compares the global structure-function estimator against the 33x33 local one
(radius 16, falling back to the global curve wherever a lag bin holds 8 pairs or
fewer), on the 72 below-cutoff conditions. Reports gain over the fixed 45% rule,
better/worse counts and agreement with the oracle, for both estimators.

Out: p2_localD.json
"""
import sys, json, time
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np

from mrs_core import (stage1_with_sources, structure_function_global, isotropic,
                      Dcurve, LAGS, risk_stage1, risk_stage2)
from mrs_run import load, add_spn, box3_nonzero_mean, psnr, IMAGES, LocalD

DENS = [0.1, 0.2, 0.3, 0.4]
SEEDS = [1, 2, 3]

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
                s1, sdi, sdj, rid, rm = stage1_with_sources(noisy, mask)
                s2 = box3_nonzero_mean(s1)
                e1 = (s1 - gt) ** 2
                e2 = (s2 - gt) ** 2
                orc = e2 < e1
                q = lambda sel: psnr(
                    img, np.clip(np.round(np.where(sel, s2, s1)), 0, 255).astype(np.uint8))

                out = {'image': nm, 'd': d, 'seed': seed,
                       'fixed': q(np.zeros_like(mask)), 'oracle': q(mask & orc)}
                for tag, Dfun in (
                        ('glob', Dcurve(isotropic(structure_function_global(
                            noisy.astype(np.float64), surv, LAGS)))),
                        ('loc', LocalD(noisy.astype(np.float64), surv, 16))):
                    V1 = risk_stage1(mask, rid, rm, Dfun)
                    V2 = risk_stage2(mask, sdi, sdj, Dfun)
                    sel = mask & (V2 < V1)
                    out['mrs_' + tag] = q(sel)
                    out['agree_' + tag] = float((sel[mask] == orc[mask]).mean())
                    out['frac_' + tag] = float(sel.sum()) / max(int(mask.sum()), 1)
                rows.append(out)
                print(f"{nm:20s} d={d} s={seed} glob={out['mrs_glob']:.3f} "
                      f"loc={out['mrs_loc']:.3f} fixed={out['fixed']:.3f} "
                      f"[{time.time()-t0:7.1f}s]", flush=True)
                json.dump(rows, open(os.path.join(ROOT,'results','p2_localD.json'), 'w'), indent=1)
    print('elapsed', time.time() - t0)
