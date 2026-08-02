#!/usr/bin/env python3
"""
Export the cross-language reference file that the MATLAB port checks itself against.

`matlab/experiments/verify_against_reference.m` feeds the MATLAB implementation the
same corrupted input the Python implementation saw, and compares every intermediate
quantity: the Stage-1 repair, the Stage-2 candidate, both predicted risks, the
structure-function curve, and the five PSNRs. Agreement to floating-point tolerance
means the port is faithful.

Run this whenever the Python side changes, or the .mat file no longer matches.

    python3 experiments/export_reference.py

Writes matlab/verification/python_reference.mat
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
from scipy.io import savemat

from mrs_core import (stage1_with_sources, structure_function_global, isotropic,
                      Dcurve, LAGS, risk_stage1, risk_stage2, _box)
from mrs_run import load, add_spn, box3_nonzero_mean, psnr, IMAGES

CASES = [('cameraman_d40_s1', 'cameraman', 0.4, 1),
         ('house_d40_s1',     'house',     0.4, 1),
         ('mandrill_d20_s2',  'mandrill',  0.2, 2),
         ('manuscript_d30_s3', 'manuscript_beowulf', 0.3, 3)]


def one(name, d, seed):
    img = load(IMAGES[name])
    noisy, mask = add_spn(img, d, seed)
    surv = ~mask
    gt = img.astype(np.float64)

    s1, sdi, sdj, rid, rm = stage1_with_sources(noisy, mask)
    s2 = box3_nonzero_mean(s1)
    curve = Dcurve(isotropic(structure_function_global(noisy.astype(np.float64), surv, LAGS)))
    V1 = risk_stage1(mask, rid, rm, curve)
    V2 = risk_stage2(mask, sdi, sdj, curve)

    gd = float(mask.mean())
    ld = _box(mask.astype(np.float64), 2) / _box(np.ones_like(s1), 2)
    e1 = (s1 - gt) ** 2
    e2 = (s2 - gt) ** 2
    q = lambda sel: psnr(img, np.clip(np.round(np.where(sel, s2, s1)), 0, 255).astype(np.uint8))

    five = [q(mask & (gd > 0.45)),
            q(mask & ((ld > 0.45) | (gd > 0.45))),
            q(mask & (V2 < V1)),
            q(mask & ((V2 < V1) | (gd > 0.45))),
            q(mask & (e2 < e1))]

    return {'clean': img.astype(np.uint8),
            'noisy': noisy.astype(np.uint8),
            'mask': mask.astype(np.uint8),
            's1': s1, 's2': s2, 'V1': V1, 'V2': V2,
            'Dr': curve.r.reshape(-1, 1), 'DD': curve.D.reshape(-1, 1),
            'psnr': np.array(five, np.float64).reshape(-1, 1)}


if __name__ == '__main__':
    out = {}
    for tag, name, d, seed in CASES:
        rec = one(name, d, seed)
        for k, v in rec.items():
            out[f'{tag}__{k}'] = v
        print(f'{tag:22s} PSNR fixed/local/mrs/hybrid/oracle = '
              + ' '.join(f'{x:.4f}' for x in rec['psnr'].ravel()))
    dest = os.path.join(ROOT, 'matlab', 'verification')
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, 'python_reference.mat')
    savemat(path, out, do_compression=True)
    print('wrote', path, f'({os.path.getsize(path)/1e6:.1f} MB)')
