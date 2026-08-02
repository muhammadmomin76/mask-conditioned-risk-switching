#!/usr/bin/env python3
"""Redraw Figure 10, the visual comparison, from the corrected D(0)=0 rule.

The published figure was drawn with the zero-lag clamp in place, where the rule
refined 0.104 of the repaired pixels. The corrected rule refines 0.344 of them,
so panel (vi), the refinement map, is materially different and the figure had to
be regenerated. Layout and panel order follow the published version.
"""
import sys
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mrs_core import (stage1_with_sources, structure_function_global, isotropic,
                      Dcurve, LAGS, risk_stage1, risk_stage2, _box)
from mrs_run import load, add_spn, box3_nonzero_mean, psnr, IMAGES

plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'figure.dpi': 200,
                     'savefig.bbox': 'tight'})

img = load(IMAGES['house'])
noisy, mask = add_spn(img, 0.4, 1)
gd = float(mask.mean())
s1, sdi, sdj, rid, rm = stage1_with_sources(noisy, mask)
s2 = box3_nonzero_mean(s1)
D = Dcurve(isotropic(structure_function_global(noisy.astype(np.float64), ~mask, LAGS)))
V1 = risk_stage1(mask, rid, rm, D)
V2 = risk_stage2(mask, sdi, sdj, D)
sel = mask & (V2 < V1)

nv = np.where(mask & (gd > 0.45), s2, s1)
ld = _box(mask.astype(float), 2) / _box(np.ones_like(s1), 2)
la = np.where(mask & ((ld > 0.45) | (gd > 0.45)), s2, s1)
mr = np.where(sel | (mask & (gd > 0.45)), s2, s1)

q = lambda a: psnr(img, np.clip(np.round(a), 0, 255).astype(np.uint8))
sl = (slice(150, 280), slice(150, 280))
panels = [(img, '(i) original'),
          (noisy, '(ii) 40% impulse noise'),
          (nv, f'(iii) fixed 45% rule, {q(nv):.2f} dB'),
          (la, f'(iv) local density, {q(la):.2f} dB'),
          (mr, f'(v) MRS, {q(mr):.2f} dB'),
          (sel.astype(float) * 255, '(vi) MRS refinement map')]

fig, axs = plt.subplots(1, 6, figsize=(9.2, 1.8))
for a, (im_, t) in zip(axs, panels):
    a.imshow(np.clip(im_, 0, 255)[sl], cmap='gray', vmin=0, vmax=255)
    a.set_title(t, fontsize=7)
    a.axis('off')
    a.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(ROOT,'figures','fig10_house_visual.png'))
plt.close()

frac = float(sel.sum()) / max(int(mask.sum()), 1)
print(f'refined fraction on this condition: {frac:.3f}')
print(f'PSNR  fixed {q(nv):.2f}  local {q(la):.2f}  MRS {q(mr):.2f}')
