#!/usr/bin/env python3
"""
The shortest possible demonstration: one image, one density, a few seconds.

    python3 experiments/mrs_demo.py                    # house at 40%, seed 1
    python3 experiments/mrs_demo.py peppers 0.3 2      # any image, density, seed

Prints the PSNR of the four configurations and writes a side-by-side picture to
figures/demo.png so you can see what the rule actually decides.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
from mrs_core import (stage1_with_sources, structure_function_global, isotropic,
                      Dcurve, LAGS, risk_stage1, risk_stage2, _box)
from mrs_run import load, add_spn, box3_nonzero_mean, psnr, IMAGES

name = sys.argv[1] if len(sys.argv) > 1 else 'house'
dens = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

if name not in IMAGES:
    raise SystemExit(f"unknown image {name!r}. choose from: {', '.join(IMAGES)}")

img = load(IMAGES[name])
noisy, mask = add_spn(img, dens, seed)
gt = img.astype(np.float64)
gd = float(mask.mean())

# --- the two candidate restorations
s1, sdi, sdj, rid, rm = stage1_with_sources(noisy, mask)   # nearest surviving ring
s2 = box3_nonzero_mean(s1)                                 # 3x3 mean of that output

# --- the two predicted errors, from the mask and the surviving pairs alone
D = Dcurve(isotropic(structure_function_global(noisy.astype(np.float64), ~mask, LAGS)))
V1 = risk_stage1(mask, rid, rm, D)          # Eq. (7)
V2 = risk_stage2(mask, sdi, sdj, D)         # Eq. (6) + (8)
refine = mask & (V2 < V1)                   # Eq. (10)

# --- the configurations compared in the paper
ld = _box(mask.astype(np.float64), 2) / _box(np.ones_like(s1), 2)
e1, e2 = (s1 - gt) ** 2, (s2 - gt) ** 2
out = {
    'fixed 45% rule [1]': np.where(mask & (gd > 0.45), s2, s1),
    'local density rule': np.where(mask & ((ld > 0.45) | (gd > 0.45)), s2, s1),
    'MRS (proposed)': np.where(refine | (mask & (gd > 0.45)), s2, s1),
    'oracle (uses the clean image)': np.where(mask & (e2 < e1), s2, s1),
}

print(f"\n{name}, density {dens}, seed {seed}   ({mask.sum()} of {mask.size} pixels corrupted)")
print(f"  noisy input{'':22s} {psnr(img, noisy):6.2f} dB")
base = None
for k, v in out.items():
    p = psnr(img, np.clip(np.round(v), 0, 255).astype(np.uint8))
    if base is None:
        base = p
    print(f"  {k:33s} {p:6.2f} dB   {p - base:+5.2f}")
frac = float(refine.sum()) / max(int(mask.sum()), 1)
orc = float((e2 < e1)[mask].mean())
print(f"\n  MRS refines {frac:.3f} of the repaired pixels; the oracle refines {orc:.3f}")
print(f"  the two decisions agree on {float((refine[mask] == (e2 < e1)[mask]).mean()):.3f} of them")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [(img, 'original'), (noisy, f'{int(dens*100)}% impulse noise'),
              (out['fixed 45% rule [1]'], 'fixed 45% rule'),
              (out['MRS (proposed)'], 'MRS'),
              (refine.astype(float) * 255, 'what MRS refines')]
    fig, axs = plt.subplots(1, len(panels), figsize=(2.1 * len(panels), 2.3))
    for a, (im_, t) in zip(axs, panels):
        a.imshow(np.clip(im_, 0, 255), cmap='gray', vmin=0, vmax=255)
        a.set_title(t, fontsize=8)
        a.axis('off')
    plt.tight_layout()
    dest = os.path.join(ROOT, 'figures', 'demo.png')
    plt.savefig(dest, dpi=150, bbox_inches='tight')
    print(f"\n  picture written to {os.path.relpath(dest, ROOT)}")
except ImportError:
    print("\n  (install matplotlib to also get the picture)")
