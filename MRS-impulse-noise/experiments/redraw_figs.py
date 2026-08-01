#!/usr/bin/env python3
"""
Redraw Figures 6-9 from the corrected D(0)=0 run.

Figure 6  gain over the fixed 45% rule, per image, (a) local density (b) MRS
Figure 7  mean gain against the oracle bound
Figure 8  the geometry ablation
Figure 9  (a) normalised structure functions, (b) roughness vs gain on the six
          images, (c) the same relation re-estimated over the 200 Berkeley images

Layout, colours, sizes and fonts follow the figures already in the manuscript, so
the redrawn versions drop straight in.
"""
import json
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mrs_core import structure_function_global, isotropic, Dcurve, LAGS
from mrs_run import load, IMAGES

OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

R = json.load(open(os.path.join(ROOT,'results','h3_rows.json')))
AB = json.load(open(os.path.join(ROOT,'results','p2_ablation_host2.json')))
BS = [r for r in json.load(open(os.path.join(ROOT,'results','h1_bsds200.json')))
      if not r.get('control') and abs(r['d'] - 0.4) < 1e-9]

DENS = sorted({r['d'] for r in R})
IMS = ['house', 'cameraman', 'peppers', 'mandrill', 'field_vangogh', 'manuscript_beowulf']
LBL = {'house': 'house', 'cameraman': 'cameraman', 'peppers': 'peppers',
       'mandrill': 'mandrill', 'field_vangogh': 'field', 'manuscript_beowulf': 'manuscript'}
C = {'house': '#1f77b4', 'cameraman': '#d62728', 'peppers': '#2ca02c',
     'mandrill': '#9467bd', 'field_vangogh': '#ff7f0e', 'manuscript_beowulf': '#8c564b'}

plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'mathtext.fontset': 'dejavuserif'})

MRS = 'psnr_mrs_d0'          # the corrected rule
mean = lambda im, d, k: float(np.mean([r[k] for r in R if r['image'] == im and r['d'] == d]))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


# ----------------------------------------------------------------- Figure 6
fig, ax = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True)
for i, (k, t) in enumerate([('psnr_las2', '(a) local density rule'),
                            (MRS, '(b) proposed MRS rule')]):
    for im in IMS:
        ax[i].plot(DENS, [mean(im, d, k) - mean(im, d, 'psnr_nvbmf') for d in DENS],
                   'o-', ms=3.5, lw=1.3, color=C[im], label=LBL[im])
    ax[i].axvline(0.45, color='k', ls=':', lw=1.0)
    ax[i].axhline(0, color='k', lw=0.7)
    ax[i].set_title(t, fontsize=9)
    ax[i].set_ylabel('PSNR gain (dB)')
ax[0].legend(fontsize=7, ncol=3, loc='upper left', framealpha=0.95)
ax[1].set_xlabel('noise density')
ax[1].set_xticks(DENS)
plt.tight_layout()
plt.savefig(f'{OUT}/fig06_gain_per_image.png')
plt.close()

# ----------------------------------------------------------------- Figure 7
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for k, lab, st, col in [('psnr_oracle', 'oracle (uses ground truth)', '--', 'k'),
                        (MRS, 'MRS (proposed)', '-', '#1f77b4'),
                        ('psnr_las2', 'local density rule', '-', '#ff7f0e')]:
    ax.plot(DENS, [np.mean([r[k] - r['psnr_nvbmf'] for r in R if r['d'] == d]) for d in DENS],
            st, marker='o', ms=3.5, lw=1.4, color=col, label=lab)
ax.axvline(0.45, color='k', ls=':', lw=1.0)
ax.axhline(0, color='k', lw=0.7)
ax.set_xlabel('noise density')
ax.set_ylabel('mean PSNR gain (dB)')
ax.set_xticks(DENS)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig07_oracle_bound.png')
plt.close()

# ----------------------------------------------------------------- Figure 8
key = {(r['image'], r['d'], r['seed']): r for r in AB}
g_geom = float(np.mean([r['geom>1.0'] - r['fixed'] for r in AB]))
g_ties = float(np.mean([r['ties<2'] - r['fixed'] for r in AB]))
g_mrs = float(np.mean([r[MRS] - r['psnr_nvbmf'] for r in R if r['d'] <= 0.4]))
labels = ['donor distance > 1', 'fewer than 2 tied donors', 'MRS (proposed)']
vals = [g_geom, g_ties, g_mrs]
fig, ax = plt.subplots(figsize=(4.6, 2.2))
bars = ax.barh(labels, vals, color=['#bbbbbb', '#7f9fc4', '#1f77b4'], height=0.55)
for b, v in zip(bars, vals):
    ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f'{v:+.3f}', va='center', fontsize=8)
ax.set_xlabel('mean PSNR gain below cutoff (dB)')
ax.set_xlim(0, 0.68)
plt.tight_layout()
plt.savefig(f'{OUT}/fig08_geometry_ablation.png')
plt.close()

# ----------------------------------------------------------------- Figure 9
fig, ax = plt.subplots(3, 1, figsize=(4.6, 6.6))
xs, ys = [], []
for im in IMS:
    img = load(IMAGES[im]).astype(np.float64)
    D = Dcurve(isotropic(structure_function_global(img, np.ones(img.shape, bool), LAGS)))
    rr = np.linspace(1, 8, 60)
    ax[0].plot(rr, D(rr) / img.var(), lw=1.3, color=C[im], label=LBL[im])
    x = float(D(1.0)) / img.var()
    y = mean(im, 0.4, MRS) - mean(im, 0.4, 'psnr_nvbmf')
    xs.append(x); ys.append(y)
    ax[1].scatter(x, y, s=34, color=C[im], zorder=3)
    ax[1].annotate(LBL[im], (x, y), textcoords='offset points', xytext=(5, 3), fontsize=7)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('lag r (pixels)')
ax[0].set_ylabel(r'$D(r)/\sigma^2$')
ax[0].set_title('(a) normalised structure function', fontsize=9)
ax[0].legend(fontsize=7, ncol=2, loc='lower right', framealpha=0.95)

ax[1].set_xscale('log')
ax[1].set_xlabel(r'$D(1)/\sigma^2$')
ax[1].set_ylabel('MRS gain at d = 0.4 (dB)')
ax[1].set_title(f'(b) six images, rank correlation = {spearman(xs, ys):.3f}', fontsize=9)
ax[1].margins(x=0.18, y=0.18)

bx = np.array([r['D1_over_var'] for r in BS])
by = np.array([r['psnr_mrs'] - r['psnr_nvbmf'] for r in BS])
ax[2].scatter(bx, by, s=9, color='#1f77b4', alpha=0.55, edgecolors='none')
ax[2].set_xscale('log')
ax[2].set_xlabel(r'$D(1)/\sigma^2$')
ax[2].set_ylabel('MRS gain at d = 0.4 (dB)')
ax[2].set_title(f'(c) 200 Berkeley images, rank correlation = {spearman(bx, by):.3f}', fontsize=9)
ax[2].axhline(0, color='k', lw=0.7)
plt.tight_layout()
plt.savefig(f'{OUT}/fig09_structure_functions.png')
plt.close()

print('Figure 6  local/MRS per image           ok')
print(f'Figure 7  mean gain, MRS {np.mean([r[MRS]-r["psnr_nvbmf"] for r in R if r["d"]<=0.4]):+.3f} dB below cutoff')
print(f'Figure 8  {g_geom:+.3f} / {g_ties:+.3f} / {g_mrs:+.3f}')
print(f'Figure 9  six-image rho {spearman(xs, ys):+.3f}, Berkeley rho {spearman(bx, by):+.3f}')
