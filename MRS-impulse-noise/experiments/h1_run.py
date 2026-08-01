#!/usr/bin/env python3
"""
H1 runner -- one dataset per process.

    python3 h1_run.py bsds200
    python3 h1_run.py testimages40

Budget note. The six-image study used 3 seeds because 6 images cannot carry a
mean on their own. With 240 images one realisation per image per density already
gives 960 paired conditions, which is far more statistical weight than 72, so the
seed axis is spent on images instead. The seed-to-seed variation is already
characterised in Section V-B1 and does not need re-measuring here.

Densities 0.1-0.4 are the band where MRS is active. 0.5 is run on the first 20
images of each set only, as a control: above the cutoff MRS must be bit-identical
to NVBMF, and the script asserts it.

Out: h1_<dataset>.json
"""
import json, glob, time
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import numpy as np

from h1_m7_benchmark import (load, run_one, ACTIVE_DENS)   # run_one has FIX_D0=True

DENS = [0.1, 0.2, 0.3, 0.4]
CONTROL_D = 0.5
CONTROL_N = 20
SEED = 1

if __name__ == '__main__':
    ds = sys.argv[1]
    files = sorted(glob.glob(os.path.join(ROOT,'data',ds,'*.png')))
    print(f"{ds}: {len(files)} images", flush=True)
    rows = []
    t0 = time.time()
    for i, fp in enumerate(files):
        img = load(fp)
        name = os.path.basename(fp)
        for d in DENS:
            r = run_one(img, d, SEED)
            r['dataset'] = ds
            r['image'] = name
            rows.append(r)
        if i < CONTROL_N:
            r = run_one(img, CONTROL_D, SEED)
            r['dataset'] = ds
            r['image'] = name
            r['control'] = True
            rows.append(r)
        if i % 5 == 0 or i == len(files) - 1:
            g = np.mean([x['psnr_mrs'] - x['psnr_nvbmf']
                         for x in rows if not x.get('control')])
            print(f"  [{i+1:3d}/{len(files)}] {name[:34]:34s} running mean gain "
                  f"{g:+.3f} dB   [{time.time()-t0:7.1f}s]", flush=True)
            json.dump(rows, open(os.path.join(ROOT,'results','h1_'+ds+'.json'), 'w'))
    json.dump(rows, open(os.path.join(ROOT,'results','h1_'+ds+'.json'), 'w'))

    ctl = [r for r in rows if r.get('control')]
    ok = all(abs(r['psnr_mrs'] - r['psnr_nvbmf']) < 1e-9 for r in ctl)
    print(f"control ({len(ctl)} conditions at d={CONTROL_D}): "
          f"MRS identical to NVBMF = {ok}", flush=True)
    print('elapsed', time.time() - t0)
