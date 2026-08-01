import os, sys, json, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrs_core import (stage1_with_sources, structure_function_global, structure_function_local,
                 isotropic, Dcurve, LAGS, risk_stage1, risk_stage2, shift, _box)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMDIR = os.path.join(ROOT, 'images')
IMAGES = {
    'cameraman': f'{IMDIR}/benchmark/cameraman.png',
    'house': f'{IMDIR}/benchmark/house.png',
    'mandrill': f'{IMDIR}/benchmark/mandrill.png',
    'peppers': f'{IMDIR}/benchmark/peppers.png',
    'field_vangogh': f'{IMDIR}/custom/field_vangogh.png',
    'manuscript_beowulf': f'{IMDIR}/custom/manuscript_beowulf.png',
}

def load(p):
    a = np.array(Image.open(p).convert('L'))
    return a.astype(np.uint8)

def add_spn(img, d, seed):
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
    ws = np.zeros_like(a); wc = np.zeros_like(a)
    for di in (0,1,2):
        for dj in (0,1,2):
            ws += ap[di:di+H, dj:dj+W] * zp[di:di+H, dj:dj+W]
            wc += zp[di:di+H, dj:dj+W]
    out = a.copy(); has = wc > 0
    out[has] = ws[has]/wc[has]
    return out

def psnr(a, b):
    m = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 99.0 if m == 0 else 10*np.log10(255.0**2/m)

class LocalD:
    """Per-pixel D(r) from local structure-function estimates."""
    def __init__(self, img, surv, radius=16):
        loc = structure_function_local(img, surv, LAGS, radius)
        acc = {}
        for (di,dj),(num,den) in loc.items():
            r2 = di*di+dj*dj
            if r2 not in acc: acc[r2] = [np.zeros_like(num), np.zeros_like(den)]
            acc[r2][0] += num; acc[r2][1] += den
        glob = Dcurve(isotropic(structure_function_global(img, surv, LAGS)))
        rs = sorted(acc)
        self.r = np.sqrt(np.array(rs, np.float64))
        stack = []
        for r2 in rs:
            num, den = acc[r2]
            v = np.where(den > 8, num/np.maximum(den,1e-9), glob(np.sqrt(r2)))
            stack.append(v)
        self.D = np.stack(stack, 0)          # (n_r, H, W)
        self.glob = glob
        H, W = self.D.shape[1:]
        self._ii, self._jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    def __call__(self, r):
        r = np.asarray(r, np.float64)
        if r.ndim == 0:
            r = np.full(self.D.shape[1:], float(r))
        idx = np.searchsorted(self.r, r, side='right') - 1
        idx = np.clip(idx, 0, len(self.r)-2)
        r0 = self.r[idx]; r1 = self.r[idx+1]
        w = np.clip((r-r0)/np.maximum(r1-r0,1e-9), 0, 1)
        ii, jj = self._ii, self._jj
        d0 = self.D[idx, ii, jj]; d1 = self.D[idx+1, ii, jj]
        v = d0*(1-w) + d1*w
        # same zero-lag convention as Dcurve.__call__
        return np.where(r <= 0.0, 0.0, v)

def run_one(name, path, d, seed, use_local=True, radius=16):
    img = load(path)
    noisy, mask = add_spn(img, d, seed)
    surv = ~mask
    s1, sdi, sdj, ring_id, ring_m = stage1_with_sources(noisy, mask)
    s2full = box3_nonzero_mean(s1)

    gd = float(mask.mean())
    # --- reference configurations
    out_fixed = np.where(mask & (gd > 0.45), s2full, s1)
    ld = _box(mask.astype(np.float64), 2) / _box(np.ones_like(s1), 2)
    out_las2 = np.where(mask & ((ld > 0.45) | (gd > 0.45)), s2full, s1)

    # --- oracle: per-pixel best of the two, using ground truth
    e1 = (s1 - img.astype(np.float64))**2
    e2 = (s2full - img.astype(np.float64))**2
    out_oracle = np.where(mask & (e2 < e1), s2full, s1)

    # --- MRS
    Dfun = LocalD(noisy.astype(np.float64), surv, radius) if use_local else \
           Dcurve(isotropic(structure_function_global(noisy.astype(np.float64), surv, LAGS)))
    V1 = risk_stage1(mask, ring_id, ring_m, Dfun)
    V2 = risk_stage2(mask, sdi, sdj, Dfun)
    sel = mask & (V2 < V1)
    out_mrs = np.where(sel, s2full, s1)
    # hybrid: keep the published global rule above its cutoff, use risk below it
    selh = mask & ((V2 < V1) | (gd > 0.45))
    out_hyb = np.where(selh, s2full, s1)

    q = lambda a: psnr(img, np.clip(np.round(a),0,255).astype(np.uint8))
    n = max(int(mask.sum()), 1)
    # decision agreement with oracle, on repaired pixels
    orc = (e2 < e1)[mask]
    agree = float((sel[mask] == orc).mean())
    return dict(image=name, d=d, seed=seed,
                psnr_fixed=q(out_fixed), psnr_las2=q(out_las2),
                psnr_mrs=q(out_mrs), psnr_hyb=q(out_hyb), psnr_oracle=q(out_oracle),
                psnr_s1=q(s1), psnr_s2=q(s2full),
                frac_mrs=float(sel.sum())/n, frac_oracle=float(orc.mean()),
                agree=agree)

if __name__ == '__main__':
    use_local = sys.argv[1] != 'global' if len(sys.argv) > 1 else True
    seeds = [int(s) for s in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1'])]
    rows = []
    t0 = time.time()
    for name, path in IMAGES.items():
        for d in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            for s in seeds:
                r = run_one(name, path, d, s, use_local)
                rows.append(r)
                print(f"{name:20s} d={d:.1f} s={s} fixed={r['psnr_fixed']:6.2f} "
                      f"las2={r['psnr_las2']:6.2f} mrs={r['psnr_mrs']:6.2f} "
                      f"oracle={r['psnr_oracle']:6.2f} agree={r['agree']:.3f} "
                      f"fr={r['frac_mrs']:.3f}/{r['frac_oracle']:.3f}", flush=True)
    tag = 'local' if use_local else 'global'
    json.dump(rows, open(os.path.join(ROOT,'results',f'results_{tag}.json'),'w'), indent=1)
    print('elapsed', time.time()-t0)
