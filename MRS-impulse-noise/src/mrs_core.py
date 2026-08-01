"""Mask-conditioned risk switching (MRS) for NVBMF-class two-stage impulse filters.

Derivation
----------
Under salt-and-pepper corruption the surviving pixels carry TRUE values, so the
restoration error at a repaired pixel p is pure prediction error: p is estimated by
an average over a set S of source locations, all of which hold true image values.

For a weighted-equally average over a source multiset S (|S| = m), the expected
squared error is the geostatistical extension variance:

    V(p, S) = (1/m) sum_{s in S} D(|p - s|)  -  (1/(2 m^2)) sum_{s,t in S} D(|s - t|)

with D(h) = E[(I(x) - I(x+h))^2] the structure function (= 2 * semivariogram).

Stage 1 estimates p by the mean of the tied-nearest surviving pixels -> S1 is known
exactly from the mask.  Stage 2 replaces that by the 3x3 non-zero mean of the Stage-1
output; each window position contributes either its own true value (if it survived) or
its donor's value (if it was repaired), so the EFFECTIVE source multiset S2 is also
known exactly from the mask.  Both are therefore computable with no ground truth.

D is estimated from surviving pixel pairs only, either globally or in a local window.
"""
import numpy as np

# ---------------------------------------------------------------- rings / offsets

def distance_rings(radius=5):
    rings = {}
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            rings.setdefault(di * di + dj * dj, []).append((di, dj))
    return [(d2, rings[d2]) for d2 in sorted(rings)]

RINGS = distance_rings(5)


def shift(a, di, dj, fill=0.0):
    out = np.full_like(a, fill, dtype=a.dtype)
    H, W = a.shape
    si0, si1 = max(0, di), min(H, H + di)
    di0, di1 = max(0, -di), min(H, H - di)
    sj0, sj1 = max(0, dj), min(W, W + dj)
    dj0, dj1 = max(0, -dj), min(W, W - dj)
    if si1 > si0 and sj1 > sj0:
        out[di0:di1, dj0:dj1] = a[si0:si1, sj0:sj1]
    return out


# ---------------------------------------------------------------- structure function

def structure_function_global(img, surv, lags):
    """D(h) estimated from surviving pairs only. lags = list of (di,dj)."""
    x = img.astype(np.float64)
    m = surv.astype(np.float64)
    out = {}
    for (di, dj) in lags:
        xs, ms = shift(x, di, dj), shift(m, di, dj)
        w = m * ms
        n = w.sum()
        out[(di, dj)] = float((w * (x - xs) ** 2).sum() / n) if n > 0 else np.nan
    return out


def _box(a, r):
    """Box sum over a (2r+1)^2 window, via summed-area table."""
    H, W = a.shape
    ii = np.zeros((H + 1, W + 1), np.float64)
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    i0 = np.clip(np.arange(H) - r, 0, H)
    i1 = np.clip(np.arange(H) + r + 1, 0, H)
    j0 = np.clip(np.arange(W) - r, 0, W)
    j1 = np.clip(np.arange(W) + r + 1, 0, W)
    return (ii[np.ix_(i1, j1)] - ii[np.ix_(i0, j1)]
            - ii[np.ix_(i1, j0)] + ii[np.ix_(i0, j0)])


def structure_function_local(img, surv, lags, radius=16):
    """D(h) estimated per pixel inside a (2r+1)^2 window, surviving pairs only."""
    x = img.astype(np.float64)
    m = surv.astype(np.float64)
    out = {}
    for (di, dj) in lags:
        xs, ms = shift(x, di, dj), shift(m, di, dj)
        w = m * ms
        num = _box(w * (x - xs) ** 2, radius)
        den = _box(w, radius)
        out[(di, dj)] = (num, den)
    return out


class Dcurve:
    """Isotropic D(r) as a function of distance, from lag estimates."""

    def __init__(self, r2_to_D):
        r2 = np.array(sorted(r2_to_D), dtype=np.float64)
        vals = np.array([r2_to_D[k] for k in sorted(r2_to_D)], dtype=np.float64)
        ok = np.isfinite(vals)
        self.r = np.sqrt(r2[ok])
        self.D = vals[ok]

    def __call__(self, r):
        r = np.asarray(r, np.float64)
        v = np.interp(r, self.r, self.D)
        # D is zero at zero lag. np.interp clamps below the smallest measured
        # lag, so without this the i=j terms of the ordered double sum in
        # Eqn (6) would be charged D(r_min) instead of 0.
        return np.where(r <= 0.0, 0.0, v)


def isotropic(dmap):
    """Average directional lag estimates into r^2 -> D."""
    acc = {}
    for (di, dj), v in dmap.items():
        if np.isfinite(v):
            acc.setdefault(di * di + dj * dj, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


LAGS = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 2), (2, -2),
        (0, 3), (3, 0), (1, 2), (2, 1), (0, 4), (4, 0), (3, 3), (3, -3),
        (0, 5), (5, 0), (0, 6), (6, 0), (0, 8), (8, 0)]


# ---------------------------------------------------------------- stage 1, vectorised

def stage1_with_sources(noisy, mask):
    """Vectorised NVBMF Stage 1.

    Returns (s1, src_di, src_dj, ring_idx, ring_m) where for each repaired pixel
    src_di/src_dj is the offset to the CENTROID of its tied-nearest donors, ring_idx
    identifies the winning ring and ring_m the number of tied donors in it. Clean
    pixels get offset (0,0).
    """
    x = noisy.astype(np.float64)
    surv = (~mask).astype(np.float64)
    H, W = x.shape

    val = np.zeros((H, W))
    done = ~mask
    ring_id = np.full((H, W), -1, np.int16)
    ring_m = np.zeros((H, W), np.float64)
    sdi = np.zeros((H, W))
    sdj = np.zeros((H, W))

    for k, (d2, offs) in enumerate(RINGS):
        todo = ~done
        if not todo.any():
            break
        s = np.zeros((H, W))
        c = np.zeros((H, W))
        adi = np.zeros((H, W))
        adj = np.zeros((H, W))
        for (di, dj) in offs:
            sm = shift(surv, di, dj)
            sv = shift(x, di, dj)
            s += sv * sm
            c += sm
            adi += di * sm
            adj += dj * sm
        hit = todo & (c > 0)
        if hit.any():
            val[hit] = s[hit] / c[hit]
            ring_id[hit] = k
            ring_m[hit] = c[hit]
            sdi[hit] = adi[hit] / c[hit]
            sdj[hit] = adj[hit] / c[hit]
            done |= hit

    # all-noise-window fallback: leave as running-mean surrogate (global clean mean)
    left = ~done
    if left.any():
        cm = x[~mask].mean() if (~mask).any() else x.mean()
        val[left] = cm
        ring_m[left] = 1.0

    s1 = np.where(mask, val, x)
    return s1, sdi, sdj, ring_id, ring_m


# ---------------------------------------------------------------- predicted risks

def _ring_pair_term(mask, ring_id, ring_m, Dfun):
    """(1/(2 m^2)) * sum_{s,t in S1} D(|s-t|) for the winning ring of each pixel."""
    H, W = mask.shape
    surv = (~mask).astype(np.float64)
    acc = np.zeros((H, W))
    for k, (d2, offs) in enumerate(RINGS):
        sel = (ring_id == k) & mask
        if not sel.any():
            continue
        tot = np.zeros((H, W))
        for (ai, aj) in offs:
            ma = shift(surv, ai, aj)
            for (bi, bj) in offs:
                mb = shift(surv, bi, bj)
                h = np.hypot(ai - bi, aj - bj)
                tot += ma * mb * np.asarray(Dfun(h))
        acc[sel] = tot[sel]
    m2 = np.maximum(ring_m, 1.0) ** 2
    return acc / (2.0 * m2)


def risk_stage1(mask, ring_id, ring_m, Dfun):
    """V1 = D(r) - pair term. All donors sit at the same distance r."""
    H, W = mask.shape
    r = np.zeros((H, W))
    for k, (d2, offs) in enumerate(RINGS):
        sel = ring_id == k
        if sel.any():
            r[sel] = np.sqrt(d2)
    first = Dfun(r)
    return np.maximum(first - _ring_pair_term(mask, ring_id, ring_m, Dfun), 1e-9)


WIN3 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]


def risk_stage2(mask, sdi, sdj, Dfun):
    """V2 for the 3x3 mean of the Stage-1 output.

    Effective source offset of window position u is u + srcoff(p+u), where srcoff is
    (0,0) for a surviving pixel and the donor centroid offset for a repaired one.
    """
    H, W = mask.shape
    S = []
    for (ui, uj) in WIN3:
        S.append((ui + shift(sdi, ui, uj), uj + shift(sdj, ui, uj)))
    m = float(len(S))
    t1 = np.zeros((H, W))
    for (ai, aj) in S:
        t1 += Dfun(np.hypot(ai, aj))
    t1 /= m
    t2 = np.zeros((H, W))
    for (ai, aj) in S:
        for (bi, bj) in S:
            t2 += Dfun(np.hypot(ai - bi, aj - bj))
    t2 /= (2.0 * m * m)
    return np.maximum(t1 - t2, 1e-9)
