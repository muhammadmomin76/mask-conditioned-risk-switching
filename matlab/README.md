# MATLAB implementation — Mask-Conditioned Risk Switching (MRS)

Reference implementation for the paper *Mask-Conditioned Risk Switching for
Impulse Noise Removal* (Khuwaja Muhammad Momin, MSCS-2025, IMSciences Peshawar).

Every filter and every metric here is written from its defining equations
using array arithmetic only. No Image Processing Toolbox function is called on
any measured quantity — `imread` is used for file I/O and `imshow`/`print` for
figures, and nothing else. The code runs unmodified in **MATLAB R2018b or
later** and in **GNU Octave 6+**.

---

## Quick start

```matlab
>> cd matlab
>> setup_paths          % puts core/ filters/ metrics/ experiments/ on the path
>> selfcheck            % hand-computable checks of every primitive  (~10 s)
>> mrs_demo             % house at 40% noise, prints a table and saves figures
```

To restore a single image:

```matlab
noisy = add_spn(load_gray('images/benchmark/house.png'), 0.4, 1);
out   = mrs_filter(noisy);                 % detector of [1] finds the mask
psnr_db(clean, out)
```

To regenerate every number in the paper:

```matlab
>> run_all              % all tables and figures  (~25 min in Octave)
```

To regenerate only the headline result of Section V-B (the 72 conditions below
the published cutoff, which is where the claim lives):

```matlab
>> rows = run_sweep([0.1 0.2 0.3 0.4], [1 2 3]);
>> summarise_sweep(rows);
```

---

## What the method is

The published two-stage filter [1] decides whether to refine a repaired pixel
by comparing the **global noise density** with a fixed constant, `tau = 0.45`
(Eq. 1). One constant is shared by every image, and a density cannot tell a
smooth region from a detailed one although the two need opposite treatment.

MRS replaces that test with a comparison of two **predicted errors** (Eq. 10):

```
refine p   iff   V2(p) < V1(p)
```

Under fixed-value impulse noise the surviving pixels are **exact**, so the
error at a repaired pixel is a pure prediction error over a source set that
the noise mask fixes. The expected squared error of an averaged estimate is the
geostatistical extension variance (Eq. 6):

```
V(p,S) = (1/m) sum_{s in S} D(|p-s|)  -  (1/2m^2) sum_{s,t in S} D(|s-t|)
```

with `D(h)` the image structure function, estimated from **surviving pixel
pairs only** (Eq. 9). Both source sets and the structure function come from the
corrupted image itself, so no reference image and no tuned constant enter the
decision.

---

## Layout

| Path | Contents |
|---|---|
| `core/` | the derivation: source sets, structure function, extension variances |
| `filters/` | the filters: MRS, NVBMF, the local-density baseline, medians |
| `metrics/` | PSNR, SSIM, MSE, residual impulse rate, all from their equations |
| `experiments/` | the scripts that produce each table of the paper |
| `images/` | the six 8-bit grayscale test images of Table I |
| `verification/` | reference data exported from the original NumPy implementation |
| `results/` | written by the experiment scripts |

### core/

| File | Role |
|---|---|
| `add_spn.m` | salt-and-pepper synthesis, Eq. (11) |
| `set_seed.m` | portable Mersenne Twister seeding (MATLAB / Octave) |
| `shift_image.m` | the array-shift primitive every windowed operation is built from |
| `distance_rings.m` | window offsets grouped into rings of equal distance |
| `stage1_with_sources.m` | Stage-1 repair **plus** the donor bookkeeping the risk needs |
| `structure_function_global.m` | `D(h)` from surviving pairs, Eq. (9) |
| `structure_function_local.m` | the per-pixel variant of Table VI |
| `isotropic_D.m`, `eval_D.m` | isotropic `D(r)` curve and its evaluation |
| `risk_stage1.m` | `V1`, Eq. (7) |
| `risk_stage2.m` | `V2`, Eq. (6) with the effective sources of Eq. (8) |
| `box_sum.m` | summed-area window sums |
| `mrs_lags.m` | the lag set at which `D` is estimated |
| `round_half_even.m` | NumPy-compatible output quantisation (see below) |

### filters/

| File | Role |
|---|---|
| `mrs_filter.m` | **the proposed method** |
| `nvbmf.m` | the published two-stage filter [1], fixed 45% rule |
| `local_density_filter.m` | the local-density baseline of Section IV-A |
| `box3_nonzero_mean.m` | Stage 2, the 3x3 non-zero mean |
| `is_noise.m` | the detector of [1] |
| `median_filter_k.m`, `adaptive_median_filter.m` | comparison baselines |
| `window_stack.m` | window stacking used by the two median filters |

### experiments/

| Script | Produces |
|---|---|
| `selfcheck.m` | hand-computable checks of every primitive |
| `verify_against_reference.m` | cross-language check against the NumPy reference |
| `run_quality_table.m` | Table II |
| `run_sweep.m` + `summarise_sweep.m` | Tables III, V, VI, VII |
| `run_ablation.m` | Table IV |
| `run_residual.m` | Table VIII |
| `run_timing.m` | Table IX |
| `mrs_demo.m` | the demonstration figures |

---

## Verification

The results in the paper were produced by a NumPy implementation. NumPy's
PCG64 generator and MATLAB's Mersenne Twister cannot produce identical noise,
so the port is verified in two independent ways.

**1. Same input, both implementations** — `verify_against_reference` loads four
corrupted images exported from the NumPy run (`verification/python_reference.mat`)
and compares every intermediate quantity. Measured result:

```
                            s1/s2       V1/V2      decisions differing   PSNR
cameraman  d=0.2  seed 2    0 / 0    2.9e-15 / 0      0 / 52 439        0.0e+00 dB
house      d=0.4  seed 1    0 / 0    1.1e-15 / 0      0 / 105 232       0.0e+00 dB
mandrill   d=0.6  seed 1    0 / 0    4.6e-16 / 0      0 / 157 501       0.0e+00 dB
manuscript d=0.3  seed 3    0 / 0          0 / 0      0 / 49 414        0.0e+00 dB
```

The two candidate restorations, both predicted risks, the estimated structure
function and all five reported PSNRs agree to machine precision, and the
refine/do-not-refine decision agrees on **every single repaired pixel**.

Two details were needed to reach exact agreement, and both are documented in
the code:

* `round_half_even.m` — MATLAB's `round` breaks ties away from zero, NumPy's
  `np.round` breaks them to even. Stage-1 repairs are means of small integer
  sets, so exact halves are common and the two conventions disagree on a few
  thousand pixels of a 512x512 image, worth about 0.002 dB.
* `box_sum.m` — the summed-area window bounds are 0-based in the reference and
  1-based here; getting this wrong shifts the 5x5 local-density window by one
  row and column and changes the local-density baseline by ~0.02 dB.

**2. End-to-end, MATLAB generating its own noise** — the 72 conditions below
the published cutoff (6 images x 4 densities x 3 seeds), run entirely in this
implementation:

| Quantity | This implementation | Paper |
|---|---|---|
| MRS mean gain over the fixed rule | **+0.497 dB** | +0.483 dB |
| MRS better in | **69 / 72** | 69 / 72 |
| Local density rule mean gain | **+0.127 dB** | +0.123 dB |
| MRS vs local density rule | **+0.370 dB** | +0.360 dB |
| Oracle bound | **+1.794 dB** | +1.764 dB |
| Seed-to-seed std of the baseline | **0.089 dB** | 0.079 dB |

Every difference is smaller than the seed-to-seed variation of the experiment
itself, which is what a different noise realisation should produce. The
better/worse *counts* of the near-zero-gain conditions move around more, as
expected: the paper's local-density rule is exactly tied on 30 of the 72
conditions, and a different realisation breaks those ties either way.

---

## Notes

* `mrs_filter` defaults to the **hybrid** configuration of Section III-E: the
  risk comparison below the published 0.45 cutoff, the published global rule
  above it. Above the cutoff the decision approaches chance (Table VII), so
  this is a property of the problem and not a tuning choice. Pass
  `opts.hybrid = false` to run the risk comparison everywhere.
* `opts.useLocalD = true` selects the per-pixel structure function of
  Table VI. It removes the single degradation and agrees slightly better with
  the oracle, at a substantially higher cost.
* The experiments pass the **true** corruption mask to the filters, so what is
  measured is the switching rule and not the detector. `mrs_filter(noisy)` with
  no mask uses the detector of [1] instead, which is what a deployment would do.
* Running times in Octave are several times those of MATLAB; compare the
  *ratios* in Table IX rather than the milliseconds.

[1] B. Turan, "A new approach for SPN removal: nearest value based mean
filter," *PeerJ Comput. Sci.*, vol. 8, p. e1160, 2022.
