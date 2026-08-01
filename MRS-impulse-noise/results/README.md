# Results

The JSON output of every experiment, exactly as used in the paper. They are committed so that
any number can be checked without running anything.

| File | Produced by | Feeds |
|---|---|---|
| `h3_rows.json` | `experiments/h3_metrics.py` | Tables II, III, IV, VIII, XII and Figures 6, 7, 8, 9 |
| `p2_ablation_host2.json` | `experiments/p2_ablation_host2.py` | Tables V and XI, Figure 8 |
| `p2_localD.json` | `experiments/p2_localD.py` | Table VII |
| `p2_host2_exact.json` | `experiments/p2_host2_exactS2.py` | the Section V-B7 diagnostic |
| `h1_bsds200.json` | `experiments/h1_run.py bsds200` | Table IX, Figure 9(c) |
| `h1_testimages40.json` | `experiments/h1_run.py testimages40` | Table IX |

## Checking a headline number

```python
import json, numpy as np
R = [r for r in json.load(open('results/h3_rows.json')) if r['d'] <= 0.4]
gain = np.array([r['psnr_mrs_d0'] - r['psnr_nvbmf'] for r in R])
print(f"{gain.mean():+.3f} dB over {len(gain)} conditions")     # +0.572 dB over 72 conditions
print("better in", int((gain > 0.005).sum()))                    # 65
```

Counts of better and worse conditions are taken at the precision of the printed tables, so a
difference below 0.005 dB is a tie. Without that convention the same data reads 70 better and 2
worse, because differences in the fourth decimal are counted as wins.

## Field names

`psnr_*`, `ssimw_*` (windowed SSIM, Wang et al. 2004), `ssimg_*` (whole-image SSIM),
`ief_*`, `rir_*` (residual impulse rate, per cent), for each of
`median3 median7 amf nvbmf las2 mrs mrs_d0 oracle stage1 stage2`.

`mrs` is the rule with the zero-lag clamp left in place; **`mrs_d0` is the corrected rule the paper
reports.** Both are stored so the effect of that one convention can be measured directly.

`agree` / `agree_d0` are the fraction of repaired pixels where the rule and the oracle make the same
decision; `frac_*` are the fractions of repaired pixels refined.
