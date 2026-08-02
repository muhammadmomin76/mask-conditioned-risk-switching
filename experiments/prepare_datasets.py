#!/usr/bin/env python3
"""
prepare_datasets.py  --  turn downloaded image folders into what H1 expects.

USE
---
    python3 prepare_datasets.py  <source-folder>  <output-name>  [--filter TEXT]  [--limit N]

    --filter TEXT   keep only files whose name contains TEXT
                    (needed for TESTIMAGES: use  --filter C00C00  to keep the 40
                     aligned base images and drop the shifted copies)
    --limit N       keep only the first N files, in alphabetical order

EXAMPLES
--------
    python3 prepare_datasets.py  BSR/BSDS500/data/images/train   bsds200
    python3 prepare_datasets.py  SAMPLING_8BIT_GRAY_1200x1200  testimages40 --filter C00C00 --limit 40
    python3 prepare_datasets.py  /Applications/MATLAB/toolbox/images/imdata  matlab20

WHAT IT DOES
------------
1. finds every image in the source folder (jpg, png, tif, bmp, pgm ...)
2. converts each to 8-bit GREYSCALE
3. saves it as PNG into  data/<output-name>/
4. does NOT resize anything -- the anchor paper uses the native sizes
5. skips anything that is not a real image, and reports what it skipped
6. writes data/<output-name>/MANIFEST.txt listing every file and its size,
   so the run can be reproduced exactly later

After running it for all three sets you should have

    data/bsds200/        200 png
    data/testimages40/    40 png
    data/matlab20/        20 png

then just run:   python3 h1_m7_benchmark.py
"""
import sys, os, glob
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from PIL import Image

EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.pgm', '.ppm', '.gif')


def main(src, name, keep=None, limit=None):
    out = os.path.join('data', name)
    os.makedirs(out, exist_ok=True)

    files = []
    for root, _, names in os.walk(src):
        for n in sorted(names):
            if n.lower().endswith(EXT):
                if keep and keep not in n:
                    continue
                files.append(os.path.join(root, n))
    files.sort()
    if limit:
        files = files[:limit]

    if not files:
        print(f"no images found under {src}")
        return

    kept, skipped = [], []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0] + '.png'
        try:
            im = Image.open(f)
            if getattr(im, 'n_frames', 1) > 1:
                im.seek(0)                     # multi-page tif: first page only
            g = im.convert('L')                # 8-bit greyscale
            if min(g.size) < 64:
                skipped.append((f, 'smaller than 64 px'))
                continue
            g.save(os.path.join(out, base))
            kept.append((base, g.size))
        except Exception as e:
            skipped.append((f, str(e)[:60]))

    with open(os.path.join(out, 'MANIFEST.txt'), 'w') as fh:
        fh.write(f"source: {src}\ncount: {len(kept)}\n\n")
        for b, s in kept:
            fh.write(f"{b}\t{s[0]}x{s[1]}\n")

    print(f"{name}: wrote {len(kept)} greyscale PNG into {out}/")
    if skipped:
        print(f"  skipped {len(skipped)}:")
        for f, why in skipped[:10]:
            print(f"    {os.path.basename(f)}  ({why})")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped)-10} more")


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 2:
        print(__doc__)
        sys.exit(1)
    keep = a[a.index('--filter') + 1] if '--filter' in a else None
    lim = int(a[a.index('--limit') + 1]) if '--limit' in a else None
    main(a[0], a[1], keep, lim)
