# Test images

All six are 8-bit greyscale PNG. They are the ground truth: the pipeline adds salt-and-pepper
noise to them at run time and scores the restoration against them. No noisy image is stored.

## `benchmark/` — the standard set

Four images long established in this literature, so that results are directly comparable with the
filter this work modifies. Source: the USC-SIPI image database, obtained through the
`standard-test-images-for-Image-Processing` collection
(https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing).
Decoded from their non-standard two-samples-per-pixel TIFFs and saved as greyscale PNG.
Long-standing free research use.

| File | Size |
|---|---|
| cameraman.png | 512 x 512 |
| house.png | 512 x 512 |
| mandrill.png | 512 x 512 |
| peppers.png | 512 x 512 |

## `custom/` — two images added for this study

Chosen to cover content the standard set does not contain: dense text edges, and painterly
texture with smooth gradients. **Both are unambiguously public domain.**

| File | Size | Source | Why |
|---|---|---|---|
| manuscript_beowulf.png | 323 x 512 | [Beowulf, first page](https://commons.wikimedia.org/wiki/File:Beowulf.firstpage.jpeg), Wikimedia Commons, PD-scan of a c. 10th-century manuscript | high-frequency text edges test edge preservation |
| field_vangogh.png | 512 x 403 | [Wheat Field with Cypresses, Van Gogh (d. 1890)](https://commons.wikimedia.org/wiki/File:Vincent_van_Gogh_-_Wheat_Field_with_Cypresses_-_Google_Art_Project.jpg), Wikimedia Commons, PD-art | dense brush texture and smooth gradients |

## Preprocessing

Colour to greyscale via Pillow on download; longer side capped at 512 pixels, aspect ratio
preserved. No contrast adjustment, sharpening or denoising, so the images are usable as ground
truth.

## Why the last column of Table I matters

The detector treats every pixel at exactly 0 or 255 as noise, so it cannot separate genuinely
extreme scene content from corruption. Table I of the paper reports what fraction of each clean
image already sits at those values. The manuscript scan is the extreme case at 0.27%, and that
quantity is why the residual impulse rate of the two-stage filters is small but not exactly zero.
