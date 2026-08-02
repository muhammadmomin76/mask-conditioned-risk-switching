"""
Comparison baselines, written from scratch.

The NVBMF paper argues that order-statistic filters break down at high impulse
density. That claim is the reason the paper exists, so this project measures it
instead of quoting it. Two baselines are implemented here:

    median_filter            the standard k x k median filter
                             (Gonzalez & Woods 3e, Section 5.3.2)
    adaptive_median_filter   the adaptive median filter that grows its window
                             (Gonzalez & Woods 3e, Section 5.3.3)

Both are built from numpy array arithmetic only. In particular the median is
obtained by SORTING the stacked window values and taking the middle element, so
no library median or convolution routine is called anywhere:

    np.sort(stack, axis=0)[k*k // 2]

The same window stack drives both filters, which keeps them fast enough to run
over the whole benchmark in a few seconds.
"""

import numpy as np


def _window_stack(image, k):
    """Stack the k*k neighbours of every pixel into one array.

    stack[w, i, j] is neighbour number w of the k x k window centred on (i, j).
    The image is padded by replicating its edge pixels, so border windows stay
    full size and no pixel is dropped. This is the vectorised equivalent of a
    nested loop over the window, written with plain array slicing.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {image.shape}")
    if k % 2 == 0 or k < 3:
        raise ValueError(f"window size k must be odd and at least 3, got {k}")
    r = k // 2
    padded = np.pad(image.astype(np.float64), r, mode="edge")
    H, W = image.shape
    return np.stack([padded[di:di + H, dj:dj + W]
                     for di in range(k) for dj in range(k)], axis=0)


def median_filter(image, k=3):
    """Standard k x k median filter (G&W Section 5.3.2).

    Sort each window and take the middle value. This is the classical answer to
    impulse noise and it works well while impulses are a minority of the window.
    Once they are the majority the middle of the sorted list is itself an
    impulse, which is exactly the failure this project measures.
    """
    ordered = np.sort(_window_stack(image, k), axis=0)
    return np.clip(np.round(ordered[(k * k) // 2]), 0, 255).astype(np.uint8)


def adaptive_median_filter(image, s_max=7):
    """Adaptive median filter (G&W Section 5.3.3), vectorised over the image.

    For each pixel the algorithm runs two levels:

      Level A: compare the window median z_med against the window minimum and
               maximum. If z_min < z_med < z_max the median is a real value, so
               go to level B. Otherwise the median is itself an impulse, so grow
               the window by 2 and try level A again, up to s_max.
      Level B: if z_min < z_xy < z_max the centre pixel is not an impulse, so
               keep it unchanged. Otherwise output z_med.

    Rather than looping per pixel, every pixel is carried through the same
    window sizes at once and a boolean mask records which pixels are still
    undecided. Pixels that never satisfy level A by s_max take the largest
    median available, which is what the textbook specifies.
    """
    out = image.astype(np.float64).copy()
    undecided = np.ones(image.shape, bool)
    z_xy = image.astype(np.float64)

    for k in range(3, s_max + 1, 2):
        ordered = np.sort(_window_stack(image, k), axis=0)
        z_min = ordered[0]
        z_med = ordered[(k * k) // 2]
        z_max = ordered[-1]

        level_a_ok = (z_med > z_min) & (z_med < z_max)   # median is not an impulse
        decide_now = undecided & level_a_ok
        keep_centre = decide_now & (z_xy > z_min) & (z_xy < z_max)   # level B

        out[decide_now] = z_med[decide_now]              # default: output the median
        out[keep_centre] = z_xy[keep_centre]             # unless the centre is clean
        undecided &= ~level_a_ok
        if not undecided.any():
            break

    if undecided.any():                                  # window grew to s_max and gave up
        ordered = np.sort(_window_stack(image, s_max), axis=0)
        biggest_median = ordered[(s_max * s_max) // 2]
        out[undecided] = biggest_median[undecided]

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Self-check: run with  python src/baselines.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # A single pepper pixel in a flat field is the textbook case: the 3x3 median
    # removes it completely.
    flat = np.full((7, 7), 100, np.uint8)
    spike = flat.copy(); spike[3, 3] = 0
    assert median_filter(spike, 3)[3, 3] == 100
    assert adaptive_median_filter(spike, 7)[3, 3] == 100

    # Hand-computed median of a 3x3 window, checked against a sorted list.
    patch = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], np.uint8)
    assert median_filter(patch, 3)[1, 1] == 50

    # Edge replication keeps border windows full, so a flat image is unchanged.
    assert np.array_equal(median_filter(flat, 3), flat)
    assert np.array_equal(median_filter(flat, 7), flat)

    # The failure the paper is about: when impulses outnumber real values, the
    # middle of the sorted window is itself an impulse. Five peppers and three
    # salt grains around one real value of 120 give a median of 0, not 120.
    window = np.sort(np.array([0, 0, 0, 0, 0, 120, 255, 255, 255], np.float64))
    assert window[4] == 0

    # The adaptive filter keeps a clean centre pixel instead of blurring it.
    ramp = np.tile(np.arange(1, 8, dtype=np.uint8) * 30, (7, 1))
    assert adaptive_median_filter(ramp, 7)[3, 3] == ramp[3, 3]

    print("OK: median and adaptive-median self-checks passed.")
