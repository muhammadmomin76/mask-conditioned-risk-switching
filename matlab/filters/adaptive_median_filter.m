function out = adaptive_median_filter(image, sMax)
%ADAPTIVE_MEDIAN_FILTER  Adaptive median filter (Gonzalez & Woods 3e, 5.3.3).
%
%   For each pixel:
%     Level A: compare the window median zMed with the window min and max. If
%              zMin < zMed < zMax the median is a real value, go to level B.
%              Otherwise the median is itself an impulse, so grow the window
%              by 2 and try level A again, up to SMAX.
%     Level B: if zMin < zXY < zMax the centre pixel is not an impulse, so
%              keep it unchanged. Otherwise output zMed.
%
%   Vectorised: every pixel is carried through the same window sizes at once
%   and a logical mask records which pixels are still undecided. Pixels that
%   never satisfy level A by SMAX take the largest median available, which is
%   what the textbook specifies.
%
%   Reference implementation: baselines.adaptive_median_filter.

if nargin < 2, sMax = 7; end

out = double(image);
undecided = true(size(image));
zXY = double(image);

for k = 3:2:sMax
    ordered = sort(window_stack(image, k), 3);
    zMin = ordered(:,:,1);
    zMed = ordered(:,:, floor(k*k/2) + 1);
    zMax = ordered(:,:,end);

    levelAok  = (zMed > zMin) & (zMed < zMax);
    decideNow = undecided & levelAok;
    keepCentre = decideNow & (zXY > zMin) & (zXY < zMax);

    out(decideNow)  = zMed(decideNow);       % default: output the median
    out(keepCentre) = zXY(keepCentre);       % unless the centre is clean
    undecided = undecided & ~levelAok;
    if ~any(undecided(:)), break; end
end

if any(undecided(:))                          % window grew to sMax and gave up
    ordered = sort(window_stack(image, sMax), 3);
    biggest = ordered(:,:, floor(sMax*sMax/2) + 1);
    out(undecided) = biggest(undecided);
end

out = uint8(min(max(round_half_even(out), 0), 255));
end
