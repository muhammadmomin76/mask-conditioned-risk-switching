function out = box3_nonzero_mean(a)
%BOX3_NONZERO_MEAN  Mean of the NON-ZERO values in every 3x3 window.
%
%   Stage 2 of the two-stage filter, Eq. (7)-(9) of [1]: zeros are excluded
%   from both the sum and the count, and border windows average only their
%   in-image neighbours. Built from nine shifted copies of the zero-padded
%   image, i.e. plain array arithmetic with no convolution routine.
%
%   Reference implementation: mrs_run.box3_nonzero_mean.

a = double(a);
[H, W] = size(a);

ap = zeros(H + 2, W + 2);
ap(2:end-1, 2:end-1) = a;
zp = zeros(H + 2, W + 2);
zp(2:end-1, 2:end-1) = double(a ~= 0);

wsum = zeros(H, W);
wcnt = zeros(H, W);
for di = 0:2
    for dj = 0:2
        win = ap(di+1:di+H, dj+1:dj+W);
        msk = zp(di+1:di+H, dj+1:dj+W);
        wsum = wsum + win .* msk;       % non-zero values only
        wcnt = wcnt + msk;              % how many were non-zero
    end
end

out = a;
has = wcnt > 0;
out(has) = wsum(has) ./ wcnt(has);
end
