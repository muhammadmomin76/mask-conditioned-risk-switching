function out = shift_image(a, di, dj, fill)
%SHIFT_IMAGE  Shift an array so that out(i,j) = a(i+di, j+dj), zero-padded.
%
%   Positions that fall outside the array are filled with FILL (default 0).
%   This is the array-shift primitive that every windowed operation in the
%   MRS pipeline is built from; no toolbox filtering routine is used
%   anywhere in this implementation.
%
%   Reference implementation: mrs_core.shift (Python/NumPy).

if nargin < 4, fill = 0; end

[H, W] = size(a);
out = repmat(cast(fill, class(a)), H, W);

% source rows/cols in A, destination rows/cols in OUT
si0 = max(1, 1 + di);   si1 = min(H, H + di);
di0 = max(1, 1 - di);   di1 = min(H, H - di);
sj0 = max(1, 1 + dj);   sj1 = min(W, W + dj);
dj0 = max(1, 1 - dj);   dj1 = min(W, W - dj);

if si1 >= si0 && sj1 >= sj0
    out(di0:di1, dj0:dj1) = a(si0:si1, sj0:sj1);
end
end
