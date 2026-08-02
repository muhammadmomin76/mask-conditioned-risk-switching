function s = box_sum(a, r)
%BOX_SUM  Sum over a (2r+1)x(2r+1) window at every pixel, via a summed-area table.
%
%   Windows are clipped at the image border (they are NOT padded), so the
%   count of contributing pixels shrinks at the edges. Dividing BOX_SUM(A,r)
%   by BOX_SUM(ONES,r) therefore gives a correctly normalised local mean.
%
%   Reference implementation: mrs_core._box.

[H, W] = size(a);
ii = zeros(H + 1, W + 1);
ii(2:end, 2:end) = cumsum(cumsum(double(a), 1), 2);

% The window of the 1-based pixel k covers rows [k-r, k+r] clipped to the
% image, i.e. summed-area rows [k-1-r, k+r) in 0-based terms. The trailing
% "+ 1" converts those 0-based summed-area offsets into MATLAB indices.
i0 = min(max((1:H)' - 1 - r, 0), H) + 1;
i1 = min(max((1:H)' + r,     0), H) + 1;
j0 = min(max((1:W)  - 1 - r, 0), W) + 1;
j1 = min(max((1:W)  + r,     0), W) + 1;

s = ii(i1, j1) - ii(i0, j1) - ii(i1, j0) + ii(i0, j0);
end
