function [s1, sdi, sdj, ringIdx, ringM] = stage1_with_sources(noisy, mask, rings)
%STAGE1_WITH_SOURCES  Vectorised NVBMF Stage 1, plus the source-set bookkeeping.
%
%   [S1, SDI, SDJ, RINGIDX, RINGM] = STAGE1_WITH_SOURCES(NOISY, MASK, RINGS)
%
%   Stage 1 of the two-stage filter of [1]: every flagged pixel is replaced by
%   the mean of the surviving pixels at the SMALLEST distance at which any of
%   them survive (ties at that distance are averaged). Surviving pixels are
%   left untouched.
%
%   The extra outputs are what makes the risk of Section III-D computable:
%       SDI, SDJ   offset from the pixel to the CENTROID of its tied donors
%                  (0,0) for a surviving pixel
%       RINGIDX    index of the winning ring, i.e. which distance won  (-1
%                  where no ring won)
%       RINGM      number of tied donors in that ring, i.e. m in Eq. (7)
%
%   Both source sets of Section III-D are read from these, so they depend on
%   the noise mask alone and never on the reference image.
%
%   Reference implementation: mrs_core.stage1_with_sources.

if nargin < 3, rings = distance_rings(5); end

x    = double(noisy);
surv = double(~mask);
[H, W] = size(x);

val     = zeros(H, W);
done    = ~mask;                 % surviving pixels need no repair
ringIdx = -ones(H, W);
ringM   = zeros(H, W);
sdi     = zeros(H, W);
sdj     = zeros(H, W);

for k = 1:numel(rings)
    todo = ~done;
    if ~any(todo(:)), break; end

    offs = rings(k).offs;
    s = zeros(H, W); c = zeros(H, W);
    adi = zeros(H, W); adj = zeros(H, W);
    for t = 1:size(offs, 1)
        di = offs(t,1); dj = offs(t,2);
        sm = shift_image(surv, di, dj);     % 1 where the neighbour survived
        sv = shift_image(x,    di, dj);
        s   = s   + sv .* sm;
        c   = c   + sm;
        adi = adi + di * sm;
        adj = adj + dj * sm;
    end

    hit = todo & (c > 0);                   % first ring that contains a survivor
    if any(hit(:))
        val(hit)     = s(hit)   ./ c(hit);  % Eq. (6): mean of the tied-nearest
        ringIdx(hit) = k;
        ringM(hit)   = c(hit);
        sdi(hit)     = adi(hit) ./ c(hit);  % donor centroid offset
        sdj(hit)     = adj(hit) ./ c(hit);
        done = done | hit;
    end
end

% Whole 11x11 window corrupted (only happens near 100% density): fall back to
% the mean of the surviving pixels of the image.
left = ~done;
if any(left(:))
    if any(~mask(:))
        cm = mean(x(~mask));
    else
        cm = mean(x(:));
    end
    val(left)   = cm;
    ringM(left) = 1;
end

s1 = x;
s1(mask) = val(mask);
end
