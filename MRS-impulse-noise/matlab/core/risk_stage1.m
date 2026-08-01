function V1 = risk_stage1(mask, ringIdx, ringM, curve, rings)
%RISK_STAGE1  Predicted squared error of the Stage-1 repair, Eq. (7).
%
%   All Stage-1 donors of a pixel lie at one common distance r, so the
%   extension variance of Eq. (6) collapses to
%
%       V1(p) = D(r) - (1 / 2m^2) * sum_{s,t in S1} D(|s - t|)
%
%   The first term is the error of a single donor at distance r; the second is
%   the reduction earned by averaging the m tied donors. Both source set and
%   m come from the noise mask alone, so no reference image is used.
%
%   Reference implementation: mrs_core.risk_stage1 + mrs_core._ring_pair_term.

if nargin < 5, rings = distance_rings(5); end

[H, W] = size(mask);
surv = double(~mask);

% ---- distance r of the winning ring, per pixel --------------------------
r = zeros(H, W);
for k = 1:numel(rings)
    sel = (ringIdx == k);
    if any(sel(:))
        r(sel) = sqrt(rings(k).d2);
    end
end
first = eval_D(curve, r);

% ---- pairwise term ------------------------------------------------------
acc = zeros(H, W);
for k = 1:numel(rings)
    sel = (ringIdx == k) & mask;
    if ~any(sel(:)), continue; end

    offs = rings(k).offs;
    n = size(offs, 1);
    % survivor indicator shifted to each offset of this ring, computed once
    SM = cell(1, n);
    for a = 1:n
        SM{a} = shift_image(surv, offs(a,1), offs(a,2));
    end

    tot = zeros(H, W);
    for a = 1:n
        for b = 1:n
            h = hypot(offs(a,1) - offs(b,1), offs(a,2) - offs(b,2));
            tot = tot + SM{a} .* SM{b} .* eval_D(curve, h);
        end
    end
    acc(sel) = tot(sel);
end

m2 = max(ringM, 1) .^ 2;
V1 = max(first - acc ./ (2 * m2), 1e-9);
end
