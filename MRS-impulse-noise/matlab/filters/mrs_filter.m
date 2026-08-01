function [out, info] = mrs_filter(noisy, mask, opts)
%MRS_FILTER  Mask-Conditioned Risk Switching, the method of this paper.
%
%   OUT = MRS_FILTER(NOISY) restores a salt-and-pepper corrupted grayscale
%   image. The noise mask is produced by the detector of [1] (a pixel is
%   flagged when its value is exactly 0 or 255).
%
%   OUT = MRS_FILTER(NOISY, MASK) uses a mask supplied by the caller. Passing
%   the true corruption mask is what the experiments of Section V do, so that
%   the switching rule is measured and not the detector.
%
%   OUT = MRS_FILTER(NOISY, MASK, OPTS) accepts the fields
%       .useLocalD  estimate D in a window around each pixel   (default false)
%       .radius     half-width of that window                  (default 16)
%       .hybrid     keep the published global rule above its
%                   0.45 cutoff, as in Section III-E           (default true)
%
%   [OUT, INFO] = ... also returns the intermediate quantities:
%       .s1, .s2      the two candidate restorations
%       .V1, .V2      the two predicted risks, Eq. (7) and Eq. (6)+(8)
%       .refined      logical map of the pixels that were refined
%       .D            the estimated structure function
%
%   The decision is Eq. (10): refine a repaired pixel when its predicted
%   refined error is below its predicted unrefined error. No threshold and no
%   noise density enter it.
%
%   Reference implementation: mrs_run.run_one (the psnr_hyb configuration).

if nargin < 2 || isempty(mask), mask = is_noise(noisy); end
if nargin < 3, opts = struct(); end
if ~isfield(opts, 'useLocalD'), opts.useLocalD = false; end
if ~isfield(opts, 'radius'),    opts.radius    = 16;    end
if ~isfield(opts, 'hybrid'),    opts.hybrid    = true;  end

x     = double(noisy);
mask  = logical(mask);
surv  = ~mask;
rings = distance_rings(5);
lags  = mrs_lags();

% ---- the two candidate restorations ------------------------------------
[s1, sdi, sdj, ringIdx, ringM] = stage1_with_sources(x, mask, rings);
s2 = box3_nonzero_mean(s1);

% ---- the structure function, from surviving pairs only ------------------
Dg = isotropic_D(lags, structure_function_global(x, surv, lags));
if opts.useLocalD
    curve = structure_function_local(x, surv, lags, opts.radius, Dg);
else
    curve = Dg;
end

% ---- the two predicted risks and the comparison -------------------------
V1 = risk_stage1(mask, ringIdx, ringM, curve, rings);
V2 = risk_stage2(mask, sdi, sdj, curve);

refined = mask & (V2 < V1);                       % Eq. (10)
if opts.hybrid
    % Above the published 0.45 cutoff the decision approaches chance
    % (Table VII), so the published global rule is kept there unchanged.
    if mean(mask(:)) > 0.45
        refined = mask;
    end
end

out = s1;
out(refined) = s2(refined);
out = uint8(min(max(round_half_even(out), 0), 255));

if nargout > 1
    info = struct('s1', s1, 's2', s2, 'V1', V1, 'V2', V2, ...
                  'refined', refined, 'D', curve);
end
end
