function v = eval_D(curve, r)
%EVAL_D  Evaluate a structure-function curve at distance(s) R.
%
%   Linear interpolation between the estimated lags, clamped to the end
%   values outside the estimated range. This matches NumPy's np.interp,
%   which the reference implementation uses (MATLAB's INTERP1 would return
%   NaN outside the range instead, so R is clamped explicitly first).
%
%   Handles both curve kinds:
%     * global curve  (curve.local == false): .r / .D are vectors
%     * local  curve  (curve.local == true) : .D is H-by-W-by-n_r, so the
%       interpolation is done per pixel and R must be an H-by-W array or a
%       scalar.
%
%   Reference implementation: mrs_core.Dcurve.__call__ and mrs_run.LocalD.

if ~curve.local
    rc = min(max(double(r), curve.r(1)), curve.r(end));
    if numel(curve.r) == 1
        v = repmat(curve.D(1), size(rc));
    else
        v = interp1(curve.r, curve.D, rc, 'linear');
    end
    v(double(r) <= 0) = 0;   % D is zero at zero lag; see note below
    return
end

% ---- per-pixel (local) structure function -------------------------------
sz = [size(curve.D, 1), size(curve.D, 2)];
r  = double(r);
if isscalar(r), r = repmat(r, sz); end

nr = numel(curve.r);
% index of the bracketing lag below r, clamped so idx+1 is always valid
idx = ones(sz);
for k = 2:nr
    idx(r >= curve.r(k)) = k;
end
idx = min(max(idx, 1), nr - 1);

r0 = curve.r(idx);
r1 = curve.r(idx + 1);
w  = min(max((r - r0) ./ max(r1 - r0, 1e-9), 0), 1);

[H, W] = deal(sz(1), sz(2));
lin  = (1:H*W)';                       % column-major pixel index
base = (idx(:) - 1) * (H * W);
d0 = curve.D(base + lin);
d1 = curve.D(base + lin + H * W);

v = reshape(d0 .* (1 - w(:)) + d1 .* w(:), sz);
v(r <= 0) = 0;               % D is zero at zero lag; see note below
end

% ZERO-LAG NOTE
% -------------
% The pairwise term of Eqn (6) sums over the ORDERED pairs of the source set,
% so it includes the m terms in which the two members are the same source. The
% lag is zero for those terms and D(0) = 0. Both branches above clamp to the
% smallest MEASURED lag, which would charge D(r_min) instead, deflating V1 by
% D(r_min)/(2*m1) and V2 by D(r_min)/18. Since m1 is small below the cutoff,
% that biases the comparison against refining. The two lines added above
% restore the convention the equation states.
