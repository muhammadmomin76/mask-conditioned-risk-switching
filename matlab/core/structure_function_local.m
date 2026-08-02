function curve = structure_function_local(img, surv, lags, radius, globalCurve)
%STRUCTURE_FUNCTION_LOCAL  Per-pixel D(r) from surviving pairs in a window.
%
%   CURVE = STRUCTURE_FUNCTION_LOCAL(IMG, SURV, LAGS, RADIUS, GLOBALCURVE)
%   estimates the structure function of Eq. (9) inside a
%   (2*RADIUS+1)-square window around every pixel, using surviving pairs
%   only, and averages the directional lags into an isotropic per-pixel
%   D(r). Where a window holds 8 or fewer surviving pairs at a given lag the
%   estimate is too noisy to use and the global curve is substituted.
%
%   CURVE.D is H-by-W-by-numel(CURVE.r); evaluate it with EVAL_D.
%
%   This is the "Local D (33x33)" configuration of Table VI (RADIUS = 16).
%
%   Reference implementation: mrs_core.structure_function_local + mrs_run.LocalD.

if nargin < 4, radius = 16; end

x = double(img);
m = double(surv);
[H, W] = size(x);

d2 = lags(:,1).^2 + lags(:,2).^2;
uniq = unique(d2);
nr = numel(uniq);

num = zeros(H, W, nr);
den = zeros(H, W, nr);

for k = 1:size(lags, 1)
    di = lags(k,1); dj = lags(k,2);
    xs = shift_image(x, di, dj);
    ms = shift_image(m, di, dj);
    w  = m .* ms;
    slot = find(uniq == d2(k), 1);
    num(:,:,slot) = num(:,:,slot) + box_sum(w .* (x - xs).^2, radius);
    den(:,:,slot) = den(:,:,slot) + box_sum(w, radius);
end

D = zeros(H, W, nr);
rr = sqrt(uniq);
for s = 1:nr
    localEst = num(:,:,s) ./ max(den(:,:,s), 1e-9);
    fallback = eval_D(globalCurve, rr(s));
    enough   = den(:,:,s) > 8;
    D(:,:,s) = localEst .* enough + fallback .* (~enough);
end

curve = struct('r', rr, 'D', D, 'local', true);
end
