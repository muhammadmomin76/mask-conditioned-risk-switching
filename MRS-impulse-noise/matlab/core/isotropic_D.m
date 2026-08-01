function curve = isotropic_D(lags, Dvals)
%ISOTROPIC_D  Average directional lag estimates into an isotropic D(r) curve.
%
%   CURVE = ISOTROPIC_D(LAGS, DVALS) groups the per-lag estimates by squared
%   distance and averages them, giving the isotropic structure function that
%   Section III-C assumes. CURVE is a struct with fields
%       .r   ascending vector of distances
%       .D   the corresponding structure-function values
%   Evaluate it with EVAL_D.
%
%   Reference implementation: mrs_core.isotropic + mrs_core.Dcurve.

d2 = lags(:,1).^2 + lags(:,2).^2;
ok = isfinite(Dvals(:));
d2 = d2(ok);  Dv = Dvals(ok);

uniq = unique(d2);
rr = zeros(numel(uniq), 1);
DD = zeros(numel(uniq), 1);
for k = 1:numel(uniq)
    rr(k) = sqrt(uniq(k));
    DD(k) = mean(Dv(d2 == uniq(k)));
end

curve = struct('r', rr, 'D', DD, 'local', false);
end
