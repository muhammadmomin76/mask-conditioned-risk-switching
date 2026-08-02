function D = structure_function_global(img, surv, lags)
%STRUCTURE_FUNCTION_GLOBAL  D(h) estimated from surviving pixel pairs only.
%
%   D = STRUCTURE_FUNCTION_GLOBAL(IMG, SURV, LAGS) implements Eq. (9) of the
%   paper,
%
%       Dhat(h) = sum_x Mbar(x) Mbar(x+h) [Y(x) - Y(x+h)]^2
%                 --------------------------------------------
%                        sum_x Mbar(x) Mbar(x+h)
%
%   where Mbar = 1 - M is the survivor indicator (SURV here). Because the
%   surviving pixels carry TRUE values, each surviving pair gives an exact
%   realisation of the squared difference at that lag, so no ground truth is
%   needed. LAGS is an N-by-2 list of (di,dj) offsets; D is the N-vector of
%   estimates, NaN where a lag has no surviving pair at all.
%
%   Reference implementation: mrs_core.structure_function_global.

x = double(img);
m = double(surv);
n = size(lags, 1);
D = nan(n, 1);

for k = 1:n
    di = lags(k, 1); dj = lags(k, 2);
    xs = shift_image(x, di, dj);
    ms = shift_image(m, di, dj);
    w  = m .* ms;                       % 1 only where BOTH members survived
    den = sum(w(:));
    if den > 0
        num  = sum(sum(w .* (x - xs).^2));
        D(k) = num / den;
    end
end
end
