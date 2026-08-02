function p = psnr_db(X, Y)
%PSNR_DB  Peak signal-to-noise ratio in dB, computed from its defining equation.
%
%       PSNR = 10 * log10(255^2 / MSE)
%
%   255 = L-1 for 8-bit images. Identical images give MSE 0; the reference
%   implementation reports 99 dB in that case rather than +Inf so that the
%   sweep tables stay finite, and this port does the same.
%
%   No toolbox metric is called anywhere.

m = mse_metric(X, Y);
if m == 0
    p = 99.0;
else
    p = 10 * log10(255^2 / m);
end
end
