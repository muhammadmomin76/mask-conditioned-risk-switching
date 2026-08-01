function y = round_half_even(x)
%ROUND_HALF_EVEN  Round to nearest, ties to even ("banker's rounding").
%
%   MATLAB's built-in ROUND breaks ties away from zero, so round(50.5) = 51.
%   The reference implementation quantises its output with NumPy's np.round,
%   which breaks ties to the nearest EVEN integer, so np.round(50.5) = 50.
%
%   Stage-1 repairs are means of small integer sets, so exact halves are
%   common and the two conventions disagree on a few thousand pixels of a
%   512x512 image -- about 0.002 dB of PSNR. Using this function at the one
%   place where the output is quantised makes the port reproduce the
%   published numbers exactly rather than approximately.

f = floor(x);
d = x - f;
y = f + double(d > 0.5) + double(d == 0.5) .* double(mod(f, 2) ~= 0);
end
