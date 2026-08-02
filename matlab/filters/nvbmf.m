function out = nvbmf(noisy, mask)
%NVBMF  Nearest Value Based Mean Filter, the published two-stage filter [1].
%
%   Stage 1 always runs. Stage 2 (the 3x3 non-zero mean refinement) runs only
%   when the estimated noise level exceeds the fixed constant tau = 0.45,
%   which is Eq. (1) of the paper:
%
%       X(p) = X2(p)  if  dhat > tau,   otherwise  X(p) = X1(p)
%
%   One constant is shared by every image, which is exactly the design choice
%   that MRS_FILTER replaces.
%
%   [1] B. Turan, "A new approach for SPN removal: nearest value based mean
%       filter," PeerJ Comput. Sci., vol. 8, p. e1160, 2022.

if nargin < 2 || isempty(mask), mask = is_noise(noisy); end
mask = logical(mask);

s1 = stage1_with_sources(double(noisy), mask);

out = s1;
if mean(mask(:)) > 0.45
    s2 = box3_nonzero_mean(s1);
    out(mask) = s2(mask);
end
out = uint8(min(max(round_half_even(out), 0), 255));
end
