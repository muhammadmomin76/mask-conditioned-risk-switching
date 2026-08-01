function out = local_density_filter(noisy, mask, radius)
%LOCAL_DENSITY_FILTER  The local-density switching baseline of Section IV-A.
%
%   The natural local counterpart of [1] in the sense of [5] and [6]: the same
%   0.45 level is applied to the corruption fraction of a 5x5 neighbourhood,
%   Eq. (2),
%
%       dhat(p) = (1 / |N(p)|) * sum_{x in N(p)} M(x)
%
%   so that MRS is compared against a fair LOCAL baseline and not only a
%   global one. The global rule is kept as a floor, exactly as in the
%   published filter.
%
%   A fraction describes the mask alone and says nothing about the image
%   below it, which is the limitation Section V-B measures: this rule gains
%   0.123 dB where MRS gains 0.483 dB.
%
%   Reference implementation: mrs_run.run_one (the psnr_las2 configuration).

if nargin < 2 || isempty(mask), mask = is_noise(noisy); end
if nargin < 3, radius = 2; end                 % 5x5 neighbourhood
mask = logical(mask);

[s1] = stage1_with_sources(double(noisy), mask);
s2   = box3_nonzero_mean(s1);

localD  = box_sum(double(mask), radius) ./ box_sum(ones(size(mask)), radius);
globalD = mean(mask(:));

refined = mask & ((localD > 0.45) | (globalD > 0.45));

out = s1;
out(refined) = s2(refined);
out = uint8(min(max(round_half_even(out), 0), 255));
end
