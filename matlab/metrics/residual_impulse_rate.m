function r = residual_impulse_rate(out)
%RESIDUAL_IMPULSE_RATE  Percentage of output pixels still at exactly 0 or 255.
%
%   The reference-free deployment check of Section V-D. A filter that draws
%   its replacement from a pixel identified as uncorrupted cannot emit an
%   impulse of its own; a median-based filter can, and does, at high density.
%
%   It carries a positive bias from scene content that legitimately occupies
%   the extremes (the last column of Table I), which is why the two-stage
%   figures are small but not exactly zero. It can also be defeated by
%   clipping the output range, so it is reported as a check and not as a new
%   measure.

out = double(out);
r = 100 * mean((out(:) == 0) | (out(:) == 255));
end
