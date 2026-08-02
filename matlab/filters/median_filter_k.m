function out = median_filter_k(image, k)
%MEDIAN_FILTER_K  Standard k-by-k median filter (Gonzalez & Woods 3e, 5.3.2).
%
%   The median is obtained by SORTING the stacked window values and taking
%   the middle element, so no toolbox median or convolution routine is
%   called. The image is padded by edge replication, so border windows stay
%   full size.
%
%   This is the classical answer to impulse noise, and the one that fails
%   once the impulses outnumber the genuine values in a window: the middle of
%   the sorted list is then itself an impulse. That failure is what Table II
%   and Table VIII of the paper measure.
%
%   Reference implementation: baselines.median_filter.

if nargin < 2, k = 3; end
if mod(k, 2) == 0 || k < 3
    error('median_filter_k:badWindow', 'k must be odd and at least 3, got %d', k);
end

stack = window_stack(image, k);
ordered = sort(stack, 3);
out = uint8(min(max(round_half_even(ordered(:,:, floor(k*k/2) + 1)), 0), 255));
end
