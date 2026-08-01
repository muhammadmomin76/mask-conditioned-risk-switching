function rings = distance_rings(radius)
%DISTANCE_RINGS  Group window offsets into rings of equal squared distance.
%
%   RINGS = DISTANCE_RINGS(RADIUS) returns a struct array, ordered nearest
%   first, with fields
%       .d2    squared distance of the ring from the centre
%       .offs  N-by-2 list of (di,dj) offsets at exactly that distance
%
%   Two offsets with the same squared distance are exactly equidistant, so a
%   whole ring is the set of "tied-nearest" candidates of Stage 1. The centre
%   offset (0,0) is excluded.
%
%   Reference implementation: mrs_core.distance_rings (Python/NumPy).

if nargin < 1, radius = 5; end

di = (-radius:radius)';
[DI, DJ] = ndgrid(di, di);
DI = DI(:); DJ = DJ(:);
keep = ~(DI == 0 & DJ == 0);
DI = DI(keep); DJ = DJ(keep);

d2all = DI.^2 + DJ.^2;
uniq  = unique(d2all);          % unique() returns ascending order

rings = struct('d2', cell(1, numel(uniq)), 'offs', cell(1, numel(uniq)));
for k = 1:numel(uniq)
    sel = (d2all == uniq(k));
    rings(k).d2   = uniq(k);
    rings(k).offs = [DI(sel), DJ(sel)];
end
end
