function [noisy, mask] = add_spn(image, density, seed)
%ADD_SPN  Salt-and-pepper corruption, Eq. (11) of the paper.
%
%       Y(p) = 255 * [ v(p) >= 0.5 ]   if u(p) < d
%       Y(p) = X(p)                    otherwise
%
%   u and v are independent uniform variables on the unit interval drawn for
%   every pixel, so salt and pepper are equally likely and the achieved
%   density is binomial around D rather than exactly D. The positions for
%   which u(p) < d form the mask M of Section III-A, which is returned so
%   that the experiments can measure the switching rule rather than the
%   detector.
%
%   The seed is fixed, so all filters receive an identical corrupted input.
%
%   Reference implementation: mrs_run.add_spn.

if nargin < 3, seed = 1; end
if density < 0 || density > 1
    error('add_spn:badDensity', 'density must be in [0,1], got %g', density);
end

set_seed(seed);
u = rand(size(image));
v = rand(size(image));

mask  = u < density;
salt  = v >= 0.5;

noisy = double(image);
noisy(mask &  salt) = 255;
noisy(mask & ~salt) = 0;
noisy = uint8(noisy);
end
