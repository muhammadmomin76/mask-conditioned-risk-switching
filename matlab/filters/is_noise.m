function m = is_noise(image)
%IS_NOISE  The salt-and-pepper detector of [1].
%
%   A pixel is flagged when its value is exactly 0 or 255. The small number of
%   clean pixels that legitimately take these values is treated as part of the
%   mask, which is the known weakness of this detector (Section V-F) and the
%   reason Table I reports the percentage of extreme-valued pixels of each
%   clean image.

image = double(image);
m = (image == 0) | (image == 255);
end
