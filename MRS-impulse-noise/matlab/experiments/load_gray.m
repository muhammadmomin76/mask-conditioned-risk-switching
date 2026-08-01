function img = load_gray(path)
%LOAD_GRAY  Read an image file and return an 8-bit grayscale plane.
%
%   IMREAD is used for file I/O only. Every filter and every metric in this
%   project is implemented from its defining equations; no toolbox filtering
%   or metric routine is called on any measured quantity.

a = imread(path);
if ndims(a) == 3 %#ok<ISMAT>
    % ITU-R BT.601 luma, the same conversion PIL's "L" mode applies
    a = double(a);
    a = 0.299*a(:,:,1) + 0.587*a(:,:,2) + 0.114*a(:,:,3);
    a = uint8(floor(a));
end
img = uint8(a);
end
