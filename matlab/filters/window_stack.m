function stack = window_stack(image, k)
%WINDOW_STACK  Stack the k*k neighbours of every pixel along the 3rd dimension.
%
%   STACK(i,j,w) is neighbour number w of the k-by-k window centred on (i,j).
%   The image is padded by replicating its edge pixels so border windows stay
%   full size and no pixel is dropped. This is the vectorised equivalent of a
%   nested loop over the window, written with plain array slicing.
%
%   Reference implementation: baselines._window_stack.

if ndims(image) ~= 2 %#ok<ISMAT>
    error('window_stack:notGray', 'expected a 2-D grayscale array');
end

r = floor(k / 2);
[H, W] = size(image);

% edge replication
ri = min(max((1:H+2*r) - r, 1), H);
ci = min(max((1:W+2*r) - r, 1), W);
padded = double(image(ri, ci));

stack = zeros(H, W, k*k);
w = 0;
for di = 1:k
    for dj = 1:k
        w = w + 1;
        stack(:,:,w) = padded(di:di+H-1, dj:dj+W-1);
    end
end
end
