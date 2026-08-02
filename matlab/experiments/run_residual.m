function R = run_residual()
%RUN_RESIDUAL  Table VIII: residual impulse rate, mean over the six images.
%
%   The fraction of OUTPUT pixels that remain at exactly 0 or 255. Median
%   based filters emit impulses of their own at high density; a filter that
%   draws its replacement from a pixel identified as uncorrupted cannot.
%
%   The two-stage figures are small but not exactly zero because of the scene
%   content that legitimately occupies the extremes (last column of Table I).

root = setup_paths();
imgs = test_images(root);
dens = [0.5 0.6 0.7 0.8 0.9];

R = zeros(numel(dens), 4);
for di = 1:numel(dens)
    acc = zeros(1, 4);
    for ii = 1:numel(imgs)
        img = load_gray(imgs(ii).path);
        [noisy, mask] = add_spn(img, dens(di), 1);
        acc(1) = acc(1) + residual_impulse_rate(median_filter_k(noisy, 3));
        acc(2) = acc(2) + residual_impulse_rate(median_filter_k(noisy, 7));
        acc(3) = acc(3) + residual_impulse_rate(adaptive_median_filter(noisy, 7));
        acc(4) = acc(4) + residual_impulse_rate(nvbmf(noisy, mask));
    end
    R(di,:) = acc / numel(imgs);
    fprintf('d=%.1f  med3=%6.2f  med7=%6.2f  amf=%6.2f  nvbmf/mrs=%5.2f\n', ...
            dens(di), R(di,1), R(di,2), R(di,3), R(di,4));
end
fprintf('Paper Table VIII, d=0.9: 75.72 / 48.38 / 31.83 / 0.01\n');
end
