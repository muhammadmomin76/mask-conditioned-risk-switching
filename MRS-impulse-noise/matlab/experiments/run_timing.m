function T = run_timing()
%RUN_TIMING  Table IX: running time on a 512x512 image at 50% density.
%
%   Median of three runs on a single core. The proposed rule costs about
%   twelve times the published filter in the reference implementation; the
%   estimation of the structure function is not the reason, the pairwise terms
%   of the two extension variances are. Absolute numbers depend on the
%   machine and on MATLAB versus Octave, so compare the RATIOS with Table IX
%   rather than the milliseconds.

root = setup_paths();
img = load_gray(fullfile(root, 'images', 'benchmark', 'cameraman.png'));
[noisy, mask] = add_spn(img, 0.5, 1);

names = {'Median 3x3', 'Median 7x7', 'Adaptive median', 'NVBMF [1]', ...
         'Local density rule', 'MRS (ours)'};
fns = { @() median_filter_k(noisy, 3), ...
        @() median_filter_k(noisy, 7), ...
        @() adaptive_median_filter(noisy, 7), ...
        @() nvbmf(noisy, mask), ...
        @() local_density_filter(noisy, mask), ...
        @() mrs_filter(noisy, mask) };

T = zeros(1, numel(fns));
fprintf('\n--- Table IX: running time, 512x512 at 50%% density -----------\n');
for k = 1:numel(fns)
    t = zeros(1, 3);
    for rep = 1:3
        tic; fns{k}(); t(rep) = toc * 1000;
    end
    T(k) = median(t);
    fprintf('%-22s %8.0f ms\n', names{k}, T(k));
end
fprintf('MRS / NVBMF cost ratio: %.1fx  (paper: 12x)\n', T(6) / T(4));
end
