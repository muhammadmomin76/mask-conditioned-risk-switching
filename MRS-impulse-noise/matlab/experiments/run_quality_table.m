function P = run_quality_table()
%RUN_QUALITY_TABLE  Table II: PSNR on cameraman across the density sweep.
%
%   Columns: 3x3 median, 7x7 median, adaptive median, NVBMF, MRS.
%   Each entry is the mean of three noise realisations.
%
%   The pattern the paper describes: the two-stage methods degrade most
%   slowly, and MRS adds to the published filter BELOW its cutoff and matches
%   it above, by construction (Section III-E).

root = setup_paths();
img = load_gray(fullfile(root, 'images', 'benchmark', 'cameraman.png'));
dens = [0.1 0.3 0.5 0.7 0.9];
seeds = [1 2 3];

P = zeros(numel(dens), 5);
fprintf('\n--- Table II: PSNR (dB) on cameraman, mean of 3 realisations ---\n');
fprintf('%-8s %9s %9s %9s %9s %9s\n', 'Density', 'Med.3x3', 'Med.7x7', 'AMF', 'NVBMF', 'MRS');
for di = 1:numel(dens)
    acc = zeros(1, 5);
    for s = seeds
        [noisy, mask] = add_spn(img, dens(di), s);
        acc(1) = acc(1) + psnr_db(img, median_filter_k(noisy, 3));
        acc(2) = acc(2) + psnr_db(img, median_filter_k(noisy, 7));
        acc(3) = acc(3) + psnr_db(img, adaptive_median_filter(noisy, 7));
        acc(4) = acc(4) + psnr_db(img, nvbmf(noisy, mask));
        acc(5) = acc(5) + psnr_db(img, mrs_filter(noisy, mask));
    end
    P(di,:) = acc / numel(seeds);
    fprintf('%-8.1f %9.2f %9.2f %9.2f %9.2f %9.2f\n', dens(di), P(di,:));
end
fprintf('Paper Table II, d=0.1: 34.45 27.32 39.57 45.85 46.04\n');
fprintf('Paper Table II, d=0.9:  6.28  8.13  9.93 26.34 26.34\n');
end
