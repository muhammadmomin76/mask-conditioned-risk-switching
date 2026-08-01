function mrs_demo(imageName, density, seed)
%MRS_DEMO  One-image demonstration of the proposed method.
%
%   MRS_DEMO                       house at 40% density, seed 1
%   MRS_DEMO(NAME, DENSITY, SEED)  any of the six test images
%
%   Corrupts the image, restores it with the 3x3 median, the 7x7 median, the
%   adaptive median, the published two-stage filter and the proposed MRS
%   rule, prints PSNR / SSIM / residual impulse rate for each, and saves a
%   comparison figure and the map of the pixels MRS chose to refine into
%   results/.
%
%   40% density is deliberately BELOW the published 0.45 cutoff, which is
%   where the published rule declines to refine at all and MRS does its work.

root = setup_paths();
if nargin < 1 || isempty(imageName), imageName = 'house'; end
if nargin < 2 || isempty(density),   density = 0.4;       end
if nargin < 3 || isempty(seed),      seed = 1;            end

imgs = test_images(root);
hit = find(strcmp({imgs.name}, imageName), 1);
if isempty(hit)
    error('mrs_demo:unknownImage', 'unknown image "%s"; known: %s', ...
          imageName, strjoin({imgs.name}, ', '));
end

clean = load_gray(imgs(hit).path);
[noisy, mask] = add_spn(clean, density, seed);

outs = struct('name', {}, 'img', {});
outs(1) = struct('name', 'Median 3x3',     'img', median_filter_k(noisy, 3));
outs(2) = struct('name', 'Median 7x7',     'img', median_filter_k(noisy, 7));
outs(3) = struct('name', 'Adaptive median','img', adaptive_median_filter(noisy, 7));
outs(4) = struct('name', 'NVBMF [1]',      'img', nvbmf(noisy, mask));
outs(5) = struct('name', 'Local density',  'img', local_density_filter(noisy, mask));
[mrsOut, info] = mrs_filter(noisy, mask);
outs(6) = struct('name', 'MRS (ours)',     'img', mrsOut);

fprintf('\n%s at %.0f%% impulse noise (seed %d)\n', imageName, 100*density, seed);
fprintf('  corrupted input          PSNR %6.2f dB  SSIM %.4f\n', ...
        psnr_db(clean, noisy), ssim_global(clean, noisy));
fprintf('  %-18s %8s %8s %10s\n', 'method', 'PSNR', 'SSIM', 'residual%');
for k = 1:numel(outs)
    fprintf('  %-18s %8.2f %8.4f %10.2f\n', outs(k).name, ...
            psnr_db(clean, outs(k).img), ssim_global(clean, outs(k).img), ...
            residual_impulse_rate(outs(k).img));
end
fprintf('  MRS refined %.1f%% of the repaired pixels; the published rule refined %.0f%%\n', ...
        100 * sum(info.refined(:)) / max(sum(mask(:)), 1), 100 * (mean(mask(:)) > 0.45));

% ---- figures ------------------------------------------------------------
resdir = fullfile(root, 'results');
if ~exist(resdir, 'dir'), mkdir(resdir); end

f = figure('Visible', 'off', 'Position', [100 100 1200 800]);
tiles = [{struct('name','Original','img',clean)}, ...
         {struct('name',sprintf('Corrupted %.0f%%',100*density),'img',noisy)}, ...
         arrayfun(@(s) {s}, outs)];
for k = 1:numel(tiles)
    subplot(2, 4, k);
    imshow(tiles{k}.img, [0 255]);
    if k <= 2
        title(tiles{k}.name);
    else
        title(sprintf('%s  %.2f dB', tiles{k}.name, psnr_db(clean, tiles{k}.img)));
    end
end
print(f, fullfile(resdir, sprintf('demo_%s_%02d.png', imageName, round(100*density))), '-dpng', '-r150');
close(f);

f = figure('Visible', 'off', 'Position', [100 100 800 400]);
subplot(1,2,1); imshow(mask, [0 1]);         title('noise mask M');
subplot(1,2,2); imshow(info.refined, [0 1]); title('pixels MRS chose to refine');
print(f, fullfile(resdir, sprintf('decision_%s_%02d.png', imageName, round(100*density))), '-dpng', '-r150');
close(f);

fprintf('  figures -> results/demo_%s_%02d.png and results/decision_%s_%02d.png\n', ...
        imageName, round(100*density), imageName, round(100*density));
end
