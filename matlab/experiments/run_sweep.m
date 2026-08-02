function rows = run_sweep(densities, seeds, useLocalD, outFile)
%RUN_SWEEP  The main experiment: six images x nine densities x three seeds.
%
%   ROWS = RUN_SWEEP() runs the full grid of Section IV and writes
%   results/sweep_global.csv. This is the table behind Table III, Table V,
%   Table VI and Table VII.
%
%   ROWS = RUN_SWEEP(DENSITIES, SEEDS, USELOCALD, OUTFILE) restricts or
%   redirects the run. The headline numbers of the paper come from the 24
%   conditions BELOW the published cutoff, i.e.
%
%       run_sweep([0.1 0.2 0.3 0.4], [1 2 3])
%
%   Use SUMMARISE_SWEEP on the result to print the paper's numbers.

root = setup_paths();
if nargin < 1 || isempty(densities), densities = 0.1:0.1:0.9; end
if nargin < 2 || isempty(seeds),     seeds = [1 2 3];         end
if nargin < 3 || isempty(useLocalD), useLocalD = false;       end
if nargin < 4 || isempty(outFile)
    tag = 'global'; if useLocalD, tag = 'local'; end
    outFile = fullfile(root, 'results', ['sweep_' tag '.csv']);
end

imgs = test_images(root);
rows = [];
t0 = tic;

for ii = 1:numel(imgs)
    img = load_gray(imgs(ii).path);
    for d = densities(:)'
        for s = seeds(:)'
            r = run_condition(img, d, s, useLocalD);
            r.image = imgs(ii).name;
            rows = [rows; r]; %#ok<AGROW>
            fprintf('%-20s d=%.1f s=%d  fixed=%6.2f local=%6.2f mrs=%6.2f oracle=%6.2f agree=%.3f\n', ...
                    r.image, d, s, r.psnr_fixed, r.psnr_las2, r.psnr_hyb, r.psnr_oracle, r.agree);
        end
    end
end

write_rows_csv(rows, outFile);
fprintf('\n%d conditions in %.1f s  ->  %s\n', numel(rows), toc(t0), outFile);
end
