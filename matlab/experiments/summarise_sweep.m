function S = summarise_sweep(rows)
%SUMMARISE_SWEEP  Reproduce the headline numbers of Section V-B from a sweep.
%
%   S = SUMMARISE_SWEEP(ROWS) reports, over the conditions BELOW the published
%   0.45 cutoff (densities 0.1 to 0.4, i.e. 24 image-density cells x 3 seeds
%   = 72 conditions):
%
%       mean gain of MRS over the fixed rule           paper: +0.483 dB
%       conditions in which MRS is better / worse      paper: 69 / 72,  1
%       largest single gain                            paper: +2.47 dB
%       mean gain of the local density rule            paper: +0.123 dB
%       local rule better / worse                      paper: 35,  7
%       MRS vs local: better, mean gain                paper: 67 / 72,  +0.360 dB
%       mean gain of the oracle                        paper: +1.764 dB
%       seed-to-seed std of the baseline               paper:  0.079 dB
%
%   Because the noise is drawn with a different random generator than the
%   reference implementation (MATLAB's Mersenne Twister rather than NumPy's
%   PCG64), individual realisations differ; the aggregate figures are what
%   should be compared, against the 0.079 dB seed-to-seed variation.

below = [rows.d] < 0.45;
r = rows(below);

fixed  = [r.psnr_fixed];
local  = [r.psnr_las2];
mrs    = [r.psnr_hyb];
oracle = [r.psnr_oracle];

gMrs    = mrs - fixed;
gLocal  = local - fixed;
gOracle = oracle - fixed;
gVs     = mrs - local;

S.n              = numel(r);
S.mrs_gain       = mean(gMrs);
S.mrs_better     = sum(gMrs > 1e-9);
S.mrs_worse      = sum(gMrs < -1e-9);
S.mrs_max_gain   = max(gMrs);
S.local_gain     = mean(gLocal);
S.local_better   = sum(gLocal > 1e-9);
S.local_worse    = sum(gLocal < -1e-9);
S.mrs_vs_local   = mean(gVs);
S.mrs_beats_local = sum(gVs > 1e-9);
S.oracle_gain    = mean(gOracle);

% seed-to-seed standard deviation of the baseline configuration
keys = strcat({r.image}, sprintf('|'), arrayfun(@(x) sprintf('%.1f', x), [r.d], 'UniformOutput', false));
u = unique(keys);
sd = zeros(1, numel(u));
for k = 1:numel(u)
    sd(k) = std(fixed(strcmp(keys, u{k})), 0);
end
S.baseline_seed_std = mean(sd);

fprintf('\n--- Below the 0.45 cutoff, %d conditions -------------------\n', S.n);
fprintf('MRS   gain over fixed rule : %+.3f dB   better %d/%d, worse %d, max %+.2f\n', ...
        S.mrs_gain, S.mrs_better, S.n, S.mrs_worse, S.mrs_max_gain);
fprintf('Local gain over fixed rule : %+.3f dB   better %d/%d, worse %d\n', ...
        S.local_gain, S.local_better, S.n, S.local_worse);
fprintf('MRS vs local density rule  : %+.3f dB   better %d/%d\n', ...
        S.mrs_vs_local, S.mrs_beats_local, S.n);
fprintf('Oracle gain (upper bound)  : %+.3f dB   MRS recovers %.0f%% of it\n', ...
        S.oracle_gain, 100 * S.mrs_gain / S.oracle_gain);
fprintf('Baseline seed-to-seed std  :  %.3f dB\n', S.baseline_seed_std);
fprintf('Paper reports              : MRS +0.483 (69/72, 1 worse, max +2.47), ');
fprintf('local +0.123 (35, 7 worse), MRS vs local +0.360 (67/72), oracle +1.764, std 0.079\n');
end
