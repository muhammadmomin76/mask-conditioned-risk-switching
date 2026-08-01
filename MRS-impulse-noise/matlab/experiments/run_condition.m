function r = run_condition(img, density, seed, useLocalD)
%RUN_CONDITION  One (image, density, realisation) cell of the experiment grid.
%
%   Runs all five switching configurations on the SAME corrupted input, so the
%   only thing that differs between them is the decision rule:
%
%       fixed    the published 45% global constant, Eq. (1)          [1]
%       local    the local-density counterpart of Section IV-A
%       mrs      the proposed rule alone, Eq. (10)
%       hybrid   the proposed rule below the cutoff, published rule
%                above it -- this is the "MRS" of every table
%       oracle   per-pixel best of the two candidates using the clean
%                image; not a method, an upper bound
%
%   Also returns the agreement between the proposed decision and the oracle
%   decision on the repaired pixels (Table VII).
%
%   Reference implementation: mrs_run.run_one.

if nargin < 4, useLocalD = false; end

clean = double(img);
[noisy, mask] = add_spn(img, density, seed);
noisy = double(noisy);
surv  = ~mask;

rings = distance_rings(5);
lags  = mrs_lags();

[s1, sdi, sdj, ringIdx, ringM] = stage1_with_sources(noisy, mask, rings);
s2 = box3_nonzero_mean(s1);

globalD = mean(mask(:));

% ---- reference configurations ------------------------------------------
outFixed = s1;
if globalD > 0.45, outFixed(mask) = s2(mask); end

localD  = box_sum(double(mask), 2) ./ box_sum(ones(size(mask)), 2);
selLas2 = mask & ((localD > 0.45) | (globalD > 0.45));
outLas2 = s1;  outLas2(selLas2) = s2(selLas2);

% ---- oracle: per-pixel best of the two, using ground truth ---------------
e1 = (s1 - clean).^2;
e2 = (s2 - clean).^2;
selOracle = mask & (e2 < e1);
outOracle = s1;  outOracle(selOracle) = s2(selOracle);

% ---- the proposed rule ---------------------------------------------------
Dg = isotropic_D(lags, structure_function_global(noisy, surv, lags));
if useLocalD
    curve = structure_function_local(noisy, surv, lags, 16, Dg);
else
    curve = Dg;
end
V1 = risk_stage1(mask, ringIdx, ringM, curve, rings);
V2 = risk_stage2(mask, sdi, sdj, curve);

selMrs = mask & (V2 < V1);
outMrs = s1;  outMrs(selMrs) = s2(selMrs);

selHyb = mask & ((V2 < V1) | (globalD > 0.45));
outHyb = s1;  outHyb(selHyb) = s2(selHyb);

q = @(a) psnr_db(img, uint8(min(max(round_half_even(a), 0), 255)));

n = max(sum(mask(:)), 1);
orc = selOracle(mask);
agree = mean(selMrs(mask) == orc);

r = struct('image', '', 'd', density, 'seed', seed, ...
           'psnr_fixed',  q(outFixed), ...
           'psnr_las2',   q(outLas2), ...
           'psnr_mrs',    q(outMrs), ...
           'psnr_hyb',    q(outHyb), ...
           'psnr_oracle', q(outOracle), ...
           'psnr_s1',     q(s1), ...
           'psnr_s2',     q(s2), ...
           'frac_mrs',    sum(selMrs(:)) / n, ...
           'frac_oracle', mean(orc), ...
           'agree',       agree);
end
