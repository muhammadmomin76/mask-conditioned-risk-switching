function verify_against_reference(matFile)
%VERIFY_AGAINST_REFERENCE  Cross-language check of the MATLAB port.
%
%   The results in the paper were produced by a NumPy reference
%   implementation. NumPy's PCG64 generator and MATLAB's Mersenne Twister
%   cannot produce identical noise, so this check feeds BOTH implementations
%   the SAME corrupted input and compares every intermediate quantity:
%
%       s1   the Stage-1 repair
%       s2   the Stage-2 candidate
%       V1   the predicted risk of the unrefined repair,  Eq. (7)
%       V2   the predicted risk of the refined repair,    Eq. (6)+(8)
%       the five PSNRs of the five switching configurations
%
%   Agreement to floating-point tolerance means the port is faithful; any
%   remaining difference in the end-to-end sweep is then attributable to the
%   noise realisation alone.
%
%   Run with:  >> setup_paths; verify_against_reference

root = setup_paths();
if nargin < 1
    matFile = fullfile(root, 'verification', 'python_reference.mat');
end
R = load(matFile);

names = fieldnames(R);
tags = {};
for k = 1:numel(names)
    p = strfind(names{k}, '__');
    if ~isempty(p), tags{end+1} = names{k}(1:p(1)-1); end %#ok<AGROW>
end
tags = unique(tags);

rings = distance_rings(5);
lags  = mrs_lags();
worst = 0;
fprintf('\n=== MATLAB port vs the NumPy reference, identical inputs ===\n');

for t = 1:numel(tags)
    tag = tags{t};
    g = @(f) R.([tag '__' f]);

    clean = double(g('clean'));      % stored as uint8 to keep the file small
    noisy = double(g('noisy'));
    mask  = logical(g('mask'));
    surv  = ~mask;

    [s1, sdi, sdj, ringIdx, ringM] = stage1_with_sources(noisy, mask, rings);
    s2 = box3_nonzero_mean(s1);

    curve = isotropic_D(lags, structure_function_global(noisy, surv, lags));
    V1 = risk_stage1(mask, ringIdx, ringM, curve, rings);
    V2 = risk_stage2(mask, sdi, sdj, curve);

    globalD = mean(mask(:));
    localD  = box_sum(double(mask), 2) ./ box_sum(ones(size(mask)), 2);
    q = @(sel) psnr_db(uint8(clean), uint8(min(max(round_half_even(pick(s1, s2, sel)), 0), 255)));

    e1 = (s1 - clean).^2;  e2 = (s2 - clean).^2;
    p = [q(mask & (globalD > 0.45)), ...
         q(mask & ((localD > 0.45) | (globalD > 0.45))), ...
         q(mask & (V2 < V1)), ...
         q(mask & ((V2 < V1) | (globalD > 0.45))), ...
         q(mask & (e2 < e1))];

    % relative differences against the reference
    dS1 = maxrel(s1, g('s1'));
    dS2 = maxrel(s2, g('s2'));
    dV1 = maxrel(V1, g('V1'));
    dV2 = maxrel(V2, g('V2'));
    dDr = maxrel(curve.r(:), g('Dr')(:));
    dDD = maxrel(curve.D(:), g('DD')(:));
    dP  = max(abs(p(:) - g('psnr')(:)));

    % the decision map is what the method IS, so check it exactly
    refRefined = (g('V2') < g('V1')) & mask;
    ourRefined = (V2 < V1) & mask;
    disagree = sum(refRefined(:) ~= ourRefined(:));

    fprintf('\n%-26s  (%d x %d, %d repaired pixels)\n', tag, size(mask,1), size(mask,2), sum(mask(:)));
    fprintf('   D curve   max rel diff : r %.2e   D %.2e\n', dDr, dDD);
    fprintf('   s1 / s2   max rel diff : %.2e / %.2e\n', dS1, dS2);
    fprintf('   V1 / V2   max rel diff : %.2e / %.2e\n', dV1, dV2);
    fprintf('   refine decisions differing on %d of %d repaired pixels\n', disagree, sum(mask(:)));
    fprintf('   PSNR fixed/local/mrs/hybrid/oracle : %.4f %.4f %.4f %.4f %.4f\n', p);
    fprintf('   reference                          : %.4f %.4f %.4f %.4f %.4f  (max diff %.2e dB)\n', ...
            g('psnr'), dP);

    worst = max([worst, dS1, dS2, dV1, dV2, dDr, dDD, dP]);
    assert(disagree == 0, 'decision maps differ on %s', tag);
end

fprintf('\nWorst discrepancy across all quantities and cases: %.3e\n', worst);
if worst < 1e-8
    fprintf('VERDICT: the MATLAB port reproduces the reference implementation exactly.\n');
else
    fprintf('VERDICT: discrepancy above 1e-8 -- investigate before trusting the port.\n');
end
end

% -------------------------------------------------------------------------
function out = pick(s1, s2, sel)
out = s1;  out(sel) = s2(sel);
end

function m = maxrel(a, b)
a = double(a(:)); b = double(b(:));
m = max(abs(a - b) ./ max(abs(b), 1));
end
