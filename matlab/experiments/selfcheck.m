function selfcheck()
%SELFCHECK  Runnable proof that the primitives behave as the derivation says.
%
%   Run with:  >> setup_paths; selfcheck
%
%   Every assertion below is a hand-computable case, so a failure points at a
%   specific function rather than at "the numbers moved".

setup_paths();
tol = 1e-9;

% ---- metrics -------------------------------------------------------------
img = uint8(mod((1:64)' * (1:64), 251) + 2);
assert(mse_metric(img, img) == 0, 'mse of an image against itself');
assert(abs(ssim_global(img, img) - 1) < tol, 'ssim of an image against itself');
a = uint8([0 0; 0 0]); b = uint8([2 0; 0 0]);
assert(mse_metric(a, b) == 1, 'hand-computed MSE: 2^2 / 4 = 1');
assert(abs(psnr_db(a, b) - 10*log10(255^2)) < tol, 'hand-computed PSNR');

% ---- noise model, Eq. (11) ----------------------------------------------
flat = uint8(100 * ones(200));
[n1, m1] = add_spn(flat, 0.30, 1);
assert(abs(mean(m1(:)) - 0.30) < 0.01, 'achieved density is binomial around d');
assert(all(n1(m1) == 0 | n1(m1) == 255), 'corrupted pixels take extreme values');
assert(any(n1(:) == 0) && any(n1(:) == 255), 'both salt and pepper appear');
assert(isequal(n1, add_spn(flat, 0.30, 1)), 'same seed gives the same noise');
assert(~isequal(n1, add_spn(flat, 0.30, 2)), 'a different seed differs');

% ---- shift primitive -----------------------------------------------------
A = [1 2 3; 4 5 6; 7 8 9];
assert(isequal(shift_image(A, 1, 0), [4 5 6; 7 8 9; 0 0 0]), 'shift down-source');
assert(isequal(shift_image(A, 0, 1), [2 3 0; 5 6 0; 8 9 0]), 'shift right-source');

% ---- rings ---------------------------------------------------------------
rings = distance_rings(5);
assert(rings(1).d2 == 1 && size(rings(1).offs, 1) == 4, 'first ring is the 4-neighbourhood');
assert(rings(2).d2 == 2 && size(rings(2).offs, 1) == 4, 'second ring is the diagonals');
assert(sum(arrayfun(@(r) size(r.offs,1), rings)) == 11*11 - 1, 'rings tile the 11x11 window');

% ---- Stage 1, hand-computed ---------------------------------------------
small = [10 20 30; 40 0 60; 70 80 90];
mask = false(3); mask(2,2) = true;
s1 = stage1_with_sources(small, mask);
assert(abs(s1(2,2) - (20+40+60+80)/4) < tol, 'Stage 1 averages the tied nearest donors');
assert(isequal(s1([1 3], :), small([1 3], :)), 'survivors are left untouched');

% ---- Stage 2, hand-computed ---------------------------------------------
s2 = box3_nonzero_mean([10 20 30; 40 99 60; 70 80 90]);
assert(abs(s2(2,2) - 499/9) < tol, '3x3 mean when nothing is zero');
s2b = box3_nonzero_mean([0 0 0; 0 100 200; 0 0 0]);
assert(abs(s2b(2,2) - 150) < tol, 'zeros are excluded from the 3x3 mean');

% ---- structure function on a known field --------------------------------
% A horizontal ramp of slope g has D(h) = (g*h_x)^2 exactly.
g = 3;
[~, J] = ndgrid(1:60, 1:60);
ramp = g * J;
surv = true(60);
lags = [0 1; 0 2; 1 0];
D = structure_function_global(ramp, surv, lags);
assert(abs(D(1) - g^2)   < 1e-6, 'D at lag (0,1) of a ramp');
assert(abs(D(2) - 4*g^2) < 1e-6, 'D at lag (0,2) of a ramp');
assert(abs(D(3))         < 1e-6, 'D along the flat direction is zero');

% ---- extension variance with exactly one donor --------------------------
% Only (3,2) survives, so the pixel at (3,3) has a single tied donor at
% distance 1. Eq. (7) then has one s = t term, D(0), which the curve clamps
% to D(1) = 10, giving V1 = D(1) - D(0)/(2*1^2) = 10 - 5 = 5.
curve = struct('r', [1; 2; 3], 'D', [10; 40; 90], 'local', false);
mask1 = true(5);  mask1(3,2) = false;
ringIdx = -ones(5); ringIdx(3,3) = 1;      % ring 1 = distance 1
ringM = zeros(5);   ringM(3,3) = 1;        % exactly one tied donor
V1 = risk_stage1(mask1, ringIdx, ringM, curve, distance_rings(5));
assert(abs(V1(3,3) - 5) < tol, 'single-donor extension variance');

% ---- eval_D clamps outside the estimated range --------------------------
assert(eval_D(curve, 0.5) == 10, 'clamped below the first lag');
assert(eval_D(curve, 99)  == 90, 'clamped above the last lag');
assert(abs(eval_D(curve, 1.5) - 25) < tol, 'linear interpolation between lags');

% ---- baselines -----------------------------------------------------------
flat7 = uint8(100 * ones(7)); spike = flat7; spike(4,4) = 0;
m3 = median_filter_k(spike, 3);
amf = adaptive_median_filter(spike, 7);
assert(m3(4,4) == 100, '3x3 median removes an isolated spike');
assert(amf(4,4) == 100, 'AMF removes an isolated spike');
assert(isequal(median_filter_k(flat7, 3), flat7), 'edge replication leaves a flat image alone');
patch = uint8([10 20 30; 40 50 60; 70 80 90]);
mp = median_filter_k(patch, 3);
assert(mp(2,2) == 50, 'hand-computed 3x3 median');

% ---- the switch itself ---------------------------------------------------
base = uint8(mod((1:120)' * (1:120), 253) + 1);      % no 0/255 -> density is exact
[lo, mlo] = add_spn(base, 0.30, 7);
[hi, mhi] = add_spn(base, 0.60, 7);
assert(isequal(nvbmf(lo, mlo), uint8(min(max(round_half_even(stage1_with_sources(double(lo), mlo)),0),255))), ...
       'below the cutoff the published filter is Stage 1 only');
assert(~isequal(nvbmf(hi, mhi), uint8(min(max(round_half_even(stage1_with_sources(double(hi), mhi)),0),255))), ...
       'above the cutoff Stage 2 runs');
[~, info] = mrs_filter(lo, mlo);
assert(any(info.refined(:)), 'MRS refines some pixels below the cutoff, where the published rule refines none');
assert(all(info.V1(:) > 0) && all(info.V2(:) > 0), 'both predicted risks are positive');

% ---- MRS improves on the published rule on a smooth image ---------------
root = setup_paths();
house = load_gray(fullfile(root, 'images', 'benchmark', 'house.png'));
[hn, hm] = add_spn(house, 0.4, 1);
pFixed = psnr_db(house, nvbmf(hn, hm));
pMrs   = psnr_db(house, mrs_filter(hn, hm));
assert(pMrs > pFixed + 1.0, sprintf('house at 40%%: MRS %.2f should beat NVBMF %.2f by >1 dB', pMrs, pFixed));

fprintf('OK: all self-checks passed.\n');
fprintf('    house @ 40%%: NVBMF %.2f dB -> MRS %.2f dB (%+.2f dB)\n', pFixed, pMrs, pMrs - pFixed);
end
