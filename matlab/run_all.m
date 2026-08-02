%RUN_ALL  Regenerate every number of the paper, in one command.
%
%   >> run_all
%
%   Order of business:
%       0  self-checks          hand-computable cases for every primitive
%       1  cross-language check the port against the NumPy reference
%       2  Table II             PSNR on cameraman across the sweep
%       3  Table III/V/VI/VII   the main sweep and its summary
%       4  Table IV             the geometry-vs-image ablation
%       5  Table VIII           residual impulse rate
%       6  Table IX             running time
%       7  demo figures         house at 40%, below the published cutoff
%
%   The full sweep is the expensive part. To reproduce only the headline
%   numbers of Section V-B, which come from the 24 conditions below the
%   0.45 cutoff, run instead:
%
%       rows = run_sweep([0.1 0.2 0.3 0.4], [1 2 3]);
%       summarise_sweep(rows);

setup_paths();
t0 = tic;

fprintf('\n===== 0. self-checks =========================================\n');
selfcheck();

fprintf('\n===== 1. cross-language verification =========================\n');
verify_against_reference();

fprintf('\n===== 2. Table II: restoration quality =======================\n');
run_quality_table();

fprintf('\n===== 3. main sweep (Tables III, V, VI, VII) =================\n');
rows = run_sweep();
summarise_sweep(rows);

fprintf('\n===== 4. Table IV: geometry vs image ablation ================\n');
run_ablation();

fprintf('\n===== 5. Table VIII: residual impulses =======================\n');
run_residual();

fprintf('\n===== 6. Table IX: running time ==============================\n');
run_timing();

fprintf('\n===== 7. demo figures ========================================\n');
mrs_demo('house', 0.4, 1);
mrs_demo('manuscript_beowulf', 0.4, 1);

fprintf('\nAll experiments finished in %.1f minutes.\n', toc(t0) / 60);
