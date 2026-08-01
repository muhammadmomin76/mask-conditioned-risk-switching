function V2 = risk_stage2(mask, sdi, sdj, curve)
%RISK_STAGE2  Predicted squared error of the refined (3x3 mean) repair.
%
%   Stage 2 averages the 3x3 window of the Stage-1 OUTPUT, so the window
%   contains repaired pixels as well as surviving ones. Each window position
%   is therefore traced back to the pixel that actually supplied its value,
%   Eq. (8):
%
%       S2(u) = u + d(p + u),    u in {-1,0,1}^2
%
%   where d(x) is the donor offset of pixel x, and zero for a survivor.
%   Substituting the resulting effective source multiset into the extension
%   variance of Eq. (6) gives V2. Like V1 it is decided by the mask alone.
%
%   Reference implementation: mrs_core.risk_stage2.

[H, W] = size(mask);

win = [-1 -1; -1 0; -1 1; 0 -1; 0 0; 0 1; 1 -1; 1 0; 1 1];
n = size(win, 1);

Si = cell(1, n);  Sj = cell(1, n);
for t = 1:n
    ui = win(t,1); uj = win(t,2);
    Si{t} = ui + shift_image(sdi, ui, uj);
    Sj{t} = uj + shift_image(sdj, ui, uj);
end

m = double(n);

% first term: mean error of the individual effective sources
t1 = zeros(H, W);
for a = 1:n
    t1 = t1 + eval_D(curve, hypot(Si{a}, Sj{a}));
end
t1 = t1 / m;

% second term: the reduction earned by averaging them
t2 = zeros(H, W);
for a = 1:n
    for b = 1:n
        t2 = t2 + eval_D(curve, hypot(Si{a} - Si{b}, Sj{a} - Sj{b}));
    end
end
t2 = t2 / (2 * m * m);

V2 = max(t1 - t2, 1e-9);
end
