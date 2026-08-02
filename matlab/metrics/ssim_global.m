function s = ssim_global(X, Y)
%SSIM_GLOBAL  Structural similarity, global (whole-image) form.
%
%       SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
%              ------------------------------------------
%              (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)
%
%   with C1 = (0.01*255)^2 and C2 = (0.03*255)^2 (Wang et al. 2004).
%   Assembled by hand from array arithmetic, not a toolbox SSIM call, so the
%   comparison does not depend on a toolbox's windowing choices.

X = double(X(:)); Y = double(Y(:));
C1 = (0.01 * 255)^2;
C2 = (0.03 * 255)^2;

muX = mean(X);       muY = mean(Y);
varX = mean((X - muX).^2);            % population variance
varY = mean((Y - muY).^2);
covXY = mean((X - muX) .* (Y - muY));

num = (2*muX*muY + C1) * (2*covXY + C2);
den = (muX^2 + muY^2 + C1) * (varX + varY + C2);
s = num / den;
end
