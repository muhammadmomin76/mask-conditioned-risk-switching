function lags = mrs_lags()
%MRS_LAGS  The lag set at which the structure function is estimated.
%
%   Short lags dominate the risk (Stage-1 donors are almost always adjacent
%   below the cutoff), so the set is dense near the origin and thins out with
%   distance. Both axis directions and both diagonals are included so that
%   the isotropic average of ISOTROPIC_D is not biased towards one direction.
%
%   Reference implementation: mrs_core.LAGS.

lags = [ 0  1;  1  0;  1  1;  1 -1;
         0  2;  2  0;  2  2;  2 -2;
         0  3;  3  0;  1  2;  2  1;
         0  4;  4  0;  3  3;  3 -3;
         0  5;  5  0;  0  6;  6  0;
         0  8;  8  0];
end
