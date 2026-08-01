function set_seed(seed)
%SET_SEED  Seed the Mersenne Twister, portably across MATLAB and Octave.
%
%   Every experiment in this project is seeded, so that all filters receive an
%   identical corrupted input and every reported number can be regenerated.

try
    rng(seed, 'twister');          % MATLAB (and recent Octave)
catch
    rand('twister', seed);         %#ok<RAND> % older Octave
end
end
