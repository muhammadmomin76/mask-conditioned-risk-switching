function root = setup_paths()
%SETUP_PATHS  Put core/, filters/, metrics/ and experiments/ on the path.
%
%   Call this once before using anything else:
%       >> setup_paths
%   Returns the project root so scripts can find images/ and results/.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);

addpath(fullfile(root, 'core'));
addpath(fullfile(root, 'filters'));
addpath(fullfile(root, 'metrics'));
addpath(fullfile(root, 'experiments'));
end
