function imgs = test_images(root)
%TEST_IMAGES  The six 8-bit grayscale test images of Table I.
%
%   Four standard benchmark images plus two added for this study, a painting
%   and a scan of a manuscript page, to cover content the standard set does
%   not contain. IMGS is a struct array with fields .name and .path.

if nargin < 1, root = fileparts(fileparts(mfilename('fullpath'))); end
b = fullfile(root, 'images', 'benchmark');
if ~isfolder(b)
    % the released layout keeps one copy of the images at the repository root,
    % one level above matlab/
    root = fileparts(root);
    b = fullfile(root, 'images', 'benchmark');
end
c = fullfile(root, 'images', 'custom');

imgs = struct( ...
    'name', {'cameraman', 'house', 'mandrill', 'peppers', ...
             'field_vangogh', 'manuscript_beowulf'}, ...
    'path', {fullfile(b, 'cameraman.png'), fullfile(b, 'house.png'), ...
             fullfile(b, 'mandrill.png'),  fullfile(b, 'peppers.png'), ...
             fullfile(c, 'field_vangogh.png'), ...
             fullfile(c, 'manuscript_beowulf.png')});
end
