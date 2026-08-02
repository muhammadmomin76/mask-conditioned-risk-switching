function write_rows_csv(rows, path)
%WRITE_ROWS_CSV  Write a struct array of result rows to CSV.
%
%   Plain FPRINTF, so the file is identical under MATLAB and Octave and needs
%   no toolbox.

f = fopen(path, 'w');
if f < 0, error('write_rows_csv:open', 'cannot write %s', path); end

fprintf(f, ['image,density,seed,psnr_fixed,psnr_local,psnr_mrs,psnr_hybrid,' ...
            'psnr_oracle,psnr_s1,psnr_s2,frac_mrs,frac_oracle,agree\n']);
for k = 1:numel(rows)
    r = rows(k);
    fprintf(f, '%s,%.1f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
        r.image, r.d, r.seed, r.psnr_fixed, r.psnr_las2, r.psnr_mrs, ...
        r.psnr_hyb, r.psnr_oracle, r.psnr_s1, r.psnr_s2, ...
        r.frac_mrs, r.frac_oracle, r.agree);
end
fclose(f);
end
