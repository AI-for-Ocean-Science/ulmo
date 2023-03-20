function [cutouts, new_metadata] = get_cutouts_and_metadata(fi, locs)
% get_cutouts_and_metadata - will do just that for the specified file - PCC
%
% INPUT
%   fi - h5 file with the data and metadata.
%   locs - a 5 element vector with the columns for lat, lon, T90, T10 and LL.
%
% OUTPUT
%   cutouts - 3d array with all the cutouts. The first two dimensions are
%    the x and y size of each cutout, the last column if for each cutout.
%   new_metadata - a struction function with the following filds:
%      lat - latitude vector for all cutouts in this dataset.
%      lon - longitude vector for all cutouts in this dataset.
%      dt - T90-T10 for each cutout.
%      LL - loglikelihood for each cutout.
%
cutouts = h5read( fi, '/valid');
cutouts_metadata = h5read( fi, '/valid_metadata');
for i=1:size(cutouts,3)
    new_metadata.lat(i) = str2num(string(cutouts_metadata(locs(1),i)));
    new_metadata.lon(i) = str2num(string(cutouts_metadata(locs(2),i)));
    new_metadata.T90(i) = str2num(string(cutouts_metadata(locs(3),i)));
    new_metadata.T10(i) = str2num(string(cutouts_metadata(locs(4),i)));
    new_metadata.LL(i) = str2num(string(cutouts_metadata(locs(5),i)));
    
    nn = strfind(cutouts_metadata{1,i}, '/');
    j = nn(end) + 1;
    Year = str2num(cutouts_metadata{1,i}(j:j+3));
    Month = str2num(cutouts_metadata{1,i}(j+4:j+5));
    Day = str2num(cutouts_metadata{1,i}(j+6:j+7));
    Hour = str2num(cutouts_metadata{1,i}(j+8:j+9));
    Minute = str2num(cutouts_metadata{1,i}(j+10:j+11));
    new_metadata.Matlab_datetime(i) = datenum(Year, Month, Day, Hour, Minute, 0);
end
new_metadata.dT = new_metadata.T90 - new_metadata.T10;

end