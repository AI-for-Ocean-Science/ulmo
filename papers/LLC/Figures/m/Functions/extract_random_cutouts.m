function [cutouts_out, metadata_out, rand_indices, means] = extract_random_cutouts(lat_range, lon_range, percent_from_median, num_to_select, cutouts_in, metadata_in, rand_indices)
% extract_random_cutouts - will select num_to_select cutouts from the provided list - PCC
%
% INPUT
%   lat_range - the latitude range of rectangle from which to select the cutouts.
%   lon_range - the longitude range of rectangle from which to select the cutouts.
%   percent_from_median - if not empty will select only from +/- of this
%    value from the median LL value.
%   num_to_select - number of cutouts to randomly select from those available. 
%   cutouts_in - 3d array of cutouts to choose from.
%   metadata_in a structure with the following fields:
%      lat - latitude of each cutout in the list
%      lon - longitude of each cutout in the list
%      dt - T90-T0 of each cutout in the list
%      LL - LL of each cutout in the list
%   rand_indices - if empty will select cutouts at random otherwise 
%    will use the ones passed in.
%
% OUTPUT
%   cutouts_out - num_to_select cutouts in 3d array.
%   metadata_out - a structure with the following fields:
%      lat - latitude of selected cutouts
%      lon - longitude of selected cutouts
%      dt - T90-T10 of selected cutouts
%      LL - LL of selected cutouts
%   rand_indices - the indices of selected cutouts.
%

nn = find( (metadata_in.lat >= lat_range(1)) & (metadata_in.lat < lat_range(2)) & (metadata_in.lon >= lon_range(1)) & (metadata_in.lon < lon_range(2)) );

% Get stats for all cutouts in the lat,lon range.

means.LL_All = mean(metadata_in.LL(nn));
means.dT_All = mean(metadata_in.dT(nn));
sigma_LL_All = std(metadata_in.LL(nn));

if (isempty(percent_from_median) == 0) & (isempty(rand_indices) == 1)
    
    % If sampling from a percentage from the median, extract the metadata 
    % and cutouts in the lat,lon range from the dataset.
    
    j = 0;
    for i=1:numel(nn)
        j = j + 1;
        metadata_in_p.lat(j) = metadata_in.lat(nn(i));
        metadata_in_p.lon(j) = metadata_in.lon(nn(i));
        metadata_in_p.LL(j) = metadata_in.LL(nn(i));
        metadata_in_p.Matlab_datetime(j) = metadata_in.Matlab_datetime(nn(i));
        metadata_in_p.dT(j) = metadata_in.dT(nn(i));
        
%         cutouts_in_p(:,:,j) = cutouts_in(:,:,nn(i));
    end
    
%     clear metadata_in cutouts_in
%     
%     metadata_in = metadata_in_p;
%     cutouts_in = cutouts_in_p;
    
    % Now find the cutouts with percent_from_median.
    
    median_LL = median(metadata_in_p.LL);
    sorted_LL = sort(metadata_in_p.LL);
    
    lower_LL = sorted_LL(floor(numel(metadata_in_p.LL) * (0.5 - percent_from_median / 100)));
    upper_LL = sorted_LL(ceil(numel(metadata_in_p.LL) * (0.5 + percent_from_median / 100)));
    
    nn = find(metadata_in.LL>=lower_LL & metadata_in.LL<upper_LL);
end

% Select cutouts to use if not specified.

if isempty(rand_indices) == 1
    rand_indices = datasample( nn, num_to_select, replace=false);
end

for i=1:num_to_select
    j = rand_indices(i);
    cutouts_out(:,:,i) = cutouts_in(:,:,j);
    metadata_out.lat(i) = metadata_in.lat(j);
    metadata_out.lon(i) = metadata_in.lon(j);
    metadata_out.dT(i) = metadata_in.dT(j);
    metadata_out.LL(i) = metadata_in.LL(j);
    metadata_out.Matlab_datetime(i) = metadata_in.Matlab_datetime(j);
end

means.LL_Selected = mean(metadata_out.LL);
means.dT_Selected = mean(metadata_out.dT);

end
