function [viirs_cutouts_out, viirs_metadata_out, llc_cutouts_out, llc_metadata_out, rand_indices, viirs_means, llc_means] = ...
    extract_random_cutout_pairs(lat_range, lon_range, percent_from_median, num_to_select, viirs_cutouts_in, viirs_metadata_in, llc_cutouts_in, llc_metadata_in, rand_indices)
% extract_random_cutout - will select num_to_select cutouts from the provided list - PCC
%
% This script expects that the VIIRS and LLC cutouts are aligned; i.e.,
% that they are on the same day (within 12 hours) and at the same latitude
% and longitude. The script will find, hopefully, a set of indices for this
% the VIIRS LLs are within percent_from_median of the median VIIRS LL for 
% the region AND the LLC LLs are within percent_from_median of the median 
% VIIRS LL for the region.
%
% INPUT
%   lat_range - the latitude range of rectangle from which to select the cutouts.
%   lon_range - the longitude range of rectangle from which to select the cutouts.
%   percent_from_median - if not empty will select only from +/- of this
%    value from the median LL value.
%   num_to_select - number of cutouts to randomly select from those available. 
%   viirs_cutouts_in - 3d array of cutouts to choose from.
%   viirs_metadata_in a structure with the following fields:
%      lat - latitude of each cutout in the list
%      lon - longitude of each cutout in the list
%      dt - T90-T0 of each cutout in the list
%      LL - LL of each cutout in the list
%   llc_cutouts_in - 3d array of cutouts to choose from.
%   llc_metadata_in a structure with the following fields:
%      lat - latitude of each cutout in the list
%      lon - longitude of each cutout in the list
%      dt - T90-T0 of each cutout in the list
%      LL - LL of each cutout in the list
%   rand_indices - if empty will select cutouts at random otherwise 
%    will use the ones passed in.
%
% OUTPUT
%   viirs_cutouts_out - num_to_select cutouts in 3d array.
%   viirs_metadata_out - a structure with the following fields:
%      lat - latitude of selected cutouts
%      lon - longitude of selected cutouts
%      dt - T90-T10 of selected cutouts
%      LL - LL of selected cutouts
%   llc_cutouts_out - num_to_select cutouts in 3d array.
%   llc_metadata_out - a structure with the following fields:
%      lat - latitude of selected cutouts
%      lon - longitude of selected cutouts
%      dt - T90-T10 of selected cutouts
%      LL - LL of selected cutouts
%   rand_indices - the indices of selected cutouts.
%

% First fine the cutouts that fall in the specified lat,lon range.

nn = find( (viirs_metadata_in.lat >= lat_range(1)) & (viirs_metadata_in.lat < lat_range(2)) & (viirs_metadata_in.lon >= lon_range(1)) & (viirs_metadata_in.lon < lon_range(2)) );

% Get stats for all cutouts in the lat,lon range.

viirs_means.LL_All = mean(viirs_metadata_in.LL(nn));
viirs_means.dT_All = mean(viirs_metadata_in.dT(nn));
viirs_sigma_LL_All = std(viirs_metadata_in.LL(nn));

llc_means.LL_All = mean(llc_metadata_in.LL(nn));
llc_means.dT_All = mean(llc_metadata_in.dT(nn));
llc_sigma_LL_All = std(llc_metadata_in.LL(nn));

if (isempty(percent_from_median) == 0) & (isempty(rand_indices) == 1)
    
    % If sampling from a percentage from the median, extract the metadata 
    % and cutouts in the lat,lon range from the dataset.
    
    j = 0;
    for i=1:numel(nn)
        j = j + 1;
        
        
        field_names = fieldnames(viirs_metadata_in);
        for iFieldname=1:numel(field_names)
            eval(['viirs_metadata_in_p.' field_names{iFieldname} '(j) = viirs_metadata_in.' field_names{iFieldname} '(nn(i));'])
            eval(['llc_metadata_in_p.' field_names{iFieldname} '(j) = llc_metadata_in.' field_names{iFieldname} '(nn(i));'])
        end        
    end
    
    % Now find the cutouts with percent_from_median.
    
%     median_LL = median(viirs_metadata_in_p.LL);
    viirs_sorted_LL = sort(viirs_metadata_in_p.LL);
    viirs_lower_LL = viirs_sorted_LL(floor(numel(viirs_metadata_in_p.LL) * (0.5 - percent_from_median / 100)));
    viirs_upper_LL = viirs_sorted_LL(ceil(numel(viirs_metadata_in_p.LL) * (0.5 + percent_from_median / 100)));
    
    llc_sorted_LL = sort(llc_metadata_in_p.LL);
    llc_lower_LL = llc_sorted_LL(floor(numel(llc_metadata_in_p.LL) * (0.5 - percent_from_median / 100)));
    llc_upper_LL = llc_sorted_LL(ceil(numel(llc_metadata_in_p.LL) * (0.5 + percent_from_median / 100)));
    
    nn = find(...
        viirs_metadata_in.LL>=viirs_lower_LL & viirs_metadata_in.LL<viirs_upper_LL & ...
        llc_metadata_in.LL>=llc_lower_LL     & llc_metadata_in.LL<llc_upper_LL );
end

if numel(nn) < num_to_select
    fprintf('Bummer, need at least %i pairs but only found %i\n', num_to_sample, numel(nn))
else
    fprintf('Great, found %i matches in %5.2f-%5.2f lat, %5.2f-%5.2f lon.\n', numel(nn), lat_range, lon_range)
end

% Select cutouts to use if not specified.

if isempty(rand_indices) == 1
    rand_indices = datasample( nn, num_to_select, replace=false);
end

for i=1:num_to_select
    viirs_cutouts_out(:,:,i) = viirs_cutouts_in(:,:,rand_indices(i));
    llc_cutouts_out(:,:,i) = llc_cutouts_in(:,:,rand_indices(i));
    
    field_names = fieldnames(viirs_metadata_in);
    for iFieldname=1:numel(field_names)
        eval(['viirs_metadata_out.' field_names{iFieldname} '(i) = viirs_metadata_in.' field_names{iFieldname} '(rand_indices(i));'])
        eval(['llc_metadata_out.' field_names{iFieldname} '(i) = llc_metadata_in.' field_names{iFieldname} '(rand_indices(i));'])
    end
end

viirs_means.LL_Selected = mean(viirs_metadata_out.LL);
viirs_means.dT_Selected = mean(viirs_metadata_out.dT);

llc_means.LL_Selected = mean(llc_metadata_out.LL);
llc_means.dT_Selected = mean(llc_metadata_out.dT);

end
