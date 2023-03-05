function [error_return, matchup_index, return_dist_sep, return_time_sep] = find_viirs_llc_matches( viirs_index, viirs_metadata, llc_metadata)
% find_viirs_llc_matches - for gallery plots in Ulmo LLC paper - PCC
%
% A matchup is defined if the closest latitude, longitude and time of an
% LLC cutout is to the same VIIRS cutout. The times also have to be with 12
% hours and the lat and lon to within 10 kilometers.
%
% INPUT
%   viirs_index - the index of the VIIRS cutout to search on.
%   viirs_metadata - metadata values for VIIRS cutouts.
%   llc_metadata - metadata values for LLC cutouts.
%
% OUTPUT
%   error_return - 0 if no error, 1 to 6 if couldn't find a match.
%   matchup_index - the LLC index of the cutout corresponding to the VIIRS
%    cutout.
%   return_dist_sep - the distance between the reference point of the 'pair' of cutouts.
%   return_time_sep - the time between the 'pair' of cutouts.
%

% Intialize thresholds.

dist_threshold = 0.01;
time_threshold = 0.5;

% Initialize return parameters.

error_return = 0;
matchup_index = nan;
return_dist_sep = nan;
return_time_sep = nan;

%

ds_datetime = viirs_metadata.Matlab_datetime(viirs_index);
ds_lon = viirs_metadata.lon(viirs_index);
ds_lat = viirs_metadata.lat(viirs_index);

lat_sep = abs(ds_lat-llc_metadata.lat);
lon_sep = abs(ds_lon-llc_metadata.lon);

dist_sep = sqrt(lat_sep.^2 + (cosd(ds_lat) * lon_sep).^2) * 111;
indices_dist = find(min(dist_sep) == dist_sep);

time_sep = abs(ds_datetime - llc_metadata.Matlab_datetime);
indices_datetime = find(time_sep < 0.5);

if isempty(indices_datetime)
    error_return = 1;
    return
end

% Now loop over the minumum distances in the event that there are more than 1.

num_matches = 0;

for iMatchup=indices_dist
    
    % Now see if there is at least one of the datetime indices that match the
    % minimum distance indices.
    
    dd = find(iMatchup == indices_datetime);
    
    if numel(dd) == 1
        num_matches = num_matches + 1;
        save_match(num_matches) = indices_datetime(dd);
    end
end

switch num_matches
    case 0
        error_return = 2;
        kk = find(min(dist_sep(indices_datetime)) == dist_sep(indices_datetime));
        matchup_index = indices_datetime(kk);
    
    case 1
        matchup_index = save_match(1);
        
    otherwise
        error_return = 3;
        kk = find(min(dist_sep(indices_datetime)) == dist_sep(indices_datetime));
        matchup_index = indices_datetime(kk);
        keyboard
end

% Now check that the separations are within limits.

lat_sep = abs(ds_lat-llc_metadata.lat(matchup_index));
lon_sep = abs(ds_lon-llc_metadata.lon(matchup_index));

return_dist_sep = sqrt(lat_sep.^2 + (cosd(ds_lat) * lon_sep).^2) * 111;
return_time_sep = abs(ds_datetime - llc_metadata.Matlab_datetime(matchup_index));

if return_dist_sep > dist_threshold
    fprintf('Separation (%8.3f) of cutout locations failed for the %i,%i (VIIRS,LLC) pair.\n', return_dist_sep, viirs_index, matchup_index)
    error_return = 4;
end

if return_time_sep > time_threshold
    fprintf('Time separation (%8.3f) failed for the %i,%i (VIIRS,LLC) pair.n', return_time_sep, viirs_index, matchup_index)
    error_return = 5;
end

end

%% Code to plot and print stuff
% 
% for ii=1:7448 % This loop is over all cutouts in gs_viirs_metadata
%     if ii == 8811.00 % This line to look at a specific cutout
%         keyboard
%     end
%     [error_return(ii), ds_indices(ii), xdist_seps(ii), xtime_seps(ii)] = find_viirs_llc_matches( ii, gs_viirs_metadata, gs_llc_metadata); % This line finds matches.
% end
% nn = find(error_return>0); whos nn  % Matches that are not exact
% 
% % Print out the LLC lats and lons on the day of a problem.
% 
% for i=1:29
%     dd = sqrt( (eq_viirs_metadata.lat(3139)-eq_llc_metadata.lat(jj(i))).^2 + (cosd(eq_viirs_metadata.lat(3139)) * (eq_viirs_metadata.lon(3139)-eq_llc_metadata.lon(jj(i)))).^2) * 111;
%     fprintf('%i) %9.2f: %9.2f, %9.2f ::  %9.2f, %9.2f \n', jj(i), dd, eq_viirs_metadata.lon(3139), eq_viirs_metadata.lat(3139), eq_llc_metadata.lon(jj(i)), eq_llc_metadata.lat(jj(i)))
% end
%
% % figure
% plot(eq_llc_metadata.lon, eq_llc_metadata.lat, '.r')
% hold on
% plot(eq_viirs_metadata.lon(nn), eq_viirs_metadata.lat(nn), 'ob', markerfacecolor='b')
% plot(eq_llc_metadata.lon(ds_indices(nn)), eq_llc_metadata.lat(ds_indices(nn)), 'og', markerfacecolor='g')
% set(gca, fontsize=26)
% legend('All LLC Locations', 'VIIRS Locations Not Matching LLC', 'LLC Locations Not Matching VIIRS')
% hh = plot(coastlon,coastlat,'c', linewidth=2);