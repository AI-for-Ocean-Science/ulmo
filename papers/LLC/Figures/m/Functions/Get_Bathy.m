function [Bathy, Bathy_Lon, Bathy_Lat, Contours] = Get_Bathy( Fig_No, Lon_Range, Lat_Range, Contour_Ranges)
% Get_Bathy - read the bathymetry for a given range and get contours - PCC
% 
% This function will load the bathymetry in the specified lat, lon ranges
% and, if contour ranges are available, will get all values in each contour
% range.
%
% INPUT
%   Fig_No - figure number into which to plot the resuls. If 0, no plot.
%   Lon_Range - [westernmost_lon, easternmost_lon, stride]
%   Lat_Range - [southernmost_lat, northernmost_lat, stride]
%   Contour_Ranges - array of contour pairs. Will get all GEBCO pixels in 
%    each specified depth range. If absent no contours will be returned.
% OUTPUT
%   Bathy - bathymetric array in the specified range.
%   Bathy_Lon - the longitude of the array.
%   Bathy_Lat - the latitude of the array.
%   Contours - a cell array with the lon,lat pairs of all GEBCO pixels in
%    each of the specified depth ranges.
%
% SAMPLE 
% To get the bathymetric contours betwee [-3 3], [-104 -100], [-1020 -1000] 
%  and [-3020 -3000] for the region 80 to 45W and 20 to 60N:
% [Bathy_Depth, Bathy_Lon, Bathy_Lat, Contours] = Get_Bathy( 1, [-80 -45 1], [20 60 1], {[-3 3] [-104 -100] [-1020 -1000] [-3020 -3000]});
% 

Contours = {};

AxisFontSize = 22;
TitleFontSize = 30;

% Where is the GEBCO bathymetry.

FIBathy = '~/Dropbox/Data/gebco_2020_netcdf/GEBCO_2020.nc';
% Read lat and lon vectors

Lat = ncread(FIBathy,'lat');
Lon = ncread(FIBathy,'lon');

% Constrain the range

mm = find(Lon > Lon_Range(1) & Lon < Lon_Range(2));
if isempty(mm)
    disp(['No longitude values selected. You probably entered the range incorrectly.'])
    keyboard
end
Bathy_Lon = Lon(mm);
disp(['Extracting bathymetry for the longitude between ' num2str(Bathy_Lon(1),4) ' and ' num2str(Bathy_Lon(end),4)])

nn = find(Lat > Lat_Range(1) & Lat < Lat_Range(2));
if isempty(nn)
    disp(['No latitude values selected. You probably entered the range incorrectly.'])
    keyboard
end
Bathy_Lat = Lat(nn);
disp(['and for the latitude between ' num2str(Bathy_Lat(1),4) ' and ' num2str(Bathy_Lat(end),4)])

% Now get the bathymetry for the specified range.

Bathy = ncread(FIBathy, 'elevation', [mm(1) nn(1)], [mm(end)-mm(1)+1 nn(end)-nn(1)+1], [Lon_Range(3) Lat_Range(3)])';

% Extract contours if Contour_Ranges is present

if exist('Contour_Ranges')
    % Loop of the number of contour ranges to extract.
    
    for iContour=1:size(Contour_Ranges,2)
        ii = find( Bathy >= Contour_Ranges{iContour}(1) & Bathy < Contour_Ranges{iContour}(2));
        if isempty(ii)
            disp(['No pixels in GEBCO between ' num2str(Contour_Ranges{iContour}(1)) ' and ' num2str(Contour_Ranges{iContour}(2))])
            Contours{iContour} = nan;
        else
            [I,J] = ind2sub(size(Bathy), ii);
            Contours{iContour} = [Bathy_Lon(J), Bathy_Lat(I)];
        end
    end    
end

% Plot the data if Fig_No is greater than zero.

if Fig_No > 0
    figure(Fig_No)
    clf
    
    imagesc(Bathy_Lon, Bathy_Lat, Bathy)
    set(gca, 'fontsize', AxisFontSize, 'ydir', 'normal')
    
    % Annotate the plot
    
    hold on
%     load coast
%     plot(long,lat,'k','linewidth',2)
    title('GEBCO Bathymetry', 'fontsize', TitleFontSize)
    
    if exist('Contour_Ranges')
        for iContour=1:size(Contour_Ranges,2)
            if rem(iContour,2) == 0
                COLOR = '.c';
            else
                COLOR = '.m';
            end
            plot(Contours{iContour}(:,1), Contours{iContour}(:,2), COLOR, 'markersize', 3) 
            plot(Contours{iContour}(:,1), Contours{iContour}(:,2), '.k', 'markersize', 1)
        end
    end
end

end

