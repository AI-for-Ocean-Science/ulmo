function HEALPix_plot_Mollweid( FigNo, subplot_details, Lat, Lon, data_to_plot, dot_size, colormap_to_use, color_range, TITLE, FigDir, PrintName, ...
    ll_threshold, color_label, color_label_size, lat_range, lon_range, xxtra, yxtra, cxtra)
% HEALPix_plot_Mollweid - scatter plots HEALPix values on Mollweid projection - PCC
%
% INPUT
%   FigNo - figure number.
%   subplot_details - position vector for subplot. 
%   Lat - latitude vector. 
%   Lon - longitude vector.
%   data_to_plot - vector of data to plot.
%   dot_size - size of filled dots to scatter plot.
%   colormap_to_use - palette to use for the plot.
%   color_range - range of palette to display.
%   TITLE - for the plot.
%   FigDir - directory for the output figure.
%   PrintName - name of file to be printed. If empty will not print out.
%   ll_threshold - if present will gray out palette from 128-ll_threshold(1) 
%       to 128+ll_threshold(2) with the gray value of ll_threshold(3), the
%       latter ranging from 0 to 1. If present and ll_threshold(1)=0 and
%       ll_threshold(2)=0 will not threshold.
%   color_label - Label to use on palette if zonal mean plots.
%   lat_range - if less than -90 to 90 otherwise []
%   lon_range - if less than -180 to 180 otherwise []
%   xxtra - longitude of extra line to plot,
%   yxtra - latitude of extra line to plot.
%   cxtra - color of extra line to plot.
%
% SAMPLE CALL
%   plot_HEALPix_Mollweid( FigNo, Lat, Lon, head_tail_diff, dot_size, color_range, ...
%    '(VIIRS 2012-2015) - (VIIRS 2018-2020) LL', ['head_tail_LL_' num2str(nsigma)])
%

% if exist('lat_range') == 0
if isempty(lat_range) == 1
    lat_range = [-90 90];
    lat_lower = -75;
    lat_upper = 75;
else
    lat_lower = lat_range(1);
    lat_upper = lat_range(2);
end

% if exist('lon_range') == 0
if isempty(lon_range) == 1
    lon_range = [-180 180];
end

% lg = 0.7;
% colormap_to_use = redblue(256);
% colormap_to_use(1,:) = [lg lg lg];
% colormap_to_use = coolwarm(256);
% colormap_to_use = colormap(jet);

% % % FigDir = '~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Figures/For_Kat/';

lg = 0.7;

[ coastlon, coastlat ] = Shift_Coast_To_Start_At_0;

if isempty(subplot_details)
    figure(FigNo)
    clf
    colormap(colormap_to_use)
else
    figure(FigNo)
    ax_sp = subplot( 'Position', subplot_details(3:end));
end

if isempty(ll_threshold) == 0

    if ll_threshold(1) ~= 0 & ll_threshold(2) ~= 0
        
        masked_cc = ll_threshold(3);

        % Set values less than the lower limit to display to that limit so
        % that they won't be grayed out. 
        
        nn = find(data_to_plot<color_range(1));
        data_to_plot(nn) = color_range(1) + (color_range(2) - color_range(1)) / 255;
        
        % Do the same for the values greater than the upper limit to display.
        
        nn = find(data_to_plot>color_range(2));
        data_to_plot(nn) = color_range(2) - (color_range(2) - color_range(1)) / 255;
        
        % Now gray out values in the middle.
        
        nn = find(data_to_plot>=ll_threshold(1) & data_to_plot<=ll_threshold(2));
        
        data_to_plot(nn) = color_range(1) - 1;
        
        pcc_pal = colormap(colormap_to_use);
        num_pal_elements = size(pcc_pal,1);
        % % %         frac = floor(num_pal_elements * ll_threshold(1) / (color_range(2) - color_range(1)));
        
        frac = (ll_threshold - color_range(1)) / (color_range(2) - color_range(1)) * num_pal_elements;

        pcc_pal(1,:) = [masked_cc, masked_cc, masked_cc];
        
        for i=floor(frac(1)):ceil(frac(2))
            pcc_pal(i,:) = [masked_cc, masked_cc, masked_cc];
        end
        
        colormap(ax_sp,pcc_pal)
    else
    	colormap(ax_sp,colormap_to_use)
    end
else
    colormap(ax_sp,colormap_to_use)
end

plinesep = 15;
if abs(diff(lat_range)) < 60
    if abs(diff(lat_range)) <= 25
        plinesep = 5;        
    else
        plinesep = 10;
    end
end

mlinesep = 30;
if abs(diff(lon_range)) <= 50
    if abs(diff(lon_range)) <= 25
        mlinesep = 5;
    else
        mlinesep = 10;
    end
end
    
if isempty(lat_range) == 1
    lon_1 = -180;
    lon_2 = 180;
else
    lon_1 = lon_range(1);
    lon_2 = lon_range(2);
end

landareas = shaperead('landareas.shp','UseGeoCoords',true);
% ax = axesm ('mollweid', 'Frame', 'on', 'Grid', 'on', maplatlimit=lat_range, maplonlimit=lon_range, plinelocation=[lat_range(1):plinesep:lat_range(2)], mlinelocation=[lon_1:mlinesep:lon_2]);
ax = axesm ('mollweid', 'Frame', 'on', 'Grid', 'on', maplatlimit=lat_range, maplonlimit=lon_range, plinelocation=[-75:plinesep:75], mlinelocation=[-150:mlinesep:150]);
% ax = axesm ('mollweid', 'Grid', 'on', maplatlimit=lat_range, maplonlimit=lon_range);

% Add missing HEALPix cells to the array to be plotted with very negative values.

% data_to_plot(data_to_plot < color_range(1)) = color_range(1) + 0.001 * (color_range(2) - color_range(1));
% 
% Latp = [Lat Lat_o(nnBad)];
% Lonp = [Lon Lon_o(nnBad)];
% data_to_plotp = [data_to_plot' Lat_o(nnBad)*0-100000];
    
scatterm( Lat, Lon, dot_size, data_to_plot, 'filled')
% scatterm( Latp, Lonp, dot_size, data_to_plotp, 'filled')
hold on

geoshow(landareas,'FaceColor',[lg, lg, lg],'EdgeColor',[.6 .6 .6]);

plotm( coastlat, coastlon, 'k', linewidth=2)

setm(ax, 'meridianlabel', 'on', 'parallellabel', 'on')


mlabel(-17)
if lat_range(1)~=-90 & lat_range(2)~=90
%     if lat_range(1) < 0 & lat_range(1) > -15
%         mlabel(lat_range(1)+abs(diff(lat_range)/20))
%     else
%         mlabel(lat_range(2)-abs(diff(lat_range)/20))
%     end
    MapLonLimit = getm(gca,'maplonlimit');
    if sum(MapLonLimit) < 0
        setm(gca,mlabellocation=[lon_range(1):mlinesep:lon_range(2)]-360, mlabelparallel=floor(lat_range(1)+abs(diff(lat_range)/20)) )
    else
        if lat_range(2) == 60
            setm(gca,mlabellocation=[lon_range(1):mlinesep:lon_range(2)], mlabelparallel=floor(lat_range(2)))
        else
            setm(gca,mlabellocation=[lon_range(1):mlinesep:lon_range(2)], mlabelparallel=floor(lat_range(1)+abs(diff(lat_range)/20)) )
        end
    end
end

setm(ax, fontsize=14)

gg = gridm('on');
set( gg, linewidth=2, color=[lg, lg, lg])

set(gca,'fontsize',24)

% % % if findstr(color_label, '$')
% % %     c.Label.Interpreter='latex';
% % %     color_label_m = ['\boldmath' color_label];
% % % else
% % %     color_label_m = ['\bf{' color_label  '}'];
% % % end

% don't display the colorbar if color_label is empty.

c = colorbar('southoutside');
if isempty(color_label) == 0
    
    if (strcmp(color_label(1), '$')) & (strcmp(color_label(end), '$'))
        c.Label.Interpreter='latex';
        color_label_m = ['\boldmath' color_label];
    else
        color_label_m = color_label;
    end
    c.Label.FontSize = color_label_size;
    c.Label.String = color_label_m;
end

caxis(color_range)

title( ['\bf{\boldmath{' TITLE '}}'], fontsize=30, interpreter='latex', fontweight='bold')

COLOR = get(gcf,'Color');
set(gca,'XColor',COLOR,'YColor',COLOR,'TickDir','out')

if isempty(lat_range)==1
    ylatt = [-89:0.1:89];
    plotm(ylatt, -ones(1,length(ylatt))*180, 'k', linewidth=2)
    plotm(ylatt, ones(1,length(ylatt))*180, 'k', linewidth=2)
else
    ylatt = [-89:0.1:89];
    plotm(ylatt, ones(1,length(ylatt))*lon_range(1), 'k', linewidth=2)
    plotm(ylatt, ones(1,length(ylatt))*lon_range(2), 'k', linewidth=2)
end

if exist('xxtra')
    for iel=1:length(xxtra)
        plotm( yxtra{iel}, xxtra{iel}, cxtra{iel}, linewidth=2)
    end
end

if isempty(PrintName) == 0
    temp1 = strrep( PrintName, '.', '_');
    temp2 = strrep( temp1, '-', 'm_');
% % %     if isempty(subplot_details)
% % %         filename_out = [FigDir 'pcc_1_' temp2];
% % %     else
% % %         filename_out = [FigDir 'pcc_2_' temp2];
% % %     end
    filename_out = [FigDir temp2];
    print( filename_out, '-dpng')
    
    fprintf('Printing figure: %s\n', filename_out)
end

end
% 