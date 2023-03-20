function subplot_details = make_quad_plot( FigNo, subplot_details, lat, lon, var_1, var_2, ll_thresholds, lat_range, lon_range, ...
     dot_size, colormap_to_use, caxis_mollweide, color_label_mollweide, xlim_zonal_mean, ylim_zonal_mean, Title_1, Title_2, FigDir, PrintName)
% make_quad_plot - make a figure with one or two Mollweide plots + zonal averages if requested - PCC
%
% INPUT
%   FigNo - the figure number to use for plotting.
%   subplot_details - if zonal means to be plotted, this is a 2-d array 
%    with the numbers needed for the location of the subplots. If all zeros 
%    and zonal means are to be plotted, this function will request information
%    to determine the position of subplots. This information will be
%    returned and can be used in subsequent subplots in this run. If not
%    empty, the first row is for the 1,2 plot and the second row for the 2,2
%    plot. If the variable is empty, will not use subplot at all.
%   lat - latitude of Mollweide cells,
%   lon - longitude of Mollweide cells,
%   var_1 - median LL values for each cell of dataset #1 to plot,
%   var_2 - median LL values for each cell of dataset #1 to plot if there
%   ll_thresholds - If there are two datasets to plot, this is a 3 element 
%    vector with the lower and upper thresholds to use when masking a range
%    of values in Mollweide plots and the value to to use for the mask. The 
%    latter assumes that a gray scale of this value. If only one, enter an 
%    empty variable here. 
%   lat_range - two element vector with the lower and upper latitudes to plot; if
%     plot; if [0 0], -90 to 90 plotted,
%   lon_range - two element vector with the left and right longitudes to
%     plot; if [0 0], -180 to 180 plotted
%   dot_size - size of dots in Mollweide plots.
%   colormap_to_use - palette to use for the Mollweide plot(s).
%   caxis_mollweide - range to use in clim,
%   color_label_mollweide - label to use for Mollweide colorbar.
%   xlim_zonal_mean - range to use for x-axis, mean LL, in zonal mean plot;
%    if empty use default.
%   ylim_zonal_mean - range to use for y-axis, latitude, in zonal mean plot.
%   Title_1 - title of first Mollweide plot,
%   Title_2 - title of second Mollweide plot; if only one Moolweide plot this
%    variable shoule be empty [],
%   PrintName - name to use for the pdf written out; if empty no pdf made,
%
% OUTPUT  

FigDir_to_use = FigDir;
PrintName_to_use = PrintName;

plot_config = [1 1];

if sum(subplot_details{2,2}) > 0
    plot_config = [2 2];
end

if sum(subplot_details{1,2}) > 0  & sum(subplot_details{2,1}) == 0
    plot_config = [1 2];
end

if sum(subplot_details{1,2}) == 0  & sum(subplot_details{2,1}) > 0
    plot_config = [2 1];
end

% % % Subplot_1_to_use = 1;
% % % Subplot_2_to_use = 2;

if isempty(ll_thresholds)
    ll_thresholds_to_use = [];
else
    ll_thresholds_to_use = ll_thresholds;
end

% Configure variables if zonal plot(s) requested.

if plot_config(2) == 2

    % Do zonal average for the first variable.

    delta = 1;

    j = 0;
    for i=-90:delta:90-delta
        nn = find(lat>=i & lat<i+delta);
        j = j + 1;
        mean_diff{1}(j) = mean(var_1(nn),'omitnan');
        mean_lat{1}(j) = i + delta/2;
    end
    
    % Do zonal average for the second variable but first constrain with
    % ll_thresholds if present.
    
    if isempty(ll_thresholds)
        var_2_mod = var_2;
        lat_mod = lat;
        lon_mod = lon;
    else
        nn = find(var_2<ll_thresholds(1) | var_2>ll_thresholds(2));
        var_2_mod = var_2(nn);
        lat_mod = lat(nn);
        lon_mod = lon(nn);
    end

    j = 0;
    for i=-90:delta:90-delta
        nn = find(lat_mod>=i & lat_mod<i+delta);
        j = j + 1;
        mean_diff{2}(j) = mean( var_2_mod(nn), 'omitnan');
        mean_lat{2}(j) = i + delta/2;
    end
    
    % Configures other variables.
    
    FigDir_to_use = [];
    PrintName_to_use = [];

% % %     if plot_config(1) == 1 
% % %         Subplot_1_to_use = [1 0.05 .1 .7 .8];
% % %     else
% % %         Subplot_1_to_use = [1 .1 .55 .7 .4];
% % %         Subplot_2_to_use = [2 .1 .05 .7 .4];
% % %     end
end

% Now make Mollweide plots

if isempty(subplot_details)
% % %     HEALPix_plot_Mollweid( FigNo, [], lat, lon, var_1, dot_size, caxis_mollweide, ...
% % %         Title_1, FigDir_to_use, PrintName_to_use, ll_thresholds_to_use, color_label_mollweide, lat_range, lon_range)
% % % else
% % %     if plot_config(1) == 1
% % %         HEALPix_plot_Mollweid( FigNo, Subplot_1_to_use, lat, lon, var_1, dot_size, caxis_mollweide, ...
% % %             Title_1, FigDir_to_use, PrintName_to_use, ll_thresholds_to_use, color_label_mollweide, lat_range, lon_range)
% % %     else
% % %         HEALPix_plot_Mollweid( FigNo, Subplot_1_to_use, lat, lon, var_1, dot_size, caxis_mollweide, Title_1, ...
% % %             FigDir_to_use, PrintName_to_use, [0 0], color_label_mollweide, lat_range, lon_range)
% % %         HEALPix_plot_Mollweid( FigNo, Subplot_2_to_use, lat, lon, var_2, dot_size, caxis_mollweide, Title_2, ...
% % %             FigDir_to_use, PrintName_to_use, ll_thresholds_to_use, color_label_mollweide, lat_range, lon_range)
% % %     end
    HEALPix_plot_Mollweid( FigNo, subplot_details, lat, lon, var_1, dot_size, colormap_to_use, caxis_mollweide, ...
        Title_1, FigDir_to_use, PrintName_to_use, ll_thresholds_to_use, color_label_mollweide, 24, lat_range, lon_range)
else
    if plot_config(1) == 1
        figure(FigNo)
        clf
        HEALPix_plot_Mollweid( FigNo, subplot_details{1,1}, lat, lon, var_1, dot_size, colormap_to_use, caxis_mollweide, ...
            Title_1, FigDir_to_use, PrintName_to_use, ll_thresholds_to_use, color_label_mollweide, 24, lat_range, lon_range)
    else
        figure(FigNo)
        clf
        if iscell(color_label_mollweide)
            HEALPix_plot_Mollweid( FigNo, subplot_details{1,1}, lat, lon, var_1, dot_size, colormap_to_use, caxis_mollweide(1,:), Title_1, ...
                FigDir_to_use, PrintName_to_use, [0 0], color_label_mollweide{1}, 30, lat_range, lon_range)
            HEALPix_plot_Mollweid( FigNo, subplot_details{2,1}, lat, lon, var_2, dot_size, colormap_to_use, caxis_mollweide(2,:), Title_2, ...
                FigDir_to_use, PrintName_to_use, ll_thresholds_to_use, color_label_mollweide{2}, 30, lat_range, lon_range)
        else
            HEALPix_plot_Mollweid( FigNo, subplot_details{1,1}, lat, lon, var_1, dot_size, colormap_to_use, caxis_mollweide, Title_1, ...
                FigDir_to_use, PrintName_to_use, [0 0], color_label_mollweide, 30, lat_range, lon_range)
            HEALPix_plot_Mollweid( FigNo, subplot_details{2,1}, lat, lon, var_2, dot_size, colormap_to_use, caxis_mollweide, Title_2, ...
                FigDir_to_use, PrintName_to_use, ll_thresholds_to_use, color_label_mollweide, 30, lat_range, lon_range)
        end
    end
end

% Return if no zonal plots requested.

if plot_config(2) == 2
    
    %% Add zonal mean plots if requested.
    
    % At this point Mollweide plots have been made and, if no zonal means, the
    % function has returned to the calling program. Have the positions vectors
    % for the zonal mean plots been passed in? If not, get them and, if they
    % have, use them dummy.
    
    for iSubplot=1:plot_config(1)
        
        if iSubplot == 1
            if plot_config(1) == 1
                title_prefix = 'b) ';
            else
                title_prefix = 'c) ';
            end
        else
            title_prefix = 'd) ';
        end
        
        if sum(subplot_details{iSubplot,2}) == 0
            subplot_details(iSubplot,:) = add_zonal_mean( mean_diff{iSubplot}, mean_lat{iSubplot}, xlim_zonal_mean, ylim_zonal_mean, title_prefix);
        else
%             subplot( 'position', [0.82 subplot_details(iSubplot,4)/subplot_details(iSubplot,2) 0.12 (subplot_details(iSubplot,6)-subplot_details(iSubplot,4))/subplot_details(iSubplot,2)])
            subplot( 'position', subplot_details{iSubplot,2}(3:end))
            
            plot(mean_diff{iSubplot}, mean_lat{iSubplot}, linewidth=2)
            hold on
            plot([0 0], [-90 90], color=[.7 .7 .7], linewidth=1)
            
            grid on
            
            if isempty(xlim_zonal_mean) == 0
                xlim(xlim_zonal_mean)
            end
            
            if isempty(ylim_zonal_mean) == 0
                ylim(ylim_zonal_mean)
            end
            
            set(gca, fontsize=24)
            
            ylabel('Latitude')
            xlabel('\boldmath$\overline{\widetilde{LL}}$', interpreter='latex')
            
            title(['\bf{' title_prefix 'Zonal Mean}'], fontsize=30, interpreter='latex')
            
        end
    end    
else
    return
end

%% Finally print the figure if PrintName is not empty.

if isempty(PrintName) == 0
    temp1 = strrep( PrintName, '.', '_');
    temp2 = strrep( temp1, '-', 'm_');
% % %     if plot_config(1) == 1
% % %         filename_out = [FigDir 'pcc_1_' temp2];
% % %     else
% % %         filename_out = [FigDir 'pcc_2_' temp2];
% % %     end
    filename_out = [FigDir temp2];
    print( filename_out, '-dpng')
    
    fprintf('Printing figure: %s\n', filename_out)
end

return
