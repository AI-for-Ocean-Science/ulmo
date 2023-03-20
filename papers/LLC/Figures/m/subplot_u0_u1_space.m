function subplot_u0_u1_space( fig_no, input_table, DT_range, PCC_AXIS, num_threshold, cloud_threshold, do_variable_plots, do_UMAP_density_plots)
% subplot_u0_u1_space - Read and plot data from UMAP space for MODIS L2 - PCC
%
% This script will extract variables of interest a Matlab table and plot
% the mean of the specified variable for the specified data range.
%
% INPUT
%   fig_no - figure number to use for the plot. If the scatterplot is
%    requested, it will increment the figure number counter by 1.
%   input_table - the UMAP table we are dealing with.
%   DT_range - 2 element vector specifying the range of DT values to
%    consider before binning in u0 , u1.
%   PCC_AXIS - 4 element axis vector. 
%   num_threshold = threshold of number of values. Means will be calculated
%    for bins with numbers exceeding this threshold, other nan.
%   cloud_threshold - only cutouts with cloud fractions less than this
%    value will be used.
%   do_variable_plots - multi UMAP plots of variables.
%   do_UMAP_density_plots - multi UMAP plots of the number of cutouts in DT
%    range.
% %   do_scatter_plot - 1 to scatterplot the variable of interest. These plots
% %    may have a lot of points so take long to render. Either 0 or not
% %    passed in and plot will not be done.
%
% OUTPUT
%   None
%
% Examples
%   fi = '/Volumes/Aqua-1/SST_OOD/MODIS_L2/Tables/MODIS_SSL_96clear_v4_DTall.parquet';
%	input_table = parquetread(fi, 'SelectedVariableNames' , {'lat' 'lon' 'datetime' 'LL' 'clear_fraction' 'mean_temperature' 'T10' 'T90' 'U0' 'U1' 'zonal_slope' 'merid_slope', 'US0', 'US1'});
%
%   DT_range = [0.5 1.0];
%   cloud_threshold = 0.015;
%   num_threshold = 1000;
%   PCC_AXIS = [-4 9 -4 6];
%   subplot_u0_u1_space( 1, input_table, DT_range, PCC_AXIS, num_threshold, cloud_threshold, do_variable_plots, do_UMAP_density_plots)

% xedges = [-1:0.25:11.5];
% yedges = [-3:0.25:6.5];
xedges = [PCC_AXIS(1):0.25:PCC_AXIS(2)];
yedges = [PCC_AXIS(3):0.25:PCC_AXIS(4)];

if do_variable_plots
    % First determine the cutouts to consider based on DT
    
    T10 = input_table{:,7};
    T90 = input_table{:,8};
    DT = T90 - T10;
    
    cloud_fraction = input_table{:,5};
    
    nn = find(DT > DT_range(1) & DT < DT_range(2) & cloud_fraction < cloud_threshold);
    
    % Extract variables from the Matlab table for this range of DT
    
    Lat = input_table{:,1}(nn);
    Lon = input_table{:,2}(nn);
    LL = input_table{:,4}(nn);
    cloud_fraction = input_table{:,5}(nn);
    mean_temperature = input_table{:,6}(nn);
    T10 = input_table{:,7}(nn);
    T90 = input_table{:,8}(nn);
    u0 = input_table{:,9}(nn);
    u1 = input_table{:,10}(nn);
    zonal_slope = input_table{:,11}(nn);
    merid_slope = input_table{:,12}(nn);
    us0 = input_table{:,13}(nn);
    us1 = input_table{:,14}(nn);
    
    % And derived variables from the above.
    
    DT = T90 - T10;
    xx = [zonal_slope, merid_slope];
    alpha_min = min(xx,[],2);
    
    % Now bin in us0,us1 space. Bins are 0.5x0.5 in size.
    
    [ num_per_bin, xedges, yedges, locsx, locsy] = histcounts2( us0, us1, xedges, yedges);
    
    % Now do the plots. Start with the population plot.
    
    figure(fig_no)
    clf
    
    subplot(231)
    
    histogram2( us0, us1, xedges, yedges, facecolor='flat', ...
        DisplayStyle='tile', ShowEmptyBins='off');
    colorbar
    grid on
    xlabel('U0')
    ylabel('U1')
    set(gca, ydir='normal', fontsize=24)
%     axis([-1 12 -3 6 0 50000])
    axis(PCC_AXIS)

    title( 'Number of Cutouts/Bin', fontsize=30)
    
    % Specify the variables to plot
    
    % var_name = {'mean_temperature' 'DT' 'cloud_fraction' 'alpha_min' 'Lat' 'Lon'};
    var_name = {'mean_temperature' 'DT' 'cloud_fraction' 'alpha_min' 'Lat' 'Lon'};
    
    max_ix = length(xedges) - 1;
    max_jy = length(yedges) - 1;
    
    % for iVarName=1:6
    for iVarName=1:5
        eval(['selected_variable = ' var_name{iVarName} ';'])
        
        if iVarName == 5
            selected_variable = abs(selected_variable);
        end
        
        for ix=1:max_ix
            for jy=1:max_jy
                nn = find(locsx==ix & locsy==jy);
                if length(nn) >= num_threshold
                    var_mean(ix,jy) = mean(selected_variable(nn));
                else
                    var_mean(ix,jy) = nan;
                end
            end
        end
        
        subplot(2, 3, iVarName+1)
        
        [x_var_length, y_var_length] = size(var_mean);
        x_dim = repmat( xedges(1:end-1)',1, y_var_length);
        y_dim = repmat( yedges(1:end-1), x_var_length, 1);
        
        %     imagesc(xedges(1:end-1), yedges(1:end-1), var_mean')
        
        pcolor( x_dim, y_dim, var_mean)
        shading flat
        
        colorbar
        grid on
        xlabel('U0')
        ylabel('U1')
        set(gca, ydir='normal', fontsize=24)
%         axis([-1 12 -3 6])
        axis(PCC_AXIS)

        if iVarName == 5
            title( ['|' strrep(var_name{iVarName}, '_', '\_') '|'], fontsize=30)
        else
            title( strrep(var_name{iVarName}, '_', '\_'), fontsize=30)
        end
        
%         % Scatter plot alpha_min vs u0,u1 if requested.
%         
%         if exist('do_scatter_plot') == 1
%             if do_scatter_plot
%                 fig_no = fig_no + 2;
%                 figure(fig_no)
%                 scatter(us0(nn), us1(nn), 5, alpha_min(nn))
%                 colorbar
%                 set(gca,fontsize=16)
%                 axis([-1 12 -3.5 4.5])
%                 grid on
%                 caxis([-3 0.2])
%             end
%         end
    end
    
    sgtitle( ['UMAP Plots for ' num2str(DT_range(1)) ' < \Delta{T} < ' num2str(DT_range(2)) ', Cloud\_fraction < ' num2str(cloud_threshold) ' And # Cutouts/Bin >' num2str(num_threshold)], fontsize=30)
end

%% Now do density plots

if do_UMAP_density_plots
    figure(fig_no+1)
    clf
    
    % Reload the data from the table for variables used here.
    
    T10 = input_table{:,7};
    T90 = input_table{:,8};    
    DT = T90 - T10;

    us0 = input_table{:,13};
    us1 = input_table{:,14};
    
    cloud_fraction = input_table{:,5}; 
            
    T_range = [0 0.5; 0.5 1.0; 1.0 1.5; 1.5 2.5; 2.5 4.0; 4.0 100.0];
    
    for iT=1:6
        
        % Get the cutouts in this DT range
        
        nn = find(DT > T_range(iT,1) & DT < T_range(iT,2) & cloud_fraction < cloud_threshold);
                
        subplot(2, 3, iT)
        
        histogram2( us0(nn), us1(nn), xedges, yedges, facecolor='flat', ...
            DisplayStyle='tile', ShowEmptyBins='off');
        colorbar
        grid on
        xlabel('U0')
        ylabel('U1')
        set(gca, ydir='normal', fontsize=24)
%         axis([-1 12 -3 6 0 50000])
        axis(PCC_AXIS)
        title( ['Cutouts/Bin ' num2str(T_range(iT,1)) ' < \Delta{T} < ' num2str(T_range(iT,2))], fontsize=30)
    end
    
    sgtitle( ['# Cutouts/Bin for Cloud\_fraction < ' num2str(cloud_threshold)], fontsize=30)
end
    