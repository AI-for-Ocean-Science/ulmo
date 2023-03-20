function examine_u0_u1_space( fig_no, input_table, DT_range, var_name, scatter_plot, num_threshold, cloud_threshold)
% examine_u0_u1_space - Read and plot data from UMAP space for MODIS L2 - PCC
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
%   var_name - the name of the variable to be averaged.
%   scatter_plot - 1 to scatterplot the variable of interest. These plots
%    may have a lot of points so take long to render.
%   num_threshold = threshold of number of values. Means will be calculated
%    for bins with numbers exceeding this threshold, other nan.
%   cloud_threshold - only cutouts with cloud fractions less than this
%    value will be used.
%
% OUTPUT
%   None
%
% Examples
%   fi = '/Volumes/Aqua-1/SST_OOD/MODIS_L2/Tables/MODIS_SSL_96clear_v4_DTall.parquet';
%	input_table = parquetread(fi, 'SelectedVariableNames' , {'lat' 'lon' 'datetime' 'LL' 'cloud_fraction' 'mean_temperature' 'T10' 'T90' 'U0' 'U1' 'zonal_slope' 'merid_slope', 'US0', 'US1'});
%
%   DT_range = [0.5 1.0];
%   cloud_threshold = 0.015;
%   num_threshold = 1000;
%
%   examine_u0_u1_space( 1, input_table, DT_range, 'mean_temperature', 0, num_threshold, cloud_threshold)
%   examine_u0_u1_space( 2, input_table, DT_range, 'DT', 0, num_threshold, cloud_threshold)
%   examine_u0_u1_space( 3, input_table, DT_range, 'alpha_min', 0, num_threshold, cloud_threshold)
%   examine_u0_u1_space( 4, input_table, DT_range, 'Lat', 0, num_threshold, cloud_threshold)
%   examine_u0_u1_space( 5, input_table, DT_range, 'cloud_fraction', 0, num_threshold, cloud_threshold)
%   examine_u0_u1_space( 6, input_table, DT_range, 'zonal_slope', 0, num_threshold, cloud_threshold)

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

[ num_per_bin, xedges, yedges, locsx, locsy] = histcounts2( us0, us1, [-1:0.25:10], [-3:0.25:6]);

% Specify the variable to average by us0,us1 bin. Then calculate the mean 
% of selected variable for bins with more than the threshold number of 
% cutouts.

eval(['selected_variable = ' var_name ';'])

max_ix = max(locsx);
max_iy = max(locsy);

for ix=1:max_ix
    for jy=1:max_iy
        nn = find(locsx==ix & locsy==jy);
        if length(nn) >= num_threshold
            var_mean(ix,jy) = mean(selected_variable(nn));
        else
            var_mean(ix,jy) = nan;
        end
    end
end

figure(fig_no)
imagesc(xedges(1:end-1), yedges(1:end-1), var_mean')
colorbar
grid on
xlabel('U0')
ylabel('U1')
set(gca, ydir='normal', fontsize=24)
% axis([-0.5 10.2 -3.5 4.5])
axis([-1 11 -3 6])
title( [strrep(var_name, '_', '\_') ' for ' num2str(DT_range(1)) ' < \Delta{T} < ' num2str(DT_range(2))], fontsize=30)

% Scatter plot alpha_min vs u0,u1 if requested.

if scatter_plot
    fig_no = fig_no + 1;
    figure(fig_no)
    scatter(us0(nn), us1(nn), 5, alpha_min(nn))
    colorbar
    set(gca,fontsize=16)
    axis([-1 10 -3.5 4.5])
    grid on
    caxis([-3 0.2])
end


