function fig_loc_values = add_zonal_mean( var_to_plot, lat, xlim_zonal_mean, ylim_zonal_mean, title_prefix)
% add_zonal_median - will ask for location in figure for zonal mean plot and plot it there - pcc.
%
% INPUT
%   LL_Range - range of LL values to use for raw median data
% %   LL_zonal_mean_range - range of LL values to use for zonal averages of median data
%
% EXAMPLES
%   HEALPix_plots_for_kat( 1, 3, 4, [-1000 700], [-500 200])
%   HEALPix_plots_for_kat( 11, 3, 5, [-1000 700], [-500 200])
%

Instructions = 'Click on the upper right hand corner of the figure.';
ur = get_pixel_loc_in_figure(Instructions);

Instructions = 'Click on [0, -60 or -90] depending on range of data in the Mollweid plot.';
cp_m = get_pixel_loc_in_figure(Instructions);

Instructions = 'Click on [0, 60 or 90] depending on range of data in the Mollweid plot.';
cp_p = get_pixel_loc_in_figure(Instructions);

subplot( 'position', [0.82 cp_m(2)/ur(2) 0.12 (cp_p(2)-cp_m(2))/ur(2)])

plot(var_to_plot, lat, linewidth=2)
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
xlabel('$\overline{\widetilde{LL}}$', interpreter='latex')

% % %             title([title_prefix 'Zonal Mean'], fontsize=30, fontweight='normal')
title([title_prefix 'Zonal Mean'], fontsize=30, interpreter='latex')

fig_loc_values = [ur, cp_m, cp_p];

return


