% HEALPix_sigmas analysis of sigmas - PCC
%
% Head_tail directory to use.

get_reordered_llc_HEALPix = 1;

% head_tail_dir = '2_1_2012-1_31_2015_and_1_1_2018-12_31_2020';
% head_tail_dir = '2_1_2012-1_31_2016_and_1_1_2017-12_31_2020';
input_HEALPix_data = '2022-Nov';

% To make plots that go from -75 to 60 set the following variable to 1

constrain_to_57 = 1;

% The idea here is to investigate sigma LL as a function of data source,
% time, location,...

% global colormap_to_use masked_cc

% mean_and_sigma is 1 for the first version of head/tail data, which
% included mean and standard deviation information for each HEALPix cell,
% and 0 for the new version, which does not include these.

mean_and_sigma = 0;

masked_cc = 0.6;

% lg = 0.7;
% colormap_to_use_ll = redblue(256);
% colormap_to_use_ll(1,:) = [lg lg lg];

colormap_to_use_default = colormap(jet);
colormap_to_use_ll = coolwarm(256);
colormap_to_use_num = colormap(parula);
colormap_to_use_bathy = colormap(jet);

min_num = 5;

Normalize = 0;

% plot_1 = 1;
% plot_11_13 = 1;
% plot_21_31 = 1;
% plot_41_45 = 1;
% plot_51_54 = 1;
% plot_81 = 1;  % Galleries
% plot_61 = 1;

plot_1 = 0;
plot_11_13 = 0;
plot_21_31 = 0;
plot_41_45 = 0;
plot_51_54 = 0;
plot_81 = 0;  % Galleries
plot_61 = 1;

FigPaperDir = '~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Figures/For_Kat/for_paper/';

color_label_size_large = 30;
color_label_size_small = 24;

% caxis_mollweide_600_600 = [-600 600];
caxis_mollweide_600_600 = [-600 600];
xlim_zonal_mean_600_600 = [-600 600];

if constrain_to_57
    lat_range = [-75 60];
    ylim_zonal_mean = [-80 82];
else
    lat_range = [];
    ylim_zonal_mean = [-90 90];
end

xlim_zonal_mean_450_450 = [-450 450];

caxis_mollweide_550_1200 = [-550 1200];
xlim_zonal_mean_500_1000 = [-500 1000];

caxis_mollweide_550_1200_600_600 = [-550 1200; -600 600];

caxis_mollweide_400_400 = [-400 400];

caxis_mollweide_600_200_0_3 = [-600 200; 0 3];
caxis_mollweide_600_200_0_4 = [-600 200; 0 4];

caxis_mollweide_600_0_0_3 = [-600 0; 0 3];

caxis_mollweide_0_3 = [0 3];
caxis_mollweide_0_4 = [0 4];

% which_thresholds_to_use = 'diff_at_100';
% which_thresholds_to_use = 'best_fit_to_head_tail_diff';
which_thresholds_to_use = 'sigma_head_tail_diff';
% which_thresholds_to_use = 'llc_gt_50_percent_viirs';

XLIM_MEAN = [-1000 1200];
XLIM_SIGMA = [0 500];

fig_no_offset = 0;

color_range = [-750 750];
dot_size = 20;

% Initialize subplot_details: figure size: 36.2 x (top-to-bottom, 30.3)

subplot_details_1x1{1,1} = [ 1000  640  0.05  0.1  0.90  0.80];
subplot_details_1x1{1,2} = [    0    0    0    0    0    0];
subplot_details_1x1{2,1} = [    0    0    0    0    0    0];
subplot_details_1x1{2,2} = [    0    0    0    0    0    0];

subplot_details_1x2{1,1} = [1480  670  0.05  0.1  0.75  0.80];
subplot_details_1x2{1,2} = [1480  670  0.82  0.185  0.12  0.706];
subplot_details_1x2{2,1} = [   0    0    0    0    0     0];
subplot_details_1x2{2,2} = [   0    0    0    0    0     0];

subplot_details_2x1{1,1} = [1557  1283  0.05 0.55 0.9 0.4];
subplot_details_2x1{1,2} = [   0     0    0     0    0     0]; ...
    subplot_details_2x1{2,1} = [1557  1283  0.05 0.05 0.9 0.4]; ...
    subplot_details_2x1{2,2} = [   0     0    0     0    0     0];

subplot_details_2x2{1,1} = [1450  1335  0.05  0.550  0.75  0.400]; % Geo 1
subplot_details_2x2{1,2} = [1450  1335  0.82  0.600  0.12  0.347]; % Zonal 1
subplot_details_2x2{2,1} = [1450  1335  0.05  0.050  0.75  0.400]; % Geo 2
subplot_details_2x2{2,2} = [1450  1335  0.82  0.091  0.12  0.347]; % Zonal 2

% Specialty plots

subplot_details_patagonia{1,1} = [1557  1283  0.05 0.55 0.9 0.4];
subplot_details_patagonia{1,2} = [   0     0    0     0    0     0]; ...
    subplot_details_patagonia{2,1} = [1557  1283  0.05 0.05 0.9 0.4]; ...
    subplot_details_patagonia{2,2} = [   0     0    0     0    0     0];

% Define the squares for the Equatorial and Gulf Stream regions hilighted.

xxeqsq = [255 245 245 255 255];
yyeqsq = [-2 -2 2 2 -2];

xxgssq = [280 305 305 280 280];
yygssq = [32 32 40 40 32];

xxacsq = [114 124 124 114 114];
yyacsq = [-55 -55 -49 -49 -55];

xxacsub1 = [119 124 124 119 119];
yyacsub1 = [-55 -55 -53 -53 -55];
xxacsub2 = [115 117 117 115 115];
yyacsub2 = [-52 -52 -49 -49 -52];

% Load bathy information

% % % load('~/Dropbox/Data/Bathymetry/200m_isobath_off_US_East_Coast');
load('200m_isobath_off_US_East_Coast');

% Get the mean and extremes of the Gulf Stream.

% temp_gs = load('~/Dropbox/Data/Gulf_Stream/path_stats');
% GS = temp_gs.mean_1982_through_1993;
% GS_north = temp_gs.northern_limit_1982_through_1986;
% GS_south = temp_gs.southern_limit_1982_through_1986;

% % % temp_gs = load('~/Dropbox/Data/Gulf_Stream/path_stats_1982_1999.mat');
temp_gs = load('path_stats_1982_1999.mat');
GS = [temp_gs.mean_lon; temp_gs.mean_lat]';
GS_north = [temp_gs.mean_lon; temp_gs.max_lat]';
GS_south = [temp_gs.mean_lon; temp_gs.min_lat]';
GS_99 = [temp_gs.mean_lon; temp_gs.lat_99]';
GS_01 = [temp_gs.mean_lon; temp_gs.lat_01]';


% If you want to use a different figure size, set these two variables to 0.

subplot_details_1x2_to_use = subplot_details_1x2;
subplot_details_2x2_to_use = subplot_details_2x2;

%% Get the data

% Get longitude and latitude for HEALPix Cells.

% % % load ~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/tail_end/hp_lats_head_v98.mat
% % % load ~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/tail_end/hp_lons_head_v98.mat
load hp_lats_head_v98.mat
load hp_lons_head_v98.mat
Lat = hp_lats_head_v98;
Lon = hp_lons_head_v98;

Lat_o = Lat;
Lon_o = Lon;

%% Get Equatorial cutouts

if get_reordered_llc_HEALPix
%     regional_cutout_structure = load('~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/gulf_stream_and_equatorial_regions');
% % %     regional_cutout_structure = load('~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/regional_cutouts_and_metadata');
    regional_cutout_structure = load('regional_cutouts_and_metadata');
    
    eq_viirs_cutouts = regional_cutout_structure.eq_viirs_cutouts_out;
    eq_viirs_metadata = regional_cutout_structure.eq_viirs_metadata_out;
    
    gs_viirs_cutouts = regional_cutout_structure.gs_viirs_cutouts_out;
    gs_viirs_metadata = regional_cutout_structure.gs_viirs_metadata_out;
    
    acc_viirs_cutouts = regional_cutout_structure.acc_viirs_cutouts_out;
    acc_viirs_metadata = regional_cutout_structure.acc_viirs_metadata_out;
    
    eq_llc_cutouts = regional_cutout_structure.eq_llc_cutouts_out;
    eq_llc_metadata = regional_cutout_structure.eq_llc_metadata_out;
    
    gs_llc_cutouts = regional_cutout_structure.gs_llc_cutouts_out;
    gs_llc_metadata = regional_cutout_structure.gs_llc_metadata_out;
    
    acc_llc_cutouts = regional_cutout_structure.acc_llc_cutouts_out;
    acc_llc_metadata = regional_cutout_structure.acc_llc_metadata_out;
else
% % %     fi = '~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/Cutouts/2022-Nov/viirs_eq_rect_cutouts.h5';
    fi = 'viirs_eq_rect_cutouts.h5';
    [eq_viirs_cutouts, eq_viirs_metadata] = get_cutouts_and_metadata(fi, [4 5 17 18 19]);
    
% % %     fi = '~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/Cutouts/2022-Nov/llc_eq_rect_cutouts.h5';
    fi = 'llc_eq_rect_cutouts.h5';
    [eq_llc_cutouts,   eq_llc_metadata] = get_cutouts_and_metadata(fi, [4 5 17 18 27]);
end

%% Get the ACC cutouts

if get_reordered_llc_HEALPix == 0
    anomalous_healpix_cell_number = {'44318' '44319' '44513' '44514'};
    
    [anomalous_acc_viirs_cutouts, anomalous_acc_viirs_metadata, unique_anomalous_acc_viirs_cutouts, unique_anomalous_acc_viirs_metadata] = get_acc_cutouts( 'VIIRS', anomalous_healpix_cell_number);
    [anomalous_acc_llc_cutouts, anomalous_acc_llc_metadata, unique_anomalous_acc_llc_cutouts, unique_anomalous_acc_llc_metadata] = get_acc_cutouts( 'LLC', anomalous_healpix_cell_number);
    
    agreeing_healpix_cell_number = {'43282' '43497'};
    
    [agreeing_acc_viirs_cutouts, agreeing_acc_viirs_metadata, unique_agreeing_acc_viirs_cutouts, unique_agreeing_acc_viirs_metadata] = get_acc_cutouts( 'VIIRS', agreeing_healpix_cell_number);
    [agreeing_acc_llc_cutouts, agreeing_acc_llc_metadata, unique_agreeing_acc_llc_cutouts, unique_agreeing_acc_llc_metadata] = get_acc_cutouts( 'LLC', agreeing_healpix_cell_number);
end

%% Get Gulf Stream cutouts

if get_reordered_llc_HEALPix == 0
% % %     fi = '~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/Cutouts/2022-Nov/viirs_gulf_rect_cutouts.h5';
    fi = 'viirs_gulf_rect_cutouts.h5';
    [gs_viirs_cutouts, gs_viirs_metadata] = get_cutouts_and_metadata(fi, [4 5 17 18 19]);
    
% % %     fi = '~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/Cutouts/2022-Nov/llc_gulf_rect_cutouts.h5';
    fi = 'llc_gulf_rect_cutouts.h5';
    [  gs_llc_cutouts,  gs_llc_metadata] = get_cutouts_and_metadata(fi, [4 5 17 18 27]);
end

%% Get the stats for each HEALPix cell.

% Head VIIRS 2012-2015

% % % fi_head = ['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' input_HEALPix_data '/head_viirs.csv'];
fi_head = 'head_viirs.csv';
T_head = readtable(fi_head);

% load(['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' head_tail_dir '/evts_head.mat'])
% N = x';
% tt = zeros(size(N));
% N(N<-100000) = 0;
%
% load(['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' head_tail_dir '/meds_head.mat'])
% median = x';
% median(median<-10000) = 0;
%
% T_head = table( tt, tt, median, N, tt);

% Tail VIIRS 2018-2020

% % % fi_tail = ['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' input_HEALPix_data '/tail_viirs.csv'];
fi_tail = 'tail_viirs.csv';
T_tail = readtable(fi_tail);

% load(['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' head_tail_dir '/evts_tail.mat'])
% N = x';
% tt = zeros(size(N));
% N(N<-100000) = 0;
%
% load(['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' head_tail_dir '/meds_tail.mat'])
% median = x';
% median(median<-10000) = 0;
%
% T_tail = table( tt, tt, median, N, tt);
% clear median

% VIIRS 2012-2020

% % % fi_viirs = ['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' input_HEALPix_data '/all_viirs.csv'];
fi_viirs = 'all_viirs.csv';

T_viirs = readtable(fi_viirs);

% LLC

% % % fi_llc = ['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/' input_HEALPix_data '/all_llc.csv'];
fi_llc = 'all_llc.csv';

T_llc = readtable(fi_llc);

% MODIS 98%

% % % fi_modis = '~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/98_modis.csv';
fi_modis = '98_modis.csv';

T_modis = readtable(fi_modis);

%% Now find only matches and redefine variables based on matches and minimum number of Cutouts/Cell. all & llc first.

nn =      find(abs(T_viirs.N - T_llc.N)<=6 & T_viirs.N ~=0 & T_llc.N ~=0);
nBad_ht = find(abs(T_viirs.N - T_llc.N) >6 | T_viirs.N ==0 | T_llc.N ==0);

lon_viirs = Lon(nn);
lat_viirs = Lat(nn);

mean_viirs = T_viirs.mean(nn);
sigma_viirs = T_viirs.sigma(nn);
median_viirs = T_viirs.median(nn);
number_viirs = T_viirs.N(nn);

lon_llc = Lon(nn);
lat_llc = Lat(nn);

mean_llc = T_llc.mean(nn);
sigma_llc = T_llc.sigma(nn);
median_llc = T_llc.median(nn);
number_llc = T_llc.N(nn);

%% And for modis-viirs comparison

nn = find(T_viirs.N>0 & T_modis.N>0 & T_viirs.N ~=0 & T_modis.N ~=0);
viirs_modis_diff = T_viirs.median(nn) - T_modis.median(nn);

lat_viirs_modis = Lat(nn);
lon_viirs_modis = Lon(nn);

nn = find(T_viirs.N>5 & T_modis.N>5 & T_viirs.N ~=0 & T_modis.N ~=0);
viirs_modis_diff_gt0 = T_viirs.median(nn) - T_modis.median(nn);

lat_viirs_modis_gt0 = Lat(nn);
lon_viirs_modis_gt0 = Lon(nn);

%% Next find only matches with at least min_num of cutouts/HEALPix cell for both VIIRS and LLC and redefine variables based on matches and minimum number of Cutouts/Cell. all & llc first.

nn =      find(abs(T_viirs.N - T_llc.N)<=6 & T_viirs.N >= min_num & T_llc.N >= min_num);
nBad_vl = find(abs(T_viirs.N - T_llc.N) >6 | T_viirs.N  < min_num | T_llc.N  < min_num);

lon_viirs_gt0 = Lon(nn);
lat_viirs_gt0 = Lat(nn);

mean_viirs_gt0 = T_viirs.mean(nn);
sigma_viirs_gt0 = T_viirs.sigma(nn);
median_viirs_gt0 = T_viirs.median(nn);
number_viirs_gt0 = T_viirs.N(nn);

lon_llc_gt0 = Lon(nn);
lat_llc_gt0 = Lat(nn);

mean_llc_gt0 = T_llc.mean(nn);
sigma_llc_gt0 = T_llc.sigma(nn);
median_llc_gt0 = T_llc.median(nn);
number_llc_gt0 = T_llc.N(nn);

% And the difference.

viirs_llc_diff = mean_viirs - mean_llc;
viirs_llc_diff_gt0 = mean_viirs_gt0 - mean_llc_gt0;

%% Do the same for head and tail; any non-zero matchups first

mm = find( T_head.N ~= 0 & T_tail.N ~= 0);

lon_head = Lon(mm);
lat_head = Lat(mm);

if mean_and_sigma
    mean_head = T_head.mean(nn);
    sigma_head = T_head.sigma(nn);
end
median_head = T_head.median(nn);
number_head = T_head.N(nn);

lon_tail = Lon(mm);
lat_tail = Lat(mm);

if mean_and_sigma
    mean_tail = T_tail.mean(nn);
    sigma_tail = T_tail.sigma(nn);
end
median_tail = T_tail.median(nn);
number_tail = T_tail.N(nn);

%% For head and tail but requiring at least min_num cutouts/HEALPix cell

mm = find( T_head.N >= min_num & T_tail.N >= min_num);

lon_head_gt0 = Lon(mm);
lat_head_gt0 = Lat(mm);

if mean_and_sigma
    mean_head_gt0 = T_head.mean(mm);
    sigma_head_gt0 = T_head.sigma(mm);
end
median_head_gt0 = T_head.median(mm);
number_head_gt0 = T_head.N(mm);

lon_tail = Lon(mm);
lat_tail = Lat(mm);

if mean_and_sigma
    mean_tail_gt0 = T_tail.mean(mm);
    sigma_tail_gt0 = T_tail.sigma(mm);
end
median_tail_gt0 = T_tail.median(mm);
number_tail_gt0 = T_tail.N(mm);

% And the difference

if mean_and_sigma
    head_tail_diff = mean_head - mean_tail;
    head_tail_diff_gt0 = mean_head_gt0 - mean_tail_gt0;
else
    head_tail_diff = median_head - median_tail;
    head_tail_diff_gt0 = median_head_gt0 - median_tail_gt0;
end

%% Correct LLC LL values. Do this cells with min_num values.

% Need to get offset to apply to LLC LL values. Start by binning the data

istep = 100;
ii = 0;

for icount=1:istep:length(median_viirs_gt0)
    ii = ii + 1;
    bin_mean_viirs_gt0(ii) = mean(median_viirs_gt0(icount:min(length(median_viirs_gt0),icount+istep)), 'omitnan');
    bin_mean_llc_gt0(ii) = mean(median_llc_gt0(icount:min(length(median_viirs_gt0),icount+istep)), 'omitnan');
    
    new_median_llc_gt0(icount:min(length(median_viirs_gt0),icount+istep)) = median_llc_gt0(icount:min(length(median_viirs_gt0),icount+istep)) -  bin_mean_llc_gt0(ii) + bin_mean_viirs_gt0(ii);
    new_median_viirs_gt0(icount:min(length(median_viirs_gt0),icount+istep)) = median_viirs_gt0(icount:min(length(median_viirs_gt0),icount+istep)) -  bin_mean_viirs_gt0(ii) + bin_mean_llc_gt0(ii);
end

%% Now bin the variables by cutout/HEALPix cell range for head, tail. Do this cells with min_num values.

step_size = 10;
iBin = 0;

nn_threshold = 100;
upper = 0;
while upper<1600
    iBin = iBin + 1;
    
    lower = upper;
    for j=lower:1600
        nn = find(number_head_gt0>=lower & number_head_gt0<j & number_tail_gt0>=lower & number_tail_gt0<j);
        if length(nn) > nn_threshold
            upper = j;
            break
        end
        upper = j;
    end
    
    midpoint_head_gt0(iBin) = (lower + upper) / 2;
    
    bin_counts_head_gt0(iBin) = length(nn);
    
    if bin_counts_head_gt0(iBin) > 0
        bin_mean_HT_diff_gt0(iBin) = mean(abs(head_tail_diff_gt0(nn)));
        bin_sigma_HT_diff_gt0(iBin) = std(head_tail_diff_gt0(nn));
        bin_median_HT_diff_gt0(iBin) = median(abs(head_tail_diff_gt0(nn)));
        bin_counts_HT_diff_gt0(iBin) = length(nn);
        
        if mean_and_sigma
            bin_mean_head_gt0(iBin) = mean(median_head_gt0(nn));
            bin_sigma_head_gt0(iBin) = std(median_head_gt0(nn));
        end
        bin_median_head_gt0(iBin) = median(median_head_gt0(nn));
        bin_counts_head_gt0(iBin) = length(nn);
        
        if mean_and_sigma
            bin_mean_tail_gt0(iBin) = mean(median_tail_gt0(nn));
            bin_sigma_tail_gt0(iBin) = std(median_tail_gt0(nn));
        end
        bin_median_tail_gt0(iBin) = median(median_tail_gt0(nn));
        bin_counts_tail_gt0(iBin) = length(nn);
    else
        bin_mean_HT_diff_gt0(iBin) = nan;
        bin_sigma_HT_diff_gt0(iBin) = nan;
        bin_median_HT_diff_gt0(iBin) = nan;
        bin_counts_HT_diff_gt0(iBin) = 0;
        
        if mean_and_sigma
            bin_mean_head_gt0(iBin) = nan;
            bin_sigma_head_gt0(iBin) = nan;
        end
        bin_median_head_gt0(iBin) = nan;
        bin_counts_head_gt0(iBin) = 0;
        
        if mean_and_sigma
            bin_mean_tail_gt0(iBin) = nan;
            bin_sigma_tail_gt0(iBin) = nan;
        end
        bin_median_tail_gt0(iBin) = nan;
        bin_counts_tail_gt0(iBin) = 0;
    end
    
    % Get bin number for the last bin with a midpoint less than 100. This
    % is used to estimate sigma.
    
    if midpoint_head_gt0(iBin) < 100
        iBin_100 = iBin;
    end
end

%% Next bin the variables by cutout/HEALPix cell range for VIIRS, LLC

mm_threshold = 10;
upper = 0;
iBin = 0;
while upper<1600
    iBin = iBin + 1;
    
    lower = upper;
    for j=lower:1600
        mm = find(number_viirs_gt0>=lower & number_viirs_gt0<j);
        if length(mm) > mm_threshold
            upper = j;
            break
        end
        upper = j;
    end
    
    midpoint_viirs_gt0(iBin) = (lower + upper) / 2;
    
    bin_counts_viirs_gt0(iBin) = length(mm);
    
    if bin_counts_viirs_gt0(iBin) > 0
        bin_mean_vl_diff_gt0(iBin) = mean(viirs_llc_diff_gt0(mm));
        bin_sigma_vl_diff_gt0(iBin) = std(viirs_llc_diff_gt0(mm));
        bin_median_vl_diff_gt0(iBin) = median(viirs_llc_diff_gt0(mm));
        bin_counts_vl_diff_gt0(iBin) = length(mm);
        
        bin_mean_viirs_gt0(iBin) = mean(median_viirs_gt0(mm));
        bin_sigma_viirs_gt0(iBin) = std(median_viirs_gt0(mm));
        bin_median_viirs_gt0(iBin) = median(median_viirs_gt0(mm));
        bin_counts_viirs_gt0(iBin) = length(mm);
        
        bin_mean_llc_gt0(iBin) = mean(median_llc_gt0(mm));
        bin_sigma_llc_gt0(iBin) = std(median_llc_gt0(mm));
        bin_median_llc_gt0(iBin) = median(median_llc_gt0(mm));
        bin_counts_llc_gt0(iBin) = length(mm);
    else
        bin_mean_vl_diff_gt0(iBin) = nan;
        bin_sigma_vl_diff_gt0(iBin) = nan;
        bin_median_vl_diff_gt0(iBin) = nan;
        bin_counts_vl_diff_gt0(iBin) = 0;
        
        bin_mean_viirs_gt0(iBin) = nan;
        bin_sigma_viirs_gt0(iBin) = nan;
        bin_median_viirs_gt0(iBin) = nan;
        bin_counts_viirs_gt0(iBin) = 0;
        
        bin_mean_llc_gt0(iBin) = nan;
        bin_sigma_llc_gt0(iBin) = nan;
        bin_median_llc_gt0(iBin) = nan;
        bin_counts_llc_gt0(iBin) = 0;
    end
end

%% Get thresholds to use in plotting extreme VIIRS-LLC values. Default to masked_cc for color of masked area.

switch which_thresholds_to_use
    case 'diff_at_100'
        nSigma = 3;
        ll_thresholds_gt0 = nSigma * [-bin_sigma_HT_diff_gt0(iBin_100) bin_sigma_HT_diff_gt0(iBin_100) masked_cc/nSigma];
        
    case 'best_fit_to_head_tail_diff'
        nSigma = 2;
        ll_thresholds_gt0 = nSigma * [-100 100 masked_cc/nSigma];
        
    case 'sigma_head_tail_diff'
        nSigma = 2;
        ll_thresholds_gt0 = nSigma * [-std(head_tail_diff_gt0,'omitnan') std(head_tail_diff_gt0,'omitnan') masked_cc/nSigma];
        
    case 'llc_gt_50_percent_viirs'  % Will get this when plotting Fig. 1.
        nSigma = nan;
        
        if Normalize
            [Values_vl, Edges] = histcounts(viirs_llc_diff_gt0, [-1000:25:1000], normalization='probability');
            [Values_ht, Edges] = histcounts(head_tail_diff_gt0, [-1000:25:1000], normalization='probability');
        else
            [Values_vl, Edges] = histcounts(viirs_llc_diff_gt0, [-1000:25:1000]);
            [Values_ht, Edges] = histcounts(head_tail_diff_gt0, [-1000:25:1000]);
        end
        
        nn = find( Values_vl(1:41) >= (2 * Values_ht(1:41)));
        ll_thresholds_gt0(1) = Edges(nn(end)+1);
        
        nn = find( Values_vl(41:end) >= (2 * Values_ht(41:end)));
        ll_thresholds_gt0(2) = Edges(nn(1)+40);
        
        ll_thresholds_gt0(3) = masked_cc;
end

%% Fix LLC

% First fix llc and plot. Start by getting offset to apply to LLC LL
% values. Bin the data first.

istep = 100;

use_all = 0; % Not clear that use_all will work since there will be different numbers of viirs and llc elements; use_all=1 with caution.

if use_all
    lat_viirs_to_use = lat_viirs;
    lon_viirs_to_use = lon_viirs;
    median_viirs_to_use =  median_viirs;
    
    lat_llc_to_use = lat_llc;
    lon_llc_to_use = lon_llc;
    median_llc_to_use =  median_llc;
else
    lat_viirs_to_use = lat_viirs_gt0;
    lon_viirs_to_use = lon_viirs_gt0;
    median_viirs_to_use =  median_viirs_gt0;
    
    lat_llc_to_use = lat_llc_gt0;
    lon_llc_to_use = lon_llc_gt0;
    median_llc_to_use =  median_llc_gt0;
end

ii = 0;
for icount=1:istep:length(median_viirs_to_use)
    ii = ii + 1;
    temp_viirs(ii) = mean(median_viirs_to_use(icount:min(length(median_viirs_to_use),icount+istep)), 'omitnan');
    temp_llc(ii) = mean(median_llc_to_use(icount:min(length(median_llc_to_use),icount+istep)), 'omitnan');
    
    new_llc(icount:min(length(median_viirs_to_use),icount+istep)) = median_llc_to_use(icount:min(length(median_viirs_to_use),icount+istep)) -  temp_llc(ii) + temp_viirs(ii);
    new_viirs(icount:min(length(median_viirs_to_use),icount+istep)) = median_viirs_to_use(icount:min(length(median_viirs_to_use),icount+istep)) -  temp_viirs(ii) + temp_llc(ii);
end

%% FIGURE 1: Histogram VIIRS-LLC LL values and tail-head values

keyboard

if plot_1
    
    % Setup the figure before plotting.
    
    FigTitle = ['pcc_1x1_LL_histograms_for_gt_' num2str(min_num) '_cutouts_per_cell'];
    %     h = undock_figure( 1, FigTitle, [100 1 subplot_details_2x2{1,1}(1) subplot_details_2x2{1,1}(2)]);
    h = undock_figure( 1, FigTitle, [100 1 1200 650]);
    
    if Normalize
        h_vl = histogram(viirs_llc_diff_gt0, [-1000:25:1000], normalization='probability');
    else
        h_vl = histogram(viirs_llc_diff_gt0, [-1000:25:1000]);
    end
    
    hold on
    
    if Normalize
        h_ht = histogram(head_tail_diff_gt0, [-1000:25:1000], normalization='probability');
        
        plot( ll_thresholds_gt0(1)*[1 1], [0 0.15], 'k', linewidth=2)
        plot( ll_thresholds_gt0(2)*[1 1], [0 0.15], 'k', linewidth=2)
        
        ylabel('Probability')
    else
        h_ht = histogram(head_tail_diff_gt0, [-1000:25:1000]);
        
        plot( ll_thresholds_gt0(1)*[1 1], [0 2500], 'k', linewidth=2)
        plot( ll_thresholds_gt0(2)*[1 1], [0 2500], 'k', linewidth=2)
        
        ylabel('Number')
    end
    
    set(gca,'fontsize',24)
    grid on
    xlabel('\boldmath$\Delta\widetilde{LL}$', interpreter='latex')
    title('\bf{\boldmath{Histograms of $\widetilde{LL}$ Differences}}' ,fontsize=32, interpreter='latex')
    
    switch which_thresholds_to_use
        case 'diff_at_100'
            legend('VIIRS-LLC', 'VIIRS: (2012-2015) - (2018-2020)', ['\pm ' num2str(nSigma) '\times\sigma of Head-Tail Difference with 100 Cutouts/Cell'])
            
        case 'best_fit_to_head_tail_diff'
            legend('VIIRS-LLC', 'VIIRS: (2012-2015) - (2018-2020)', ['\pm \sigma of Best Fit to Head-Tail Histogram'])
            
        case 'sigma_head_tail_diff'
            legend('VIIRS_{2012-2020}-LLC', 'VIIRS_{(2012-2015)} - VIIRS_{(2018-2020)}', ['\pm ' num2str(nSigma) '\sigma_{(2012-2015) - (2018-2020)}'])
            
        case 'llc_gt_50_percent_viirs'
            legend('VIIRS_{2012-2020}-LLC', 'VIIRS_{2012-2015} - VIIRS_{2018-2020}', ['\Delta{LL}=' num2str(ll_thresholds_gt0(1),3) ', ' num2str(ll_thresholds_gt0(2),3)])
    end
    
    filename_out = [FigPaperDir FigTitle];
    print( filename_out, '-dpng')
    
    % And dock the figure.
    
    set(h, WindowStyle='docked')
    
    fprintf('Printing figure 1: %s\n', filename_out)
    
    nn = find(viirs_llc_diff_gt0 < ll_thresholds_gt0(1));
    num_viirs_below = length(nn) * 100 / length(viirs_llc_diff_gt0);
    
    nn = find(viirs_llc_diff_gt0 > ll_thresholds_gt0(2));
    num_viirs_above = length(nn) * 100 / length(viirs_llc_diff_gt0);
    
    fprintf('\nPercent of VIIRS-LLC population with values less than %4.2f and values more than %4.2f:\n %6.1f%%, %6.1f%% \n\n', ...
        ll_thresholds_gt0(1), ll_thresholds_gt0(2), num_viirs_below, num_viirs_above)
    
    nn = find(head_tail_diff_gt0 < ll_thresholds_gt0(1));
    num_head_tail_below = length(nn) * 100 / length(head_tail_diff_gt0);
    
    nn = find(head_tail_diff_gt0 > ll_thresholds_gt0(2));
    num_head_tail_above = length(nn) * 100 / length(head_tail_diff_gt0);
    
    fprintf('Percent of VIIRS-LLC population with values less than %4.2f and values more than %4.2f:\n %6.1f%%, %6.1f%% \n', ...
        ll_thresholds_gt0(1), ll_thresholds_gt0(2), num_head_tail_below, num_head_tail_above)
end


%% 11: head and tail plots.

FigNo = 11;

FigTitle = ['pcc_1x1_log_10_head_and_tail_counts_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x1{1,1}(1) subplot_details_1x1{1,1}(2)]);

%     color_label_mollweide = 'Log_{10} of \# Cutouts/HEALPix Cell';
color_label_mollweide = [];

Title_1 = '$Log_{10}$ \# VIIRS$_{2012-2015}$ Cutouts/HEALPix Cell';

subplot_temp = make_quad_plot( FigNo, subplot_details_1x1, ...
    lat_head_gt0, lon_head_gt0, log10(number_head_gt0), log10(number_tail_gt0), [], lat_range, [], ...
    dot_size, colormap_to_use_num, caxis_mollweide_0_3, color_label_mollweide, [], [], Title_1, '', FigPaperDir, FigTitle);

set(h, WindowStyle='docked')

%% 12: VIIRS and LLC plots.

FigNo = 12;

FigTitle = ['pcc_1x1_log_10_counts_VIIRS_and_LLC_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x1{1,1}(1) subplot_details_1x1{1,1}(2)]);

%     color_label_mollweide = 'Log_{10} of # Cutouts/HEALPix Cell';
color_label_mollweide = [];

Title_1 = '$Log_{10}$ \# Cutouts/HEALPix Cell for VIIRS, LLC Matchups';

subplot_temp = make_quad_plot( FigNo, subplot_details_1x1, lat_viirs_gt0, lon_viirs_gt0, log10(number_viirs_gt0), [], [], lat_range, [], ...
    dot_size, colormap_to_use_num, caxis_mollweide_0_3, color_label_mollweide, [], ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);

set(h, WindowStyle='docked')

%%   21: VIIRS (top) and LLC (bottom).

FigNo = 21;

FigTitle = ['pcc_2x2_VIIRS_and_LLC_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [1 1 subplot_details_2x2{1,1}(1) subplot_details_2x2{1,1}(2)]);

%     color_label_mollweide = '$\widetilde{LL}$';
color_label_mollweide = [];

Title_1 = 'a) $\widetilde{LL}_{VIIRS}$';
Title_2 = 'b) $\widetilde{LL}_{LLC}$';

subplot_details_2x2 = make_quad_plot( FigNo, subplot_details_2x2, lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, median_llc_gt0, [], lat_range, [], ...
    dot_size, colormap_to_use_ll, caxis_mollweide_550_1200, color_label_mollweide, xlim_zonal_mean_500_1000, ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);

set(h, WindowStyle='docked')

%% VIIRS-LLC

FigNo = 27;

FigTitle = ['pcc_1x2_VIIRS_minus_LLC_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x2{1,1}(1) subplot_details_1x2{1,1}(2)]);

%     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
color_label_mollweide = [];

Title_1 = 'a) $\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';

subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, [], [0 0], lat_range, [], ...
    dot_size, colormap_to_use_ll, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);

set(h, WindowStyle='docked')    %% Now for the related Mollweide plots

%% 28: VIIRS-LLC masked

FigNo = 28;

FigTitle = ['pcc_1x1_' num2str(ll_thresholds_gt0(1),3) '_lt_VIIRS_minus_LLC_gt_' num2str(ll_thresholds_gt0(2),3) '_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x1{1,1}(1) subplot_details_1x1{1,1}(2)]);

%     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
color_label_mollweide = [];

Title_1 = ['$\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $>$ ' num2str(ll_thresholds_gt0(2),3)];

%     subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, [], ll_thresholds_gt0, lat_range, [], ...
%      dot_size, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);

HEALPix_plot_Mollweid( FigNo, subplot_details_1x1{1,1}, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, ...
    dot_size, colormap_to_use_ll, caxis_mollweide_600_600, Title_1, FigPaperDir, FigTitle, ll_thresholds_gt0, color_label_mollweide, color_label_size_small, ...
    lat_range, [], {xxeqsq xxgssq xxacsq}, {yyeqsq yygssq yyacsq}, {'y' 'c' 'g'})

set(h, WindowStyle='docked')

%% 30: (2012-2015) - (2018-2020)

FigNo = 30;

FigTitle = ['pcc_1x2_head_tail_diff_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x2{1,1}(1) subplot_details_1x2{1,1}(2)]);

%     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';
color_label_mollweide = [];

Title_1 = 'a) $\widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';

subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_head_gt0, lon_head_gt0, head_tail_diff_gt0, [], [0 0], lat_range, [], ...
    dot_size, colormap_to_use_ll, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);

set(h, WindowStyle='docked')

%% VIIRS (top) and corrected LLC (bottom).

FigNo = 44;

FigTitle = ['pcc_1x2_VIIRS_and_corrected_LLC_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_2x2{1,1}(1) subplot_details_2x2{1,1}(2)]);

%     color_label_mollweide = '$\widetilde{LL}$';
color_label_mollweide = [];

Title_1 = 'a) $\widetilde{LL}_{VIIRS}$';
Title_2 = 'b) Corrected $\widetilde{LL}_{LLC}$';

subplot_temp = make_quad_plot( FigNo, subplot_details_2x2, lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, new_llc, [], lat_range, [], ...
    dot_size, colormap_to_use_ll, caxis_mollweide_550_1200, color_label_mollweide, xlim_zonal_mean_500_1000, ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);

set(h, WindowStyle='docked')

    %% Now zoom into the southern hemisphere for the ACC plots
    
    FigNo = 47;
    
    figfact = 0.75;
    
% % %     load('~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/feature1');
    load('HEALPix/feature1');
    
    FigTitle = ['pcc_3x1_zoomed_ACC_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 1557*figfact 1283]);
    
    color_label_mollweide = [];
    
    lat_zoom_range = [-75 -35];
    lon_zoom_range = [45 220];
    
    Title_1 = 'a) $\widetilde{LL}_{VIIRS}$';
    Title_2 = 'b) $\widetilde{LL}_{LLC}$';
    Title_3 = 'c) $\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    
    HEALPix_plot_Mollweid( FigNo, [1557*figfact 1283 0.0500 0.660 0.9000 0.31], lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, ...
        100, colormap_to_use_ll, [-800 800], Title_1, ...
        FigPaperDir, [], [0 0], [], color_label_size_large, lat_zoom_range, lon_zoom_range, {xxacsub1 xxacsub2}, {yyacsub1 yyacsub2}, {'w' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, [1557*figfact 1283 0.0500 0.3300 0.9000 0.31], lat_viirs_gt0, lon_viirs_gt0, median_llc_gt0, ...
        100, colormap_to_use_ll, [-800 800], Title_2, ...
        FigPaperDir, [], [0 0], [], color_label_size_large, lat_zoom_range, lon_zoom_range, {xxacsub1 xxacsub2}, {yyacsub1 yyacsub2}, {'w' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, [1557*figfact 1283 0.0500 0.0100 0.9000 0.31], lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, ...
        100, colormap_to_use_ll, [-600 600], Title_3, ...
        FigPaperDir, FigTitle, ll_thresholds_gt0, [], color_label_size_large, lat_zoom_range, lon_zoom_range, {xxacsub1 xxacsub2}, {yyacsub1 yyacsub2}, {'w' 'k'})
    
    set(h, WindowStyle='docked')

    %% Now zoom into the North Atlantic and Pacific head and tail comparison.

FigNo = 48;

FigTitle = ['pcc_2x1_zoomed_head_and_tail_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_2x1{1,1}(1) subplot_details_2x1{1,1}(2)]);

%     color_label_mollweide = {'$\widetilde{LL}$' '\Delta LL'};
color_label_mollweide = {'' ''};

Title_1 = 'a) $VIIRS_{2012-2015}$';
Title_2 = ['b) $\widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $>$ ' num2str(ll_thresholds_gt0(2),3)];

%     subplot_temp = make_quad_plot( FigNo, subplot_details_2x1, lat_head_gt0, lon_head_gt0, median_head_gt0, head_tail_diff_gt0, ll_thresholds_gt0, [-20 70], [-80 180], ...
subplot_temp = make_quad_plot( FigNo, subplot_details_2x1, lat_head_gt0, lon_head_gt0, median_head_gt0, head_tail_diff_gt0, ll_thresholds_gt0, [-20 60], [-80 180], ...
    40, colormap_to_use_ll, caxis_mollweide_550_1200_600_600, color_label_mollweide, [], ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);

set(h, WindowStyle='docked')

%% Galleries for Equatorial region

FigNo = 83;
FigTitle = ['pcc_4x3x3_eq_viirs_llc_gallery'];

use_eq_indices = 1;

clear eq_indices_new

if use_eq_indices
%     eq_indices(1,1,:) = [1060 1678 6723 618 1965 477 3426 93 6729];
%     eq_indices(2,1,:) = [1552 699 2366 3555 764 7801 2591 2272 443];
    
    eq_indices(1,1,:) = [3107.00       2530.00        872.00       6965.00       1057.00       2669.00       6825.00       3694.00       4553.00];
    eq_indices(2,1,:) = [4455.00       5296.00        320.00       3746.00       7712.00       6324.00       2888.00       6455.00       2359.00];
       
    eq_indices_1_1 = eq_indices(1,1,:);
    eq_indices_2_1 = eq_indices(2,1,:);
    
    for ii=1:9
        if get_reordered_llc_HEALPix
            eq_indices(1,2,ii) = eq_indices(1,1,ii);
            eq_indices(2,2,ii) = eq_indices(2,1,ii);
        else
        [error_return_1_1, eq_indices(1,2,ii)] = find_viirs_llc_matches( eq_indices(1,1,ii), eq_viirs_metadata, eq_llc_metadata);
        [error_return_2_1, eq_indices(2,2,ii)] = find_viirs_llc_matches( eq_indices(2,1,ii), eq_viirs_metadata, eq_llc_metadata);
        end
        
        eq_indices_1_2(ii) = eq_indices(1,2,ii);
        eq_indices_2_2(ii) = eq_indices(2,2,ii);
    end
else
    eq_indices_1_1 = [];
    eq_indices_1_2 = [];
    eq_indices_2_1 = [];
    eq_indices_2_2 = [];
end

eq_lon_range = [245 255] - 360;
if get_reordered_llc_HEALPix
    [cutouts_master{1}, metadata_master{1}, cutouts_master{2}, metadata_master{2}, eq_indices_new(1,1,:), eq_means_to_use(1,1,:), eq_means_to_use(1,2,:)] = ...
        extract_random_cutout_pairs( [0 2], eq_lon_range, 10, 9, eq_viirs_cutouts, eq_viirs_metadata, eq_llc_cutouts, eq_llc_metadata, squeeze(eq_indices_1_1(:))');
    
    [cutouts_master{3}, metadata_master{3}, cutouts_master{4}, metadata_master{4}, eq_indices_new(2,1,:), eq_means_to_use(2,1,:), eq_means_to_use(2,2,:)] = ...
        extract_random_cutout_pairs( [-2 0], eq_lon_range, 10, 9, eq_viirs_cutouts, eq_viirs_metadata, eq_llc_cutouts, eq_llc_metadata, squeeze(eq_indices_2_1(:))');
else
    [cutouts_master{1}, metadata_master{1}, eq_indices_viirs_above, eq_means_to_use(1,1,:)] = extract_random_cutouts([0 2], eq_lon_range, 9, eq_viirs_cutouts, eq_viirs_metadata, squeeze(eq_indices_1_1(:))');
    [cutouts_master{3}, metadata_master{3}, eq_indices_viirs_below, eq_means_to_use(2,1,:)] = extract_random_cutouts([-2 0], eq_lon_range, 9, eq_viirs_cutouts, eq_viirs_metadata, squeeze(eq_indices_1_2(:))');
    
    [cutouts_master{2}, metadata_master{2}, eq_indices_llc_above, eq_means_to_use(1,2,:)] = extract_random_cutouts([0 2], eq_lon_range, 9, eq_llc_cutouts, eq_llc_metadata, squeeze(eq_indices_2_1(:))');
    [cutouts_master{4} metadata_master{4}, eq_indices_llc_below, eq_means_to_use(2,2,:)] = extract_random_cutouts([-2 0], eq_lon_range, 9, eq_llc_cutouts, eq_llc_metadata, squeeze(eq_indices_2_2(:))');
end

make_4x3x3_gallery(FigNo, FigPaperDir, FigTitle, cutouts_master, metadata_master, [-1.5 1.5], 11, {'VIIRS' 'LLC' 'Above Equator' 'Below Equator'}, eq_means_to_use)

%% Galleries for ACC

FigNo = 84;
FigTitle = ['pcc_4x3x3_ACC_viirs_llc_gallery'];

use_acc_indices = 1;

clear acc_indices_new

if use_acc_indices
    acc_indices(1,1,:) = [246   604   386   390    12   519   320   488   256];
    acc_indices(2,1,:) = [417   314   629   324    38   409   276   162   552];
    
    acc_indices_1_1 = acc_indices(1,1,:);
    acc_indices_2_1 = acc_indices(2,1,:);
    
    for ii=1:9
        if get_reordered_llc_HEALPix
            acc_indices(1,2,ii) = acc_indices(1,1,ii);
            acc_indices(2,2,ii) = acc_indices(2,1,ii);
        else
            [error_return_1_1, acc_indices(1,2,ii), dist_sep(ii), time_sep(ii)] = find_viirs_llc_matches( acc_indices(1,1,ii), acc_viirs_metadata, acc_llc_metadata);
            [error_return_2_1, acc_indices(2,2,ii), dist_sep(ii), time_sep(ii)] = find_viirs_llc_matches( acc_indices(2,1,ii), acc_viirs_metadata, acc_llc_metadata);
        end
        
        acc_indices_1_2(ii) = acc_indices(1,2,ii);
        acc_indices_2_2(ii) = acc_indices(2,2,ii);
    end
else
    acc_indices_1_1 = [];
    acc_indices_1_2 = [];
    acc_indices_2_1 = [];
    acc_indices_2_2 = [];
end

if get_reordered_llc_HEALPix
    [cutouts_master{1}, metadata_master{1}, cutouts_master{2}, metadata_master{2}, acc_indices_new(1,1,:), acc_means_to_use(1,1,:), acc_means_to_use(1,2,:)] = ...
        extract_random_cutout_pairs( [-52 -49], [115 117], 10, 9, acc_viirs_cutouts, acc_viirs_metadata, acc_llc_cutouts, acc_llc_metadata, squeeze(acc_indices_1_1(:))');
    
    [cutouts_master{3}, metadata_master{3}, cutouts_master{4}, metadata_master{4}, acc_indices_new(2,1,:), acc_means_to_use(2,1,:), acc_means_to_use(2,2,:)] = ...
        extract_random_cutout_pairs( [-55 -53], [119 124], 10, 9, acc_viirs_cutouts, acc_viirs_metadata, acc_llc_cutouts, acc_llc_metadata, squeeze(acc_indices_2_1(:))');
else
    cutouts_master = {unique_agreeing_acc_viirs_cutouts, unique_agreeing_acc_llc_cutouts, unique_anomalous_acc_viirs_cutouts, unique_anomalous_acc_llc_cutouts};
    metadata_master = {unique_agreeing_acc_viirs_metadata, unique_agreeing_acc_llc_metadata, unique_anomalous_acc_viirs_metadata, unique_anomalous_acc_llc_metadata};
end

make_4x3x3_gallery(FigNo, FigPaperDir, FigTitle, cutouts_master, metadata_master, [-2.5 2.5], 11, {'VIIRS' 'LLC' 'Agrees (black rectangle)' 'Anomalous (white rectangle)'}, acc_means_to_use)

%% Galleries for Gulf Stream region

FigNo = 85;
FigTitle = ['pcc_4x3x3_gs_viirs_llc_gallery'];

use_gs_indices = 1;

clear gs_indices_new

if use_gs_indices
    %         gs_indices(1,1,:) = [7321        5924        7188        3324        2270        7095        2593        1425        2207];
    %         gs_indices(2,1,:) = [5323        4665        5510        7294        7043        4632        7042         662         215];
    gs_indices(1,1,:) = [1884.00       5711.00       5570.00       3314.00       3311.00       1631.00       4177.00       5125.00       2425.00];
    gs_indices(2,1,:) = [2836.00       2857.00       2126.00       3730.00       5239.00       6556.00       4245.00       1352.00       3676.00];
    
    gs_indices_1_1 = gs_indices(1,1,:);
    gs_indices_2_1 = gs_indices(2,1,:);
    
    for ii=1:9
        if get_reordered_llc_HEALPix
            gs_indices(1,2,ii) = gs_indices(1,1,ii);
            gs_indices(2,2,ii) = gs_indices(2,1,ii);
        else
            [error_return_1_1, gs_indices(1,2,ii), dist_sep(ii), time_sep(ii)] = find_viirs_llc_matches( gs_indices(1,1,ii), gs_viirs_metadata, gs_llc_metadata);
            [error_return_2_1, gs_indices(2,2,ii), dist_sep(ii), time_sep(ii)] = find_viirs_llc_matches( gs_indices(2,1,ii), gs_viirs_metadata, gs_llc_metadata);
        end
        
        gs_indices_1_2(ii) = gs_indices(1,2,ii);
        gs_indices_2_2(ii) = gs_indices(2,2,ii);
    end
else
    gs_indices_1_1 = [];
    gs_indices_1_2 = [];
    gs_indices_2_1 = [];
    gs_indices_2_2 = [];
end

if get_reordered_llc_HEALPix
    [cutouts_master{1}, metadata_master{1}, cutouts_master{2}, metadata_master{2}, gs_indices_new(1,1,:), gs_means_to_use(1,1,:), gs_means_to_use(1,2,:)] = ...
    extract_random_cutout_pairs( [40 42], [-60 -50], 10, 9, gs_viirs_cutouts, gs_viirs_metadata, gs_llc_cutouts, gs_llc_metadata, squeeze(gs_indices_1_1(:))');

    [cutouts_master{3}, metadata_master{3}, cutouts_master{4}, metadata_master{4}, gs_indices_new(2,1,:), gs_means_to_use(2,1,:), gs_means_to_use(2,2,:)] = ...
    extract_random_cutout_pairs( [34 36], [-70 -60], 10, 9, gs_viirs_cutouts, gs_viirs_metadata, gs_llc_cutouts, gs_llc_metadata, squeeze(gs_indices_2_1(:))');
else
[cutouts_master{1}, metadata_master{1}, gs_indices_new(1,1,:), gs_means_to_use(1,1,:)] = extract_random_cutouts( [40 42], [-60 -50], 10, 9, gs_viirs_cutouts, gs_viirs_metadata, squeeze(gs_indices_1_1(:))');
[cutouts_master{3}, metadata_master{3}, gs_indices_new(2,1,:), gs_means_to_use(2,1,:)] = extract_random_cutouts( [34 36], [-70 -60], 10, 9, gs_viirs_cutouts, gs_viirs_metadata, squeeze(gs_indices_2_1(:))');

[cutouts_master{2}, metadata_master{2}, gs_indices_new(1,2,:), gs_means_to_use(1,2,:)] = extract_random_cutouts( [40 42], [-60 -50], 10, 9, gs_llc_cutouts, gs_llc_metadata, squeeze(gs_indices_1_2(:))');
[cutouts_master{4} metadata_master{4}, gs_indices_new(2,2,:), gs_means_to_use(2,2,:)] = extract_random_cutouts( [34 36], [-70 -60], 10, 9, gs_llc_cutouts, gs_llc_metadata, squeeze(gs_indices_2_2(:))');
end

make_4x3x3_gallery(FigNo, FigPaperDir, FigTitle, cutouts_master, metadata_master, [-4 4], 11, {'VIIRS' 'LLC' 'In the Gulf Stream' 'South of the Gulf Stream'}, gs_means_to_use)

%****************************************************************************************
%****************************************************************************************
%****************************************************************************************

%% Figures 21-31: MOLLWEIDE PLOTS OF MEAN(MEDIAN(LL)) for data with at least min_num Cutouts/Cell
%   21: VIIRS (top) and LLC (bottom).
%   22: VIIRS-LLC (top) and VIIRS-LLC masked (bottom).
%   23: 2012-2015 (top) and 2018-2020 (bottom).
%   14: (2012-2015) - (2018-2020) (top) and (2012-2015) - (2018-2020) masked (bottom).
%   25: VIIRS
%   26: LLC
%   27: VIIRS-LLC
%   28: VIIRS-LLC masked
%   29: 2012-2015
%   30: (2012-2015) - (2018-2020)
%   31: (2012-2015) - (2018-2020) masked
%
%   32: viirs-modis with at least 5 cutouts/HEALPix cell
%   33: masked viirs-modis with at least 5 cutouts/HEALPix cell

if plot_21_31
    %%
    
    FigNo = 22;
    
    FigTitle = ['pcc_2x2_VIIRS_minus_LLC_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [1 1 subplot_details_2x2{1,1}(1) subplot_details_2x2{1,1}(2)]);
    
    %     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    %
    %     Title_1 = 'a) $\Delta_{LL}$';
    %     Title_2 = ['b) $\Delta_{LL} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $\Delta_{LL} >$ ' num2str(ll_thresholds_gt0(2),3)];
    %     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    color_label_mollweide = [];
    
    Title_1 = 'a) $\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    Title_2 = ['b) $\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $>$ ' num2str(ll_thresholds_gt0(2),3)];
    
    subplot_temp = make_quad_plot( FigNo, subplot_details_2x2, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, viirs_llc_diff_gt0, ll_thresholds_gt0, lat_range, [], ...
        dot_size, colormap_to_use_ll, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);
    
    set(h, WindowStyle='docked')
    
    %%
    
    if mean_and_sigma
        FigNo = 23;
        
        FigTitle = ['pcc_2x2_head_and_tail_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
        h = undock_figure( FigNo, FigTitle, [1 1 subplot_details_2x2{1,1}(1) subplot_details_2x2{1,1}(2)]);
        
        %     color_label_mollweide = '$\widetilde{LL}$';
        color_label_mollweide = [];
        
        Title_1 = 'a) $\widetilde{LL}_{2012-2015}$';
        Title_2 = 'b) $\widetilde{LL}_{2018-2020}$';
        
        subplot_temp = make_quad_plot( FigNo, subplot_details_2x2, lat_head_gt0, lon_head_gt0, mean_head_gt0, mean_tail_gt0, [0 0], lat_range, [], ...
            dot_size, colormap_to_use_ll, caxis_mollweide_550_1200, color_label_mollweide, xlim_zonal_mean_500_1000, ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);
        
        set(h, WindowStyle='docked')
    end
    
    %%
    
    FigNo = 24;
    
    FigTitle = ['pcc_2x2_head_minus_tail_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [1 1 subplot_details_2x2{1,1}(1) subplot_details_2x2{1,1}(2)]);
    
    %     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';
    color_label_mollweide = [];
    
    Title_1 = 'a) $\widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';
    Title_2 = ['b) $ \widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $>$ ' num2str(ll_thresholds_gt0(2),3)];
    
    subplot_temp = make_quad_plot( FigNo, subplot_details_2x2, lat_head_gt0, lon_head_gt0, head_tail_diff_gt0, head_tail_diff_gt0, ll_thresholds_gt0, lat_range, [], ...
        dot_size, colormap_to_use_ll, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);
    
    set(h, WindowStyle='docked')
    
    %% Now plot each one separately, at least for the important ones.
    
    FigNo = 25;
    
    FigTitle = ['pcc_1x2_VIIRS_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x2{1,1}(1) subplot_details_1x2{1,1}(2)]);
    
    %     color_label_mollweide = '$\widetilde{LL}$';
    color_label_mollweide = [];
    
    Title_1 = '$\widetilde{LL}_{LLC}$';
    
    subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, [], [0 0], lat_range, [], ...
        dot_size, colormap_to_use_ll, caxis_mollweide_550_1200, color_label_mollweide, xlim_zonal_mean_500_1000, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
    
    set(h, WindowStyle='docked')
    %%
    
    FigNo = 26;
    
    FigTitle = ['pcc_1x2_LLC_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x2{1,1}(1) subplot_details_1x2{1,1}(2)]);
    
    %     color_label_mollweide = '$\widetilde{LL}$';
    color_label_mollweide = [];
    
    Title_1 = 'a) $\widetilde{LL}_{LLC}$';
    
    subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_viirs_gt0, lon_viirs_gt0, median_llc_gt0, [], [0 0], lat_range, [], ...
        dot_size, colormap_to_use_ll, caxis_mollweide_550_1200, color_label_mollweide, xlim_zonal_mean_500_1000, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
    
    set(h, WindowStyle='docked')
    
    %%
    
    if mean_and_sigma
        FigNo = 29;
        
        FigTitle = ['pcc_1x2_head_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
        h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x2{1,1}(1) subplot_details_1x2{1,1}(2)]);
        
        %     color_label_mollweide = '$\widetilde{LL}$';
        color_label_mollweide = [];
        
        Title_1 = '$\widetilde{LL}_{2012-2015}$';
        
        subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_head_gt0, lon_head_gt0, mean_head_gt0, [], [0 0], lat_range, [], ...
            dot_size, colormap_to_use_ll, caxis_mollweide_550_1200, color_label_mollweide, xlim_zonal_mean_500_1000, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
        
        set(h, WindowStyle='docked')
    end
    %%
    
    FigNo = 31;
    
    FigTitle = ['pcc_1x2_' num2str(ll_thresholds_gt0(1),3) '_head_minus_tail_gt_' num2str(ll_thresholds_gt0(2),3) '_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x2{1,1}(1) subplot_details_1x2{1,1}(2)]);
    
    %     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';
    color_label_mollweide = [];
    
    Title_1 = ['$\widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $>$ ' num2str(ll_thresholds_gt0(2),3)];
    
    subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_head_gt0, lon_head_gt0, head_tail_diff_gt0, [], ll_thresholds_gt0, lat_range, [], ...
        dot_size, colormap_to_use_ll, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
    
    set(h, WindowStyle='docked')
    
    %%
    
    FigNo = 32;
    
    FigTitle = ['pcc_1x1_VIIRS_minus_MODIS'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x1{1,1}(1) subplot_details_1x1{1,1}(2)]);
    
    %     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    color_label_mollweide = [];
    
    Title_1 = '$\widetilde{LL}_{VIIRS} - \widetilde{LL}_{MODIS}$';
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, [], ll_thresholds_gt0, lat_range, [], ...
    %      dot_size, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_1x1{1,1}, lat_viirs_modis_gt0, lon_viirs_modis_gt0, viirs_modis_diff_gt0, ...
        dot_size, colormap_to_use_ll, caxis_mollweide_600_600, Title_1, FigPaperDir, FigTitle, [], [], color_label_size_small, ...
        [], [], {xxeqsq xxgssq xxacsq}, {yyeqsq yygssq yyacsq}, {'m' 'y' 'g'})
    
    set(h, WindowStyle='docked')
    
    %%
    
    FigNo = 33;
    
    FigTitle = ['pcc_1x1_' num2str(ll_thresholds_gt0(1),3) '_lt_VIIRS_minus_modis_gt_' num2str(ll_thresholds_gt0(2),3) '_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x1{1,1}(1) subplot_details_1x1{1,1}(2)]);
    
    color_label_mollweide = [];
    
    Title_1 = ['$\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $>$ ' num2str(ll_thresholds_gt0(2),3)];
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_1x2, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, [], ll_thresholds_gt0, lat_range, [], ...
    %      dot_size, caxis_mollweide_600_600, color_label_mollweide, xlim_zonal_mean_450_450, ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_1x1{1,1}, lat_viirs_modis_gt0, lon_viirs_modis_gt0, viirs_modis_diff_gt0, ...
        dot_size, colormap_to_use_ll, caxis_mollweide_600_600, Title_1, FigPaperDir, FigTitle, ll_thresholds_gt0, [], color_label_size_small, ...
        [], [], {xxeqsq xxgssq xxacsq}, {yyeqsq yygssq yyacsq}, {'m' 'y' 'g'})
    
    set(h, WindowStyle='docked')
    
end

%% Figures 41-45: MOLLWEIDE PLOTS OF MEAN(MEDIAN(LL)) for more than min_num of Cutouts/Cell

%   41: LL for VIIRS and LLC plotted versus the HEALPix cell index + the
%       LLC corrected values.
%   42: VIIRS (top) and LLC (bottom).
%   43: Zoomed in to the South Atlantic and part of the Southern Ocean for VIIRS (top) and LLC (bottom).
%   44: VIIRS (top) and corrected LLC (bottom).
%   45: Zoomed in to the South Atlantic and part of the Southern Ocean for VIIRS (top) and corrected LLC (bottom).
%   46: Zoomed in to the western North Atlantic for VIIRS 2012-2015 (top) and VIIRS 2018-2020 (bottom).

if plot_41_45
    
    iFig = 41;
    figure(iFig)
    clf
    
    plot( median_viirs_to_use, '.k', markersize=10)
    hold on
    
    plot( median_llc_to_use, '.c');
    plot( new_llc, '.r');
    
    % Annotate
    
    set( gca, 'fontsize', 24)
    grid on
    
    xlabel('HEALPix Index for Cells with Values')
    ylabel('LL')
    ylim([-1000 1200])
    
    legend( 'VIIRS', 'LLC', 'Corrected LLC')
    title('Correcting LLC to Mean VIIRS', fontsize=30)
    
    filename_out = [FigPaperDir 'pcc_1x1_correcting_LLC_to_mean_VIIRS'];
    print( filename_out, '-dpng')
    fprintf('Printing figure %i: %s\n', iFig, filename_out)
    
    %% Now zoom into the southern hemisphere Atlantic + Indian Oceans.
    
    FigNo = 45;
    
% % %     load('~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/feature1');
    load('feature1');
    
    FigTitle = ['pcc_2x1_zoomed_VIIRS_and_corrected_LLC_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_patagonia{1,1}(1) subplot_details_patagonia{1,1}(2)]);
    
    %     ylim_zonal_mean = [-90 90];
    %     color_label_mollweide = '$\widetilde{LL}$';
    color_label_mollweide = [];
    
    lat_zoom_range = [-60 0];
    lon_zoom_range = [-90 90];
    
    Title_1 = 'a) $\widetilde{LL}_{VIIRS}$';
    Title_2 = 'b) Corrected $\widetilde{LL}_{LLC}$';
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_patagonia, lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, new_llc, [], lat_zoom_range, lon_zoom_range, ...
    %         60, caxis_mollweide_550_1200, color_label_mollweide, xlim_zonal_mean_500_1000, ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_patagonia{1,1}, lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, ...
        60, colormap_to_use_ll, [-200 800], Title_1, ...
        FigPaperDir, [], [0 0], color_label_mollweide, color_label_size_large, lat_zoom_range, lon_zoom_range, {xxcircle xxlocate}, {yycircle yylocate}, {'k' 'r'})
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_patagonia{2,1}, lat_viirs_gt0, lon_viirs_gt0, new_llc, ...
        60, colormap_to_use_ll, [-200 800], Title_2, ...
        FigPaperDir, FigTitle, [0 0], color_label_mollweide, color_label_size_large, lat_zoom_range, lon_zoom_range, {xxcircle xxlocate}, {yycircle yylocate}, {'k' 'r'})
    
    set(h, WindowStyle='docked')
    
    %% Now zoom into the southern hemisphere Atlantic + Indian Oceans and plot viirs, llc, bathy
    
    FigNo = 46;
    
    [Bathy, Bathy_Lon, Bathy_Lat, Contours] = Get_Bathy( 0, [-90 90 1], [-60 0 1]);
    bLon = Bathy_Lon(1:25:end);
    bLat = Bathy_Lat(1:25:end);
    bTopo = Bathy(1:25:end,1:25:end);
    [bLonm, bLatm] = meshgrid(bLon, bLat);
    
% % %     load('~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/feature1');
    load('feature1');
    
    FigTitle = ['pcc_2x1_zoomed_VIIRS_LLC_LL_and_bathy_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_patagonia{1,1}(1) subplot_details_patagonia{1,1}(2)]);
    %     figfact = 0.5;
    %     h = undock_figure( FigNo, FigTitle, [100 1 1557*figfact 1283]);
    
    %     ylim_zonal_mean = [-90 90];
    %     color_label_mollweide = '$\widetilde{LL}$';
    color_label_mollweide = [];
    
    lat_zoom_range = [-60 0];
    lon_zoom_range = [-90 90];
    %     lat_zoom_range = [-55 -35];
    %     lon_zoom_range = [0 60];
    
    %     HEALPix_plot_Mollweid( FigNo, [1557*figfact 1283 0.0500 0.690 0.9000 0.23], lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, ...
    %         dot_size, caxis_mollweide_550_1200, 'a) $\widetilde{LL}_{VIIRS}$', ...
    %         FigPaperDir, [], [0 0], color_label_mollweide, color_label_size_large, lat_zoom_range, lon_zoom_range, {xxviirs xcircle}, {yyviirs ycircle}, {'r' 'r'})
    %
    %     HEALPix_plot_Mollweid( FigNo, [1557*figfact 1283 0.0500 0.3700 0.9000 0.23], lat_viirs_gt0, lon_viirs_gt0, median_llc_gt0, ...
    %         dot_size, caxis_mollweide_550_1200, 'b) $\widetilde{LL}_{LLC}$', ...
    %         FigPaperDir, [], [0 0], color_label_mollweide, color_label_size_large, lat_zoom_range, lon_zoom_range, {xxllc xcircle}, {yyllc ycircle}, {'r' 'r'})
    %
    %     HEALPix_plot_Mollweid( FigNo, [1557*figfact 1283 0.0500 0.0500 0.9000 0.23], bLatm(:), bLonm(:), bTopo(:), ...
    %         dot_size, [-8000 0], 'b) Bathymetry', ...
    %         FigPaperDir, FigTitle, [0 0], 'Depth (m)', color_label_size_large, lat_zoom_range, lon_zoom_range, {xxviirs xxllc xcircle}, {yyviirs yyllc ycircle}, {'r' 'k' 'r'})
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_patagonia{1,1}, lat_viirs_gt0, lon_viirs_gt0, median_llc_gt0, ...
        60, colormap_to_use_ll, caxis_mollweide_550_1200, 'a) $\widetilde{LL}_{LLC}$', ...
        FigPaperDir, [], [0 0], color_label_mollweide, color_label_size_large, lat_zoom_range, lon_zoom_range, {xxcircle xxlocate}, {yycircle yylocate}, {'k' 'r'})
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_patagonia{2,1}, bLatm(:), bLonm(:), bTopo(:), ...
        60, colormap_to_use_bathy, [-8000 0], 'b) Bathymetry', ...
        FigPaperDir, FigTitle, [0 0], 'Depth (m)', color_label_size_large, lat_zoom_range, lon_zoom_range, {xxllc xxviirs xxcircle xxlocate}, {yyllc yyviirs yycircle yylocate}, {'m' ':k' 'k' 'r'})
    
    set(h, WindowStyle='docked')
    
    %% Finally, just the Gulf Stream region.
    
    FigNo = 49;
    
    FigTitle = ['pcc_1x1_zoomed_GS_head_minus_tail_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_1x1{1,1}(1) subplot_details_1x1{1,1}(2)]);
    
    %     color_label_mollweide = '$\widetilde{LL}$';
    color_label_mollweide = [];
    
    Title_1 = '$\widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_1x1, lat_head_gt0, lon_head_gt0, head_tail_diff_gt0, [], [], [20 70], [-80 0], ...
    %         40, colormap_to_use_ll, caxis_mollweide_400_400, color_label_mollweide, [], ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle, {GS(:,2) GS_north(:,2)}, {GS(:,1) GS_north(:,1)}, {'k' 'm'});
    %
    %     % Plot Gulf Stream lines and print figure again
    %
    %     plotm( GS(:,2), GS(:,1)+360, 'k', linewidth=2)
    %     plotm( GS_north(:,2), GS_north(:,1)+360, 'm', linewidth=2)
    %
    %
    %     filename_out = [FigPaperDir FigTitle];
    %     print( filename_out, '-dpng')
    %
    %     fprintf('Printing figure: %s\n', filename_out)
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_1x1{1,1}, lat_head_gt0, lon_head_gt0, head_tail_diff_gt0, ...
        60, colormap_to_use_ll, caxis_mollweide_400_400, Title_1, FigPaperDir, FigTitle, [], color_label_mollweide, color_label_size_small, ...
        [20 70], [-80 0], {GS(:,2) GS_north(:,2)}, {GS(:,1) GS_north(:,1)}, {'k' 'm'});
    
    set(h, WindowStyle='docked')
    
    %%
    
    FigNo = 100;
    
    FigTitle = ['pcc_2x1_zoomed_eq_atlantic_pacific_viirs_llc_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_2x1{1,1}(1) subplot_details_2x1{1,1}(2)]);
    
    %     color_label_mollweide = {'LL_{VIIRS}-LL_{LLC}' 'log_{10}(# Cutouts)' };
    color_label_mollweide = {'' ''};
    
    Title_1 = 'a) $\widetilde{LL}_{VIIRS}-\widetilde{LL}_{LLC}$';
    Title_2 = 'b) log$_{10}$(Cutouts/HEALPix Cell)';
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_2x1, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, log10(number_viirs_gt0), [], [-30 30], [-150 30], ...
    %         80, caxis_mollweide_600_200_0_4, color_label_mollweide, [], ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);
    
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_2x1{1,1}, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, ...
        80, colormap_to_use_ll, [-450 100], Title_1, FigPaperDir, [], [], color_label_mollweide{1}, color_label_size_small, ...
        [-30 30], [-150 30], {xxeqsq [-150+.1 -80.75] [-150+.1 -80.75] [-51 9.33] [-51 9.33]}, {yyeqsq [-0.05 -0.05] [0.05 0.05] [0.05 0.05] [0.05 0.05]}, {'y' 'k' 'k' 'k' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_2x1{2,1}, lat_viirs_gt0, lon_viirs_gt0, log10(number_viirs_gt0), ...
        80, colormap_to_use_num, caxis_mollweide_600_200_0_3(2,:), Title_2, FigPaperDir, FigTitle, [], color_label_mollweide{2}, color_label_size_small, ...
        [-30 30], [-150 30], {xxeqsq [-150+.1 -80.75] [-150+.1 -80.75] [-51 9.33] [-51 9.33]}, {yyeqsq [-0.05 -0.05] [0.05 0.05] [0.05 0.05] [0.05 0.05]}, {'y' 'k' 'k' 'k' 'k'})
    
    set(h, WindowStyle='docked')
    
    %%
    
    FigNo = 101;
    
    FigTitle = ['pcc_2x1_zoomed_eq_pacific_viirs_llc_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 subplot_details_2x1{1,1}(1) subplot_details_2x1{1,1}(2)]);
    
    Title_1 = 'a) $\widetilde{LL}_{VIIRS}-\widetilde{LL}_{LLC}$';
    Title_2 = 'b) log$_{10}$(Cutouts/HEALPix Cell)';
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_2x1, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, log10(number_viirs_gt0), [], [-10 10], [-130 -90], ...
    %         400, caxis_mollweide_600_0_0_3, color_label_mollweide, [], ylim_zonal_mean, Title_1, Title_2, FigPaperDir, FigTitle);
    %
    %     caxis([0 4])
    
    filename_out = [FigPaperDir FigTitle];
    print( filename_out, '-dpng')
    
    fprintf('Printing figure: %s\n', filename_out)
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_2x1{1,1}, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, ...
        400, colormap_to_use_ll, [-450 100], Title_1, FigPaperDir, [], [], 'LL_{VIIRS}-LL_{LLC}', color_label_size_small, ...
        [-10 10], [-130 -90], {xxeqsq [-130+.1 -90-.1] [-130+.1 -90-.1]}, {yyeqsq [-0.05 -0.05] [0.05 0.05]}, {'m' 'k' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_2x1{2,1}, lat_viirs_gt0, lon_viirs_gt0, log10(number_viirs_gt0), ...
        400, colormap_to_use_num, [0 2.5], Title_2, FigPaperDir, FigTitle, [], 'log_{10}(# Cutouts)', color_label_size_small, ...
        [-10 10], [-130 -90], {xxeqsq [-130+.1 -90-.1] [-130+.1 -90-.1]}, {yyeqsq [-0.05 -0.05] [0.05 0.05]}, {'m' 'k' 'k'})
    
    set(h, WindowStyle='docked')
    
end

%%

if plot_61
    
    FigNo = 61;
    
    xBox1 = [290.1 299.9 299.9 290.1 290.1];
    yBox1 = [34 34 36 36 34];
    
    xBox2 = [300.1 309.9 309.9 300.1 300.1];
    yBox2 = [40 40 42 42 40];
    
    FigTitle = ['pcc_2x2_zoomed_gs_viirs_llc_diff_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    %     h = undock_figure( FigNo, FigTitle, [100 1 750 1283]);
    h = undock_figure( FigNo, FigTitle, [100 1 1500 1283]);
    
    %     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    color_label_mollweide = [];
    
    Title_1 = 'a) $\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    Title_2 = 'b) $\widetilde{LL}_{VIIRS} - \widetilde{LL}_{LLC}$';
    Title_3 = 'c) $\widetilde{LL}_{VIIRS}$';
    Title_4 = 'd) $\widetilde{LL}_{LLC}$';
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_1x1, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, viirs_llc_diff_gt0, [], [30 45], 360-[80 40], ...
    %         800, [-300 300], color_label_mollweide, [], ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
    
    HEALPix_plot_Mollweid( FigNo, [1500 1283 0.05 0.55 0.40 0.40], lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, ...
        225, colormap_to_use_ll, [-700 700], Title_1, FigPaperDir, FigTitle, [], color_label_mollweide, color_label_size_small, ...
        [25 50], 360-[80 40], {Bathy_EC_contour_200_m(1,:) 360+GS(:,1) 360+GS_north(:,1) 360+GS_south(:,1)}, ...
        {Bathy_EC_contour_200_m(2,:) GS(:,2) GS_99(:,2) GS_01(:,2)}, {':r' 'm' 'k' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, [1500 1283 0.55 0.55 0.40 0.40], lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, ...
        225, colormap_to_use_ll, [-700 700], Title_2, FigPaperDir, FigTitle, ll_thresholds_gt0, color_label_mollweide, color_label_size_small, ...
        [25 50], 360-[80 40], {Bathy_EC_contour_200_m(1,:) 360+GS(:,1) 360+GS_north(:,1) 360+GS_south(:,1)}, ...
        {Bathy_EC_contour_200_m(2,:) GS(:,2) GS_99(:,2) GS_01(:,2)}, {':r' 'm' 'k' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, [1500 1283 0.05 0.05 0.40 0.40],   lat_viirs_gt0, lon_viirs_gt0, median_viirs_gt0, ...
        225, colormap_to_use_ll, [-800 1200], Title_3, FigPaperDir, FigTitle, [], color_label_mollweide, color_label_size_small, ...
        [25 50], 360-[80 40], {xBox1 xBox2 Bathy_EC_contour_200_m(1,:) 360+GS(:,1) 360+GS_north(:,1) 360+GS_south(:,1)}, ...
        {yBox1 yBox2 Bathy_EC_contour_200_m(2,:) GS(:,2) GS_99(:,2) GS_01(:,2)}, {'r' 'w' ':r' 'm' 'k' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, [1500 1283 0.55 0.05 0.40 0.40], lat_viirs_gt0, lon_viirs_gt0, median_llc_gt0, ...
        225, colormap_to_use_ll, [-800 1200], Title_4, FigPaperDir, FigTitle, [], color_label_mollweide, color_label_size_small, ...
        [25 50], 360-[80 40], {xBox1 xBox2 Bathy_EC_contour_200_m(1,:) 360+GS(:,1) 360+GS_north(:,1) 360+GS_south(:,1)}, ...
        {yBox1 yBox2 Bathy_EC_contour_200_m(2,:) GS(:,2) GS_99(:,2) GS_01(:,2)}, {'r' 'w' ':r' 'm' 'k' 'k'})
    
    set(h, WindowStyle='docked')
    
    %%
    FigNo = 62;
    
    FigTitle = ['pcc_2x1_zoomed_gs_head_tail_diff_LL_gt_' num2str(min_num) '_cutouts_per_cell'];
    h = undock_figure( FigNo, FigTitle, [100 1 750 subplot_details_2x1{1,1}(2)]);
    
    %     color_label_mollweide = '$\Delta_{LL} = \widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';
    color_label_mollweide = [];
    
    Title_1 = 'a) $\widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020}$';
    Title_2 = ['b) $\widetilde{LL}_{2012-2015} - \widetilde{LL}_{2018-2020} <$ ' num2str(ll_thresholds_gt0(1),3) ' or $>$ ' num2str(ll_thresholds_gt0(2),3)];
    
    %     subplot_temp = make_quad_plot( FigNo, subplot_details_1x1, lat_viirs_gt0, lon_viirs_gt0, viirs_llc_diff_gt0, viirs_llc_diff_gt0, [], [30 45], 360-[80 40], ...
    %         800, [-300 300], color_label_mollweide, [], ylim_zonal_mean, Title_1, [], FigPaperDir, FigTitle);
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_2x1{1,1}, lat_head_gt0, lon_head_gt0, head_tail_diff_gt0, ...
        225, colormap_to_use_ll, [-600 600], Title_1, FigPaperDir, FigTitle, [], color_label_mollweide, color_label_size_small, ...
        [25 50], 360-[80 40], {Bathy_EC_contour_200_m(1,:) 360+GS(:,1) 360+GS_north(:,1) 360+GS_south(:,1)}, {Bathy_EC_contour_200_m(2,:) GS(:,2) GS_north(:,2) GS_south(:,2)}, {'b' 'm' 'k' 'k'})
    
    HEALPix_plot_Mollweid( FigNo, subplot_details_2x1{2,1}, lat_head_gt0, lon_head_gt0, head_tail_diff_gt0, ...
        225, colormap_to_use_ll, [-600 600], Title_2, FigPaperDir, FigTitle, ll_thresholds_gt0, color_label_mollweide, color_label_size_small, ...
        [25 50], 360-[80 40], {Bathy_EC_contour_200_m(1,:) 360+GS(:,1) 360+GS_north(:,1) 360+GS_south(:,1)}, {Bathy_EC_contour_200_m(2,:) GS(:,2) GS_north(:,2) GS_south(:,2)}, {'b' 'm' 'k' 'k'})
    %         [25 50], 360-[80 40], {360+GS(:,1) 360+GS_north(:,1) 360+GS_south(:,1)}, {GS(:,2) GS_north(:,2) GS_south(:,2)}, {'m' 'k' 'k'})
    
    set(h, WindowStyle='docked')
    
end

%% Figures 81- Galleries

if plot_81
    
    %     tRange = [-2.5 2.5];
    tRange = [-3 3];
    
    FigNo = 81;
    FigTitle = ['pcc_2x3x3_ACC_viirs_llc_anomalous_gallery'];
    make_2x3x3_gallery(FigNo, FigPaperDir, FigTitle, unique_anomalous_acc_viirs_cutouts, unique_anomalous_acc_viirs_metadata, unique_anomalous_acc_llc_cutouts, anomalous_acc_llc_metadata, tRange, 11, {'a) VIIRS' 'b) LLC'})
    
    %%
    
    FigNo = 82;
    FigTitle = ['pcc_2x3x3_ACC_viirs_llc_agreeing_gallery'];
    make_2x3x3_gallery(FigNo, FigPaperDir, FigTitle, unique_agreeing_acc_viirs_cutouts, unique_agreeing_acc_viirs_metadata, unique_agreeing_acc_llc_cutouts, unique_agreeing_acc_llc_metadata, tRange, 11, {'a) VIIRS' 'b) LLC'})
    
end
%% Figure 51 through 53: mean sigma of LL as a function of the number of cutouts/HEALPix bin

if plot_51_54
    % Get the mean of the sigmas for each count of Cutouts/Cell
    
    clear xx
    for i=1:1600
        xx(i) = i;
        yyh(i) = mean(sigma_head(number_head==i));
        yyt(i) = mean(sigma_tail(number_tail==i));
        yyv(i) = mean(sigma_viirs(number_viirs==i));
        yyl(i) = mean(sigma_llc(number_llc==i), 'omitnan');
    end
    
    % Smooth these values
    
    yyhs = smoothdata( yyh, 'gaussian');
    yyts = smoothdata( yyt, 'gaussian');doc
    yyvs = smoothdata( yyv, 'gaussian');
    yyls = smoothdata( yyl, 'gaussian');
    
    % Simulate these functions.
    
    simSigma = 165;
    for i=4:1200
        r = randn(i,1000) * simSigma;
        rms(i) = mean(std(r,1));
    end
    rms(1:3) = nan;
    rmss = smoothdata( rms, 'gaussian');
    
    % And plot
    
    iFig = 51;
    figure(iFig)
    clf
    
    plot( xx, yyhs, linewidth=1.2)
    hold on
    plot( xx, yyts, linewidth=1.2)
    plot( xx, yyvs, linewidth=1.2)
    
    plot( rms, 'k', linewidth=2)
    
    grid on
    axis([0 200 0 300])
    set( gca, 'fontsize', 24)
    xlabel('Number of Cutouts/Cell')
    ylabel('$\overline{\sigma_{\widetilde{LL}}}$', interpreter='latex')
    
    legend('2012-2015', '2018-2020', '2012-2020', ['Simulated with sigma of ' num2str(simSigma)])
    
    title(['Gaussian Smoothed $\overline{\sigma_{LL}}$ for $\sigma=$' num2str(simSigma)], interpreter='latex', fontsize=30)
    
    filename_out = [FigPaperDir 'pcc_1x1_VIIRS_simulated_sigma_vs_cutouts_per_cell_for_intrasigma_HEALPix_of_' num2str(simSigma) ];
    print( filename_out, '-dpng')
    fprintf('Printing figure %i: %s\n', iFig, filename_out)
    
    % Next plot VIIRS, LLC and simulated
    
    iFig = 52;
    figure(iFig)
    clf
    
    plot( xx, yyvs, linewidth=2)
    hold on
    plot( xx, yyls, linewidth=1.2)
    plot( rms, 'k', linewidth=2)
    
    grid on
    axis([0 200 0 300])
    set( gca, 'fontsize', 24)
    xlabel('Number of Cutouts/Cell')
    ylabel('$\overline{\sigma_{\widetilde{LL}}}$', interpreter='latex')
    
    legend('VIIRS', 'LLC', ['Simulated with sigma of ' num2str(simSigma)])
    
    title(['Gaussian Smoothed $\overline{\sigma_{LL}}$ for $\sigma=$' num2str(simSigma)], interpreter='latex', fontsize=30)
    
    filename_out = [FigPaperDir 'pcc_1x1_VIIRS_LLC_simulated_sigma_vs_cutouts_per_cell_for_intrasigma_HEALPix_of_' num2str(simSigma) ];
    print( filename_out, '-dpng')
    fprintf('Printing figure %i: %s\n', iFig, filename_out)
    
    % Finally only VIIRS and simulated
    
    iFig = 53;
    figure(iFig)
    clf
    
    plot( xx, yyvs, linewidth=2)
    hold on
    plot( rms, 'k', linewidth=2)
    
    grid on
    axis([0 200 0 300])
    set( gca, 'fontsize', 24)
    xlabel('Number of Cutouts/Cell')
    ylabel('$\overline{\sigma_{\widetilde{LL}}}$', interpreter='latex')
    
    legend('VIIRS', ['Simulated with sigma of ' num2str(simSigma)])
    
    title(['Gaussian Smoothed $\overline{\sigma_{LL}}$ for $\sigma=$' num2str(simSigma)], interpreter='latex', fontsize=30)
    
    filename_out = [FigPaperDir 'pcc_1x1_VIIRS_simulated_sigma_vs_cutouts_per_cell_for_intrasigma_HEALPix_of_' num2str(simSigma) ];
    print( filename_out, '-dpng')
    fprintf('Printing figure %i: %s\n', iFig, filename_out)
end

%% Functions

% function [cutouts, new_metadata] = get_cutouts_and_metadata(fi, locs)
% % get_cutouts_and_metadata - will do just that for the specified file - PCC
% %
% % INPUT
% %   fi - h5 file with the data and metadata.
% %   locs - a 5 element vector with the columns for lat, lon, T90, T10 and LL.
% %
% % OUTPUT
% %   cutouts - 3d array with all the cutouts. The first two dimensions are
% %    the x and y size of each cutout, the last column if for each cutout.
% %   new_metadata - a struction function with the following filds:
% %      lat - latitude vector for all cutouts in this dataset.
% %      lon - longitude vector for all cutouts in this dataset.
% %      dt - T90-T10 for each cutout.
% %      LL - loglikelihood for each cutout.
% %
% cutouts = h5read( fi, '/valid');
% cutouts_metadata = h5read( fi, '/valid_metadata');
% for i=1:size(cutouts,3)
%     new_metadata.lat(i) = str2num(string(cutouts_metadata(locs(1),i)));
%     new_metadata.lon(i) = str2num(string(cutouts_metadata(locs(2),i)));
%     new_metadata.T90(i) = str2num(string(cutouts_metadata(locs(3),i)));
%     new_metadata.T10(i) = str2num(string(cutouts_metadata(locs(4),i)));
%     new_metadata.LL(i) = str2num(string(cutouts_metadata(locs(5),i)));
%     
%     nn = strfind(cutouts_metadata{1,i}, '/');
%     j = nn(end) + 1;
%     Year = str2num(cutouts_metadata{1,i}(j:j+3));
%     Month = str2num(cutouts_metadata{1,i}(j+4:j+5));
%     Day = str2num(cutouts_metadata{1,i}(j+6:j+7));
%     Hour = str2num(cutouts_metadata{1,i}(j+8:j+9));
%     Minute = str2num(cutouts_metadata{1,i}(j+10:j+11));
%     new_metadata.Matlab_datetime(i) = datenum(Year, Month, Day, Hour, Minute, 0);
% end
% new_metadata.dT = new_metadata.T90 - new_metadata.T10;
% 
% end