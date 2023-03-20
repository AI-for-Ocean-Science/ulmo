% reorder_llc - matches LLC to VIIRS cutouts - PCC
%
% For set of VIIRS and LLC cutouts and their asociated metadata, this
% script will cycle through the VIIRS cutouts looking for LLC matches. When
% one is found, it will write a new metadata and cutout structure for
% VIIRS, in the event that a matchup was not found and a new metadata and
% cutout structure for LLC with the same locations in the structure. To be
% a match, date, lat and lon much be the same.
%

region_in = {'gulf' 'eq', 'south'};
region_out = {'gs' 'eq', 'acc'};

% The next array indicates which locations in the input table of VIIRS (1st
% row and LLC 2nd row.

element_vector = ...
    [4 5 17 18 19; ...
     4 5 17 18 27];

for iRegion=1:numel(region_in)
    
    % Read in the data.
    
    eval(['fi = ''~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/Cutouts/2022-Nov/viirs_' region_in{iRegion} '_rect_cutouts.h5'';'])
    eval(['[ viirs_cutouts, viirs_metadata] = get_cutouts_and_metadata(fi, [' num2str(element_vector(1,:)) ']);'])

    eval(['fi = ''~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/Cutouts/2022-Nov/llc_' region_in{iRegion} '_rect_cutouts.h5'';'])
    eval(['[ llc_cutouts, llc_metadata] = get_cutouts_and_metadata(fi, [' num2str(element_vector(2,:)) ']);'])

%     % Put the data in temporary variable.
%     
%     eval(['viirs_metadata = ' region{iRegion} '_viirs_metadata;'])
%     eval(['viirs_cutouts = ' region{iRegion} '_viirs_cutouts;'])
%     
%     eval(['llc_metadata = ' region{iRegion} '_llc_metadata;'])
%     eval(['llc_cutouts = ' region{iRegion} '_llc_cutouts;'])
    
    iMatch = 0;
    for viirs_index=1:numel(viirs_metadata.LL)
        [error_return, matchup_index, return_dist_sep, return_time_sep] = find_viirs_llc_matches( viirs_index, viirs_metadata, llc_metadata);
        
        if (return_dist_sep == 0) & (return_time_sep == 0)
            iMatch = iMatch + 1;
            
            field_names = fieldnames(viirs_metadata);
            for iFieldname=1:numel(field_names)
                eval([region_out{iRegion} '_viirs_metadata_out.' field_names{iFieldname} '(iMatch) = viirs_metadata.' field_names{iFieldname} '(viirs_index);'])
                eval([region_out{iRegion} '_llc_metadata_out.' field_names{iFieldname} '(iMatch) = llc_metadata.' field_names{iFieldname} '(matchup_index);'])
            end
            
            eval([region_out{iRegion} '_viirs_cutouts_out(:,:,iMatch) = viirs_cutouts(:,:,viirs_index);'])
            eval([region_out{iRegion} '_llc_cutouts_out(:,:,iMatch) = llc_cutouts(:,:,matchup_index);'])
        end
    end
end

save('~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/regional_cutouts_and_metadata', '*_out')