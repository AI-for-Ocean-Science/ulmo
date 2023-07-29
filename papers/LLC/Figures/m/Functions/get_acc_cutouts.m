function [cutouts, metadata_structure, unique_cutouts, unique_metadata_structure] = get_acc_cutouts_t( dataset, healpix_cell_number)
% get_acc_cutputs - gets the cutouts for the specified HEALPix cells and combines them - PCC
%
% For each HEALPix cell read all of the cutouts and associated metadata and
% combine them into a structure for the metadata and a 3-D array for the
% cutouts.
%
% INPUT
%   dataset - 'VIIRS' for VIIRS SST and 'LLC' for LLC SST.
%   healpix_cell_number - cell array of HEALPix cell numbers.
%
% OUTPUT
%   cutouts - 3-D array of cutouts. The 3rd dimension separates cutouts.
%   unique_cutouts - 3-D array of uniques cutouts, only one for a given
%    HEALPix cell on the given day and time.
%   metadata_structure_t - structure of metdata, which includes:
%       filename - the filename from which the cutout was extracted.
%       lat - the latitude of the cutout.
%       lon - the longitude of the cutout.
%       LL - the log likelihood of the cutout.
%       dT - the temperature range in the cutout 10th percentile to 90th percentile.
%       Matlab_datetime - date and time of the cutout in Matlab time.
%       matlab_date_index - the indices of cutouts sorted by Matlab time.
%       unique_index - the set of unique (only one cutout at location and
%        time) indices of cutouts. 
%
% EXAMPLE
%   healpix_cell_number = {'44318' '44319' '44513' '44514'};
%   [cutouts, unique_cutouts, metadata_structure_t] = get_acc_cutputs( 'LLC', healpix_cell_number);
%
% healpix_cell_number = {'44318' '44319' '44513' '44514'};

% Which dataset.

dataset = lower(dataset);
if strcmp( dataset, 'viirs')
    ll_index = 19;
elseif strcmp( dataset, 'llc')
    ll_index = 27;
else
    fprintf('Did not recognize the dataset you specified %s. It should be either ''VIIRS'' or ''LLC''.\n', dataset)
    metadata_structure_t = {''};
    cutouts = [];
    return
end

jCutout = 0;

for iHEALPix=1:length(healpix_cell_number)
    
    fprintf('Working on HEALPix cell %s\n', healpix_cell_number{iHEALPix})
    
    % Get the data for this HEALPix cell
    
    fi.healpix_cell_number{iHEALPix} = ['~/Dropbox/ComputerPrograms/Satellite_Model_SST_Processing/AI-SST/Data/HEALPix/Southern_Ocean_Cutouts/ACC_cutouts_' healpix_cell_number{iHEALPix} '_'  dataset '.h5'];
    
    cutouts_tt = h5read( fi.healpix_cell_number{iHEALPix}, '/valid');
    
    metadata_t = h5read( fi.healpix_cell_number{iHEALPix}, '/valid_metadata');
    
    % Unpack cell arrays and get Matlab date, time for each cutout
    
    for iCutout=1:size(metadata_t,2)
        jCutout = jCutout + 1;
                
        metadata_structure_t.filename{jCutout} = metadata_t{1,iCutout};
        metadata_structure_t.lat(jCutout) = str2num(metadata_t{4,iCutout});
        metadata_structure_t.lon(jCutout) = str2num(metadata_t{5,iCutout});
        metadata_structure_t.LL(jCutout) = str2num(metadata_t{ll_index,iCutout});
        
        sst = squeeze(cutouts_tt(:,:,iCutout));
        sst_sorted = sort(sst(:));
        metadata_structure_t.dT(jCutout) = sst_sorted(3686) - sst_sorted(410);

        % Get the Matlab date-time for this cutout
        
        temp_filename = metadata_structure_t.filename{jCutout};
        nn = strfind(temp_filename, '/');
        dd = temp_filename(nn(end)+1:nn(end)+14);
        Year = str2num(dd(1:4));
        Month = str2num(dd(7:8));
        Day = str2num(dd(5:6));
        Year = str2num(dd(1:4));
        Hour = str2num(dd(9:10));
        Minute = str2num(dd(11:12));
        metadata_structure_t.Matlab_datetime(jCutout) = datenum( Year, Month, Day, Hour, Minute, 0);

        cutouts_t(:,:,jCutout) = sst;
    end
end

% Now reorder the cutouts by date-time.

[matlab_date_sorted, matlab_date_index] = sort(metadata_structure_t.Matlab_datetime);

for iCutout=1:size(matlab_date_sorted,2)
    jCutout = matlab_date_index(iCutout);
    
    metadata_structure.filename{iCutout} = metadata_structure_t.filename{jCutout};
    metadata_structure.lat(iCutout) = metadata_structure_t.lat(jCutout);
    metadata_structure.lon(iCutout) = metadata_structure_t.lon(jCutout);
    metadata_structure.LL(iCutout) = metadata_structure_t.LL(jCutout);
    metadata_structure.dT(iCutout) = metadata_structure_t.dT(jCutout);
    metadata_structure.Matlab_datetime(iCutout) = metadata_structure_t.Matlab_datetime(jCutout);
    
    cutouts(:,:,iCutout) = cutouts_t(:,:,jCutout);
    imagesc(squeeze(cutouts(:,:,iCutout)))
end

% Now get unique cutouts

nndd = diff(metadata_structure.Matlab_datetime);

jCutout = 1;
unique_cutouts(:,:,jCutout) = cutouts(:,:,1);

unique_metadata_structure.filename{jCutout} = metadata_structure.filename{1};
unique_metadata_structure.lat(jCutout) = metadata_structure.lat(1);
unique_metadata_structure.lon(jCutout) = metadata_structure.lon(1);
unique_metadata_structure.LL(jCutout) = metadata_structure.LL(1);
unique_metadata_structure.dT(jCutout) = metadata_structure.dT(1);
unique_metadata_structure.Matlab_datetime(jCutout) = metadata_structure.Matlab_datetime(1);

for iCutout=2:length(metadata_structure_t.Matlab_datetime)
    if nndd(iCutout-1) ~= 0
        jCutout = jCutout + 1;

        unique_cutouts(:,:,jCutout) = cutouts(:,:,iCutout);
        
        unique_metadata_structure.filename{jCutout} = metadata_structure.filename{iCutout};
        unique_metadata_structure.lat(jCutout) = metadata_structure.lat(iCutout);
        unique_metadata_structure.lon(jCutout) = metadata_structure.lon(iCutout);
        unique_metadata_structure.LL(jCutout) = metadata_structure.LL(iCutout);
        unique_metadata_structure.dT(jCutout) = metadata_structure.dT(iCutout);
        unique_metadata_structure.Matlab_datetime(jCutout) = metadata_structure.Matlab_datetime(iCutout);
        
%         metadata_structure.unique_index(jCutout) = metadata_structure_t.matlab_date_index(iCutout);
    end
end

end

