function make_2x3x3_gallery(FigNo, FigDir, FigTitle, cutouts_1, metadata_1, cutouts_2, metadata_2, tRange, num_ticks, gallery_labels)
% make_2x3x3_gallery - make a pair of 3x3 galleries - PCC
%
% INPUT
%   FigNo - figure number.
%   FigDir - the directory into which the image is saved. Nothing saved if empty.
%   FigTitle - title to use for figure and saved file.
%   cutouts_1 - cutouts for first set of 3x3.
%   metadata_1 - metadata for first set of 3x3.
%   cutouts_2 - cutouts for second set of 3x3.
%   metadata_2 - metadata for second set of 3x3.
%   tRange - 2-element vector for the temperature range for colorbar.
%   num_ticks - number of ticks for the colorbar axis (including top and bottom values).
%   gallery_labels - a 2-elememt cell array with the labels of the two 
%    galleries. These will be put at the top of each of the two of each
%    gallery. This should be something like {'VIIRS' 'LLC'}.
%

datetime_format = 'yyyy/mm/dd HH:MM';

hfig = undock_figure( FigNo, FigTitle, [10 500 2000 1000]);

iCutout_spacing = floor(size(cutouts_1,3) / 9);

nsamp_t = datasample([1:size(cutouts_1,3)], 9, replace=false);
nsamp = sort(nsamp_t);

% First do VIIRS

% iCutout = 1 - iCutout_spacing;
jCutout = 0;

for nCutout=1:3
    for mCutout=1:3
        subplot( position=[0.01+(mCutout-1)*0.16 (3-nCutout)*0.32 0.138 0.29])
        
%         iCutout = iCutout + iCutout_spacing;
        jCutout = jCutout + 1;
        iCutout = nsamp(jCutout);
        
        imagesc(squeeze(cutouts_1(:,:,iCutout)))
        set(gca,xtick=[],ytick=[])
        caxis(tRange)
        
        title([datestr(metadata_1.Matlab_datetime(iCutout), datetime_format) ...
            '  LL=' num2str(metadata_1.LL(iCutout),'%5.0f') ...
            '  dT=' num2str(metadata_1.dT(iCutout),2)], fontsize=18)
    end
end

% Add the label at the top for VIIRS

subplot(position=[0.240 0.985 0.2 0.03])
text( 0, 0, [gallery_labels{1} '     LL=' num2str(mean(metadata_1.LL),'%5.0f') '     dT=' num2str(mean(metadata_1.dT),'%5.1f')], fontsize=32, HorizontalAlignment='center')
set( gca, xtick=[] ,ytick=[], xColor=[1 1 1])
h = get(gca,'XAxis');
h.Visible = 'off';
h = get(gca,'YAxis');
h.Visible = 'off';

% Now do LLC.

cutout_offset = 0.53;

% iCutout = 1 - iCutout_spacing;
jCutout = 0;

for nCutout=1:3
    for mCutout=1:3
        subplot( position=[cutout_offset+(mCutout-1)*0.16 (3-nCutout)*0.32 0.138 0.29])
        
%         iCutout = iCutout + iCutout_spacing;
        jCutout = jCutout + 1;
        iCutout = nsamp(jCutout);

        imagesc(squeeze(cutouts_2(:,:,iCutout)))
        set(gca,xtick=[],ytick=[])
        caxis(tRange)
        
%         title([datestr(metadata_2.Matlab_datetime(iCutout), datetime_format) ...
%             '  LL=' num2str(metadata_2.LL(iCutout),'%5.0f') ...
%             '  dT=' num2str(metadata_2.dT(iCutout),2)], fontsize=18)
        title(['LL=' num2str(metadata_2.LL(iCutout),'%5.0f') ...
            '  dT=' num2str(metadata_2.dT(iCutout),2)], fontsize=18)
    end
end

% Add the label at the top for LLC

subplot(position=[0.240+cutout_offset 0.985 0.2 0.03]);
text( 0, 0, [gallery_labels{2} '     LL=' num2str(mean(metadata_2.LL),'%5.0f') '     dT=' num2str(mean(metadata_2.dT),'%5.1f')], fontsize=32, HorizontalAlignment='center')
set( gca, xtick=[] ,ytick=[], xColor=[1 1 1])
h = get(gca,'XAxis');
h.Visible = 'off';
h = get(gca,'YAxis');
h.Visible = 'off';

% Now add the colorbar

thin_colorbar = 0;
if thin_colorbar
    % This version uses the Matlab generated colorbar.
    
    subplot( position=[0.472 0.0400 0.0350 0.9])
    
    colormap(jet)
    caxis([-2.5 2.5])
    c = colorbar;
    c.FontSize = 16;
    c.Label.String = 'SSTa (K)';
    c.Label.FontSize = 16;
    
    h = get(gca,'XAxis');
    h.Visible = 'off';
    h = get(gca,'YAxis');
    h.Visible = 'off';
else
    % This version uses the colorbar I generated.
    
    for i=1:256
        color_image(i,:) = -((i-1)*5/255 - 2.5) * [1 1 1];
    end
    
    subplot( position=[0.48 0.0200 0.010 0.9])
    
    imagesc(color_image)
    
    set(gca, XTickLabel=[], YTick=[1:256/(num_ticks-1):256 256], ...
        YTickLabel=[tRange(2):(tRange(1)-tRange(2))/(num_ticks-1):tRange(1) tRange(2)], YAxisLocation='right', fontsize=16)
    
    ylabel('SSTa (K)')
end


% Print the figure.

if isempty(FigTitle)==0
    filename_out = [FigDir FigTitle];
    print( filename_out, '-dpng')
    fprintf('Printing figure %i: %s\n', FigNo, filename_out)
end

% And dock the figure.

set(hfig, WindowStyle='docked')

end
