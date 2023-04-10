function make_4x3x3_gallery(FigNo, FigDir, FigTitle, cutouts_master, metadata_master, tRange, num_ticks, gallery_labels, means_to_use)
% make_4x3x3_gallery - make 2 sets of 3x3 galleries - PCC
%
% INPUT
%   FigNo - figure number.
%   FigDir - the directory into which the image is saved. Nothing saved if empty.
%   FigTitle - title to use for figure and saved file.
%   first_set_label - label to be written to the left of the top row of galleries.
%   second_set_label - label to be written to the left of the bottom row of galleries.
%   cutouts_master - cell array with the cutouts for the 4 sets of galleries
%   metadata_master - cell array with the metadata for the 4 sets of galleries
%   tRange - 2-element vector for the temperature range for colorbar. If
%    empty will take the default caxis.
%   num_ticks - number of ticks for the colorbar axis (including top and
%    bottom values).
%   gallery_labels - a 4-elememt cell array with the labels of the two
%    columns of galleries followed by the labels for the two rows. something
%    like: {'VIIRS Cutouts' 'LLC Cutouts' 'Anomalous' 'Similar'}.
%   means_to_use - in title of each gallery
%

% datetime_format = 'yyyy/mm/dd HH:MM';
datetime_format = 'mm/dd/yy HH';

hfig = undock_figure( FigNo, FigTitle, [10 500 1500 1750]);

title_prefix = {'a) ' 'b) ' 'c) ' 'd) '};

% iCutout_spacing = floor(size(cutouts_1,3) / 9);

% Get the offsets for the galleries.

offsets = [0.05 0.52; 0.55 0.52; 0.05 0.01; 0.55 0.01];

upper_label = [0.985 0.4925];
left_label = 0.01;

x_cutout_step = 0.14;
y_cutout_step = 0.15;

x_cutout_size = 0.13;
y_cutout_size = 0.13;

% Loop over galleries.

kGallery = 0;
for iGallery=1:2
    for jGallery=1:2
        
        kGallery = kGallery + 1;
        
        cutouts = cutouts_master{kGallery};
        metadata = metadata_master{kGallery};
        
        % get the random sampling for this set of cutouts.
        
%         if jGallery == 1
%             if iGallery == 1
%                 nsamp_t = datasample([1:size(cutouts,3)], 9, replace=false);
%                 nsamp = sort(nsamp_t);
%             else
%                 nsamp_t = datasample([1:size(cutouts,3)], 9, replace=false);
%                 nsamp = sort(nsamp_t);
%             end
%         end
        nsamp = [1:9];
        
        jCutout = 0;
                
        for nCutout=1:3
            for mCutout=1:3
                subplot( position=[offsets(kGallery,1)+(mCutout-1)*x_cutout_step offsets(kGallery,2)++(3-nCutout)*y_cutout_step x_cutout_size y_cutout_size])
                
                jCutout = jCutout + 1;
                iCutout = nsamp(jCutout);
                
                imagesc(squeeze(cutouts(:,:,iCutout)))
                set(gca,xtick=[],ytick=[])
                if isempty(tRange) == 0
                    caxis(tRange)
                end
                
                %                 title([datestr(metadata.Matlab_datetime(iCutout), datetime_format) ...
                %                     '  LL=' num2str(metadata.LL(iCutout),'%5.0f') ...
                %                     '  dT=' num2str(metadata.dT(iCutout),2)], fontsize=18)
                
                if jGallery == 1
                    title([datestr(metadata.Matlab_datetime(iCutout), datetime_format)  ...
                        'h (' num2str(metadata.LL(iCutout),'%5.0f') ...
                        ', ' num2str(metadata.dT(iCutout),2) ')'], fontsize=18)
                else
                    title(['(' num2str(metadata.LL(iCutout),'%5.0f') ...
                        ', ' num2str(metadata.dT(iCutout),2) ')'], fontsize=18)
                end
            end
        end
        
        % Add the top labels.
        
        % Get the mean and dT for the selected cutouts
        
        selected_sst = cutouts(:,:,nsamp);
        if isempty(means_to_use)
            sorted_selected_sst = sort(selected_sst(:));

            LL_All_sst = mean(metadata.LL);
            dT_All_sst = mean(metadata.dT);

            LL_selected_sst = mean(metadata.LL(nsamp));
            dT_selected_sst = sorted_selected_sst(floor(0.9*numel(sorted_selected_sst))) - sorted_selected_sst(floor(0.1*numel(sorted_selected_sst)));
        else
            LL_All_sst = means_to_use(iGallery, jGallery).LL_All;
            dT_All_sst = means_to_use(iGallery, jGallery).dT_All;

            LL_selected_sst = means_to_use(iGallery, jGallery).LL_Selected;
            dT_selected_sst = means_to_use(iGallery, jGallery).dT_Selected;
        end
        
%         if jGallery == 1
%             subplot(position=[offsets(1,1)+1.5*x_cutout_step upper_label(iGallery) 0.2 0.01])
%             text( 0, 0, [title_prefix{kGallery} gallery_labels{1} ...
            subplot(position=[offsets(jGallery,1)+1.5*x_cutout_step upper_label(iGallery) 0.2 0.01])
            text( 0, 0, [title_prefix{kGallery} gallery_labels{jGallery} ...
                '   LL=' num2str(LL_All_sst,'%5.0f') '   dT=' num2str(dT_All_sst,'%5.1f') ...
                'K  (' num2str(LL_selected_sst,'%5.0f') ', ' num2str(dT_selected_sst,'%5.1f') 'K)'], ...
                fontsize=32, HorizontalAlignment='center')
            set( gca, xtick=[] ,ytick=[], xColor=[1 1 1])
            h = get(gca,'XAxis');
            h.Visible = 'off';
            h = get(gca,'YAxis');
            h.Visible = 'off';
%         else
%             subplot(position=[offsets(2,1)+1.5*x_cutout_step upper_label(iGallery) 0.2 0.01]);
%             text( 0, 0, [title_prefix{kGallery} gallery_labels{2} ...
%                 '     LL=' num2str(LL_All_sst,'%5.0f') '     dT=' num2str(dT_All_sst,'%5.1f') ...
%                 ' K  (' num2str(LL_selected_sst,'%5.0f') ', ' num2str(dT_selected_sst,'%5.1f') ' k)'], ...
%                 fontsize=32, HorizontalAlignment='center')
%             set( gca, xtick=[] ,ytick=[], xColor=[1 1 1])
%             h = get(gca,'XAxis');
%             h.Visible = 'off';
%             h = get(gca,'YAxis');
%             h.Visible = 'off';
%         end
        
        % Add the label on the left hand side.
        
        if jGallery == 1
            if iGallery == 1
                subplot(position=[left_label offsets(1,2)+1.5*y_cutout_step 0.02 0.02])
                text( 0, 0, [gallery_labels{3}], fontsize=32, HorizontalAlignment='center', rotation=90)
                set( gca, xtick=[] ,ytick=[], xColor=[1 1 1])
                h = get(gca,'XAxis');
                h.Visible = 'off';
                h = get(gca,'YAxis');
                h.Visible = 'off';
            else
                subplot(position=[left_label offsets(3,2)+1.5*y_cutout_step 0.02 0.02]);
                text( 0, 0, [gallery_labels{4}], fontsize=32, HorizontalAlignment='center', rotation=90)
                set( gca, xtick=[] ,ytick=[], xColor=[1 1 1])
                h = get(gca,'XAxis');
                h.Visible = 'off';
                h = get(gca,'YAxis');
                h.Visible = 'off';
            end
        end
    end
end

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

    colormap(jet)
    
    for i=1:256
        color_image(i,:) = -((i-1)*5/255 - 2.5) * [1 1 1];
    end
    
    subplot( position=[0.48 0.0200 0.010 0.9])
    
    imagesc(color_image)
    
    if isempty(tRange) == 0
        set(gca, XTickLabel=[], YTick=[1:256/(num_ticks-1):256 256], ...
            YTickLabel=[tRange(2):(tRange(1)-tRange(2))/(num_ticks-1):tRange(1) ...
            tRange(2)], YAxisLocation='right', fontsize=16)
    else
        set(gca, XTickLabel=[], YTick=[1:256/(num_ticks-1):256 256], ...
            YAxisLocation='right', fontsize=16)
    end
    
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
