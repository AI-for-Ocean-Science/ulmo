function xy_pair = get_pixel_loc_in_figure(Instructions)
% get_pixel_loc_in_figure - ast it says - PCC
%
% This function will get the location of a point in a figure in pixels
% relative to the lower left hand corner of the figure. It is used to line
% up a subplot with one that has weird coordinates such as a Mollweide
% plot.
%   

x = nan;
y = nan;

fprintf('\n%s \n', Instructions)
ans = input('Then click <cr> to capture the point, k for keyboard or q to quit: ', 's');

switch ans
    case 'k'
        keyboard
        
    case 'q'
        return
        
    otherwise
end

xy_pair = get(gcf,'currentpoint');

end

