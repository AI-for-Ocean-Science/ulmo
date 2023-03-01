function h = undock_figure( FigNo, FigName, Position)
% undock_figure - open, rename and undock this figure.
%
% INPUT
%   FigNo - figure number.
%   FigName - name to give this figure.
%   Position - 4 element vector used in set(gcf command.
%
% OUTPUT
%   h - figure handle.

FigNameFixed = strrep( FigName, '_', ' ');

% close the figure if already open to make clean out the projection,
% which isn't cleared with clf.

if ishandle(FigNo)
    close(FigNo)
end

% OK, now create it.

h = figure(FigNo);

set( h, name=FigNameFixed, numbertitle='on')
clf

set(h, WindowStyle='normal')

set(gcf,'position', Position)

