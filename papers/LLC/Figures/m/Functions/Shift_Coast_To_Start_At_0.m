function [ long, lat ] = Shift_Coast_To_Start_At_0
%
% [ long, lat ] = Shift_Coast_To_Start_At_0;
%
%   This function will read in the coast line from coast and then shift so
%   that it goes from 0 to 360 rather than -180 to 180.
%
%   Written by Peter Cornillon 9/22/13

load coastlines.mat

longT = coastlon;
latT = coastlat;
clear coastlat coastlon

nn = find(longT < 0);
longT(nn) = 360 + longT(nn);

longdiff = diff(longT);

% Now have to add nan between points that are separated at the prime
% meridian.

jlong = 0;
for ilong=1:length(longT)-1
    jlong = jlong + 1;
    long(jlong) = longT(ilong);
    lat(jlong) = latT(ilong);
    if abs(longdiff(ilong)) > 10
        jlong = jlong + 1;
        long(jlong) = nan;
        lat(jlong) = nan;
    end
end

jlong = jlong + 1;
long(jlong) = longT(ilong);
lat(jlong) = latT(ilong);

end

