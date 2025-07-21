function dis_diur = p_energy_dis(temp, salt, lat, depth, border)
% FUNCTION p_energy_dis
% Calculates diurnal internal tide potential energy displacement
% Inputs:
%   temp - temperature profile (depth×time)
%   salt - salinity profile (depth×time)
%   lat  - latitude (degrees)
%   depth- depth array (m)
%   border- temporal border cutoff (samples)
% Output:
%   dis_diur - diurnal component of potential energy displacement

%% 1. Define tidal frequencies (rad/s)
wm2 = 1/(12.420601*3600);  % M2 constituent
ws2 = 1/(12*3600);         % S2
wk1 = 1/(23.9345*3600);    % K1 (diurnal)
wo1 = 1/(25.8193*3600);    % O1 
wn2 = 1/(12.6583*3600);    % N2
wq1 = 1/(26.8684*3600);    % Q1
wp1 = 1/(24.0659*3600);    % P1
wk2 = 1/(11.9672*3600);    % K2

%% 2. Calculate buoyancy frequency (N²)
bottom = depth(end);
top = depth(1);
[m1, m2] = size(temp);
N2 = nan(m1,m2);
pressure = sw_pres(depth, lat);

for j = 1:m2
    [n2, ~, ~] = sw_bfrq(salt(:,j), temp(:,j), pressure', lat);
    n2(n2<0) = nan;
    valid_idx = find(~isnan(n2));
    
    if length(valid_idx) >= 2
        N2(:,j) = interp1(depth(valid_idx), n2(valid_idx), depth, 'linear', 'extrap');
    end
end

%% 3. Horizontal interpolation of N²
N2_interp = nan(m1,m2);
for i = 1:m1
    valid_idx = find(~isnan(N2(i,:)));
    N2_interp(i,:) = interp1(valid_idx, N2(i,valid_idx), 1:m2, 'linear', 'extrap');
end

%% 4. Potential density calculation
pr = zeros(m1,1);
pden = nan(m1,m2);
for j = 1:m2
    pden(:,j) = sw_pden(salt(:,j), temp(:,j), pressure', pr);
end

%% 5. Background density (5-day lowpass)
smooth_pden = nan(m1,m2);
for i = 1:m1
    smooth_pden(i,:) = lowpass(pden(i,:)', 1/(5*24*3600), 1/(3600*3)); 
end

%% 6. Density perturbation calculation
sm_pden = smooth_pden(:, 1+border:end-border);
spden = pden(:, 1+border:end-border);
pden_perturbation = spden - sm_pden;
m2 = m2 - 2*border;  % Adjust time dimension

%% 7. Potential energy displacement
dis_all = nan(m1,m2);
for j = 1:m2
    valid_idx = find(~isnan(spden(:,j)));
    if ~isempty(valid_idx)
        dis_all(:,j) = depth' - interp1(spden(valid_idx,j), depth(valid_idx), sm_pden(:,j));
    end
end

%% 8. Interpolate displacement
dis_interp = nan(m1,m2);
for i = 1:m1
    valid_idx = find(~isnan(dis_all(i,:)));
    dis_interp(i,:) = interp1(valid_idx, dis_all(i,valid_idx), 1:m2, 'linear', 'extrap');
end

%% 9. Diurnal component extraction (K1 band)
fc = 1/(3*3600);
[b,a] = butter(4, [0.85,1.06]*wk1/(fc/2), 'bandpass');

dis_diur = nan(m1,m2);
for j = 1:m1
    dis_diur(j,:) = filter(b, a, dis_interp(j,:)');
end
end