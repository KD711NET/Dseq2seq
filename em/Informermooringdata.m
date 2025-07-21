%% Data Preprocessing for Diurnal Internal Tide Analysis
% Author: [Your Name]
% Date: [YYYY-MM-DD]
% Description: Processes ADCP data for diurnal internal tide extraction

clear all;
addpath('C:\Matlab\bin\sci'); % Add custom function path

%% 1. Load and Prepare ADCP Data
load('data202404-202407.mat');

% Depth and velocity parameters
dep01 = ADCP_depth;
len = length(dep01);
u1_all = ADCP_u./100; % Convert cm/s to m/s
v1_all = ADCP_v./100; % Convert cm/s to m/s

% Convert time format
for i = 1:length(Time)
    datetime_obj = datetime(Time(i), 'ConvertFrom', 'datenum');
    date_strings{i} = datestr(datetime_obj, 'yyyy-mm-dd HH:MM');
end

%% 2. Depth Selection and Baroclinic Mode Extraction
dep02 = dep01(7:46,:); % Select specific depth range (40 layers)
u1_all1 = u1_all(7:46,:);
v1_all1 = v1_all(7:46,:);

% Tidal frequency parameters (rad/s)
wm2 = 1/(12.420601*3600);  % M2 tide
ws2 = 1/(12*3600);         % S2 tide
wk1 = 1/(23.9345*3600);    % K1 tide
wo1 = 1/(25.8193*3600);    % O1 tide
wn2 = 1/(12.6583*3600);    % N2 tide
wq1 = 1/(26.8684*3600);    % Q1 tide
wp1 = 1/(24.0659*3600);    % P1 tide
wk2 = 1/(11.9672*3600);    % K2 tide

% Calculate barotropic and baroclinic components
top = dep02(1);
bottom = dep02(end);
[m1,m2] = size(u1_all1);

for ii = 1:m2
   u_barotropic(ii) = h_integral(u1_all1(:,ii),dep02,top,bottom)/(bottom-top);
   v_barotropic(ii) = h_integral(v1_all1(:,ii),dep02,top,bottom)/(bottom-top);
   u_baroclinic(:,ii) = u1_all1(:,ii) - u_barotropic(ii);
   v_baroclinic(:,ii) = v1_all1(:,ii) - v_barotropic(ii);
end

% Handle missing values
if any(isnan(u_baroclinic(:)))
    u_baroclinic = fillmissing(u_baroclinic, 'linear', 2);
end

%% 3. Temporal Filtering and Downsampling
[depth, time] = size(u_baroclinic);

% Original sampling (hourly)
fs_original = 1/3600;  % Sampling frequency (Hz)

% Target resolution (3-hourly)
fs_target = 1/(3*3600);  % Target frequency (Hz)

% Low-pass filtering (5th order Butterworth)
cutoff = fs_target/2;
order = 5;
u_baroclinic_filtered = zeros(size(u_baroclinic));

for i = 1:depth
    u_baroclinic_filtered(i,:) = lowpass_filter(u_baroclinic(i,:), cutoff, fs_original, order);
end

% Downsample to 3-hour intervals
downsample_factor = 3;
u_baroclinic_downsampled = u_baroclinic_filtered(:,1:downsample_factor:end);

%% 4. Diurnal Tide Extraction (K1 band)
fc = 1/(3*3600);  % Nyquist frequency for 3-hour data
[b,a] = butter(4,[0.85,1.06]*wk1/(fc/2),'bandpass'); % K1 band filter

for i = 1:m1
    uux = u_baroclinic_downsampled(i,:);
    udiur(i,:) = filter(b,a,uux'); % Diurnal component extraction
end

%% 5. Data Quality Control and Output
time = Time(1:3:end);
date_strings = cell(size(time));

for i = 1:length(time)
    datetime_obj = datetime(time(i), 'ConvertFrom', 'datenum');
    date_strings{i} = datestr(datetime_obj, 'yyyy-mm-dd HH:MM');
end

% Remove NaN-containing depth levels
valid_rows = ~any(isnan(udiur),2);
valid_indices = find(valid_rows);
lens = length(valid_indices);
filtered_udiur = udiur(valid_rows,:);

% Create output directory
folderPath = 'C:\code\pythonProject9\pythonProject1\diurmooring_data';
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
    disp(['Directory created: ', folderPath]);
end

%% 6. Save Results (Per Depth Level)
for i = 1:lens
    data = filtered_udiur(i,:);
    new_filename = fullfile(folderPath, ['diurdep', num2str(i), '.csv']);
    
    dateCellArray = cellstr(date_strings);
    outputData = [{'date','OT'}; dateCellArray', num2cell(data')];
    
    writecell(outputData, new_filename);            
end