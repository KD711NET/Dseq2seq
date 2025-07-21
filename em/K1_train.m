% Clear workspace and add path to scientific functions
clear all;
addpath 'C:\Matlab\bin\sci'

% Load bathymetry data
load('depgrd_zgh_NEW(1).mat');

% Define time range for analysis (Jan 1, 2015 to Jun 1, 2019)
time1 = datenum(2015,1,1,0,0,0);
time2 = datenum(2019,6,1,0,0,0);
border = 25; % Boundary truncation length

% Create time vectors with 3-hour intervals
tt1 = time1:1/8:time2;
tt2 = tt1(1+border:end-border);

% Get file list from data directory
rectotal = dir('D:\NMDIS-CORA2\SCS_original_grid');
filenames = {rectotal.name}';
filenames1 = filenames(3:end); % Remove system folders

% Calculate indices for time slicing
index1 = (time1-735965)*8+1;
index2 = (time2-735965)*8+1;

% Split data into 5 segments
num_points = 5;
indices = linspace(index1, index2, num_points + 1);
indices = round(indices);

% Define data slices
slice1 = [indices(1):indices(4)];
slice2 = [indices(5):indices(6)];
filenames2 = filenames1(slice1);

length_names = size(filenames2,1);

% Extract latitude/longitude data
a8 = depgrdzghNEW1(:,3:4);
a9 = depgrdzghNEW1(:,1:2);
a88 = table2array(a8);
a99 = table2array(a9);

% Load seasonal data
load('D:\dataseasonall.mat');

% Reshape coordinates into 2D grids
lon_2d = reshape(a88(:,1),113,85);
lat_2d = reshape(a88(:,2),113,85);
Index_2d = reshape(1:9605,113,85);

rows = size(Index_2d, 1);
cols = size(Index_2d, 2);

% Load depth data
load('depth_cora2.mat')

% Define South China Sea boundaries
lat_min = 8;
lat_max = 24;
lon_min = 108;
lon_max = 120.5;

% Get indices within defined boundaries
values = getValuesWithinBounds(lat_2d, lon_2d, Index_2d, lat_min, lat_max, lon_min, lon_max);
mall = values(1:40:end);
%Obtain 75 non empty data points

% Define tidal frequencies (in 1/seconds)
wm2 = 1/(12.420601*3600);  % M2 tidal frequency
ws2 = 1/(12*3600);         % S2 tidal frequency
wk1 = 1/(23.9345*3600);    % K1 tidal frequency
wo1 = 1/(25.8193*3600);    % O1 tidal frequency
wn2 = 1/(12.6583*3600);    % N2 tidal frequency
wq1 = 1/(26.8684*3600);    % Q1 tidal frequency
wp1 = 1/(24.0659*3600);    % P1 tidal frequency
wk2 = 1/(11.9672*3600);    % K2 tidal frequency

% Remove file extensions from names
fileNamesWithoutExtension2 = cell(size(filenames2));
for i = 1:numel(filenames2)
    [~, fileName, ~] = fileparts(filenames2{i});
    fileNamesWithoutExtension2{i} = fileName;
end
rowNames = string(fileNamesWithoutExtension2);

% Process diurnal tide data (K1 component)
for i = 1:length(mall)
    i
    % Extract variables for current point
    ux = squeeze(uxs(i,:,slice1));
    vx = squeeze(vxs(i,:,slice1));
    saltx = squeeze(saltxs(i,:,slice1));
    tempx = squeeze(tempxs(i,:,slice1));
    
    % Process data
    ux2 = redata2(ux);
    vx2 = redata2(vx);
    tempx2 = redata2(tempx);
    saltx2 = redata2(saltx);

    m = mall(i);
    mstr = num2str(m);
    
    % Get coordinates
    j = a99(m,1);
    w = a99(m,2);
    
    % Create output directory for diurnal data
    folder_path = 'C:\code\pythonProject9\pythonProject1\K1_diurtrain\';
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
        disp(['Folder created: ', folder_path]);
    else
        disp(['Folder exists: ', folder_path]);
    end

    new_filename = ['C:\code\pythonProject9\pythonProject1\K1_diurtrain\',mstr, '.csv'];
    
    if isempty(ux2) 
        % Create empty file if no data
        fileID = fopen(new_filename, 'w');
        fclose(fileID);     
    else
        m1 = length(ux2(:,1)); % Depth levels
        m2 = length(ux2(1,:)); % Time steps
        
        if m1<1
            % Create empty file if insufficient depth levels
            fileID = fopen(new_filename, 'w');
            fclose(fileID);
        else
            % Calculate barotropic and baroclinic components
            dep0 = depth_cora2(1:m1);
            for ii = 1:m2
               bottom = dep0(end);
               top = dep0(1);
               u_barotropic(ii) = h_integral(ux2(:,ii),dep0,top,bottom)/(bottom-top);
               u_baroclinic(:,ii) = ux2(:,ii) - u_barotropic(ii);
            end
            
            % Extract diurnal (K1) signal using bandpass filter
            fc = 1/(3*3600);
            [b,a] = butter(4,[0.85,1.06]*wk1/(fc/2),'bandpass');
            shen = size(tempx2,1);
            tindex = round(mean(find_thermocline(tempx2', dep0(1:shen))));
            uux = u_baroclinic(tindex,:);
            udiur(1,:) = filter(b,a,uux');
            data = udiur(1,:);
            
            % Format dates and prepare output
            dateTimes = datetime(rowNames, 'InputFormat', 'yyyyMMddHH');
            dateStrings = datestr(dateTimes, 'yyyy/mm/dd HH:MM');
            dateCellArray = cellstr(dateStrings);
            outputData = [{'date'},{'OT'};dateCellArray, num2cell(data')];
            
            % Write to CSV file
            writecell(outputData, new_filename);            
        end
    end
    clear ux ux2 u_baroclinic v_baroclinic udiur
end


%%
filenames2 = filenames1(slice2);

% Get number of files in this slice
length_names = size(filenames2,1);

% Extract latitude/longitude data from bathymetry table
a8 = depgrdzghNEW1(:,3:4);  % Columns 3-4 contain coordinates
a9 = depgrdzghNEW1(:,1:2);  % Columns 1-2 contain indices
a88 = table2array(a8);      % Convert to numerical array
a99 = table2array(a9);      % Convert to numerical array

% Load seasonal ocean data
load('D:\dataseasonall.mat');

%%
% Reshape coordinates into 2D grids (113x85)
lon_2d = reshape(a88(:,1),113,85);  % Longitude grid
lat_2d = reshape(a88(:,2),113,85);  % Latitude grid
Index_2d = reshape(1:9605,113,85);  % Reshape indices to match grid

% Get grid dimensions
rows = size(Index_2d, 1);
cols = size(Index_2d, 2);

% Load depth data
load('depth_cora2.mat')

% Define South China Sea study area boundaries
lat_min = 8;    % Minimum latitude
lat_max = 24;   % Maximum latitude
lon_min = 108;  % Minimum longitude
lon_max = 120.5; % Maximum longitude

% Get indices of grid points within study area
values = getValuesWithinBounds(lat_2d, lon_2d, Index_2d, lat_min, lat_max, lon_min, lon_max);
mall = values(1:40:end);  % Subsample points (every 40th)

%%
% Define tidal frequencies (in 1/seconds)
wm2 = 1/(12.420601*3600);  % M2 tidal frequency
ws2 = 1/(12*3600);         % S2 tidal frequency
wk1 = 1/(23.9345*3600);    % K1 tidal frequency
wo1 = 1/(25.8193*3600);    % O1 tidal frequency
wn2 = 1/(12.6583*3600);    % N2 tidal frequency
wq1 = 1/(26.8684*3600);    % Q1 tidal frequency
wp1 = 1/(24.0659*3600);    % P1 tidal frequency
wk2 = 1/(11.9672*3600);    % K2 tidal frequency

% Reload bathymetry data
load('depgrd_zgh_NEW(1).mat');

% Remove file extensions from names
fileNamesWithoutExtension2 = cell(size(filenames2));
for i = 1:numel(filenames2)
    [~, fileName, ~] = fileparts(filenames2{i});
    fileNamesWithoutExtension2{i} = fileName;
end
rowNames = string(fileNamesWithoutExtension2);  % Convert to string array
    
% Process each point in study area
for i = 1:length(mall)
    i  % Display current point number
    
    % Extract variables for current point from 3D arrays
    ux = squeeze(uxs(i,:,slice2));  % Eastward current
    vx = squeeze(vxs(i,:,slice2));  % Northward current
    saltx = squeeze(saltxs(i,:,slice2));  % Salinity
    tempx = squeeze(tempxs(i,:,slice2));  % Temperature
    
    % Process raw data
    ux2 = redata2(ux);
    vx2 = redata2(vx);
    tempx2 = redata2(tempx);
    saltx2 = redata2(saltx);

    m = mall(i);  % Get current point index
    mstr = num2str(m);  % Convert to string
    
    % Get coordinates
    j = a99(m,1);  % Longitude index
    w = a99(m,2);  % Latitude index
    
    % Create output directory for diurnal tide data
    folder_path = 'C:\code\pythonProject9\pythonProject1\K1_diurtrain\';
    
    % Check if directory exists, create if needed
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
        disp(['Folder created: ', folder_path]);
    else
        disp(['Folder exists: ', folder_path]);
    end

    % Create output filename with +1 suffix
    new_filename = ['C:\code\pythonProject9\pythonProject1\K1_diurtrain\',mstr,'+1', '.csv'];
    
    % Handle empty data case
    if isempty(ux2) 
        % Create empty file
        fileID = fopen(new_filename, 'w');
        fclose(fileID);     
    else
        m1 = length(ux2(:,1)); % Number of depth levels
        m2 = length(ux2(1,:)); % Number of time steps
        
        % Check if sufficient depth levels exist
        if m1 < 1
            % Create empty file if insufficient depth levels
            fileID = fopen(new_filename, 'w');
            fclose(fileID);
        else
            % Depth integration preparation
            dep0 = depth_cora2(1:m1); % Depth levels
            
            % Calculate barotropic and baroclinic components
            for ii = 1:m2
               bottom = dep0(end);  % Bottom depth
               top = dep0(1);      % Surface depth
               % Depth-average to get barotropic component
               u_barotropic(ii) = h_integral(ux2(:,ii),dep0,top,bottom)/(bottom-top);
               % Remove barotropic to get baroclinic component
               u_baroclinic(:,ii) = ux2(:,ii) - u_barotropic(ii);
            end
            
            % Extract diurnal (K1) internal tide signal using bandpass filter
            fc = 1/(3*3600);  % Filter cutoff frequency
            [b,a] = butter(4,[0.85,1.06]*wk1/(fc/2),'bandpass'); % 4th order Butterworth filter
            shen = size(tempx2,1);  % Number of depth levels
            % Find thermocline depth
            tindex = round(mean(find_thermocline(tempx2', dep0(1:shen)))); 
            uux = u_baroclinic(tindex,:);  % Currents at thermocline
            udiur(1,:) = filter(b,a,uux'); % Filtered diurnal signal
            data = udiur(1,:);  % Final processed data
            
            % Format timestamps for output
            dateTimes = datetime(rowNames, 'InputFormat', 'yyyyMMddHH');
            dateStrings = datestr(dateTimes, 'yyyy/mm/dd HH:MM');
            dateCellArray = cellstr(dateStrings);
      
            % Prepare output data array
            outputData = [{'date'},{'OT'};dateCellArray, num2cell(data')];
            
            % Write to CSV file
            writecell(outputData, new_filename);            
        end
    end
    % Clear variables for next iteration
    clear ux ux2 u_baroclinic v_baroclinic udiur
end


