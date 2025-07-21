clear all;
% All depth values should be positive
% m=8099 represents the nearest data point to EW3 mooring location
addpath 'C:\Matlab\bin\sci'
load('depgrd_zgh_NEW(1).mat');

% Extract and convert data from tables
a8 = depgrdzghNEW1(:,3:4);
a9 = depgrdzghNEW1(:,1:2);
a88 = table2array(a8);
a99 = table2array(a9);

% Reshape data into 2D grids
lon_2d = reshape(a88(:,1),113,85);
lat_2d = reshape(a88(:,2),113,85);
Index_2d = reshape(1:9605,113,85);

rows = size(Index_2d, 1);
cols = size(Index_2d, 2);

% Target number of grid points
targetPoints = 900;

% Calculate current total grid points
totalPoints = rows * cols;

% Calculate scaling ratio
scaleRatio = targetPoints / totalPoints;

% Calculate new dimensions
newRows = round(rows * sqrt(scaleRatio));
newCols = round(cols * sqrt(scaleRatio));

% Display new dimensions
disp(['New rows: ' num2str(newRows)]);
disp(['New columns: ' num2str(newCols)]);

% Ensure non-zero dimensions
if newRows == 0
    newRows = 1;
end
if newCols == 0
    newCols = 1;
end

% Calculate sampling intervals
rowInterval = floor(rows / newRows);
colInterval = floor(cols / newCols);

% Downsample data
resizedIndex_2d = Index_2d(1:rowInterval:end, 1:colInterval:end);
newlon_2d = lon_2d(1:rowInterval:end, 1:colInterval:end);
newlat_2d = lat_2d(1:rowInterval:end, 1:colInterval:end);

% Display new grid size
disp(['Resized grid points: ' num2str(size(resizedIndex_2d, 1) * size(resizedIndex_2d, 2))]);

mall = reshape(resizedIndex_2d,1,[]);

%%
% Define time period and tidal frequencies
time1 = datenum(2015,1,1,0,0,0);
time2 = datenum(2019,12,15,0,0,0);

% Tidal frequency definitions (1/period in seconds)
wm2 = 1/(12.420601*3600);  % M2 tide
ws2 = 1/(12*3600);         % S2 tide
wk1 = 1/(23.9345*3600);    % K1 tide
wo1 = 1/(25.8193*3600);    % O1 tide
wn2 = 1/(12.6583*3600);    % N2 tide
wq1 = 1/(26.8684*3600);    % Q1 tide
wp1 = 1/(24.0659*3600);    % P1 tide
wk2 = 1/(11.9672*3600);    % K2 tide

border = 25;  % Boundary truncation length
tt1 = time1:1/8:time2;
tt2 = tt1(1+border:end-border);

% Get file list from directory
rectotal = dir('D:\NMDIS-CORA2\SCS_original_grid');
filenames = {rectotal.name}';
filenames1 = filenames(3:end);  % Skip . and ..

% Calculate time indices
index1 = (time1-735965)*8+1;
index2 = (time2-735965)*8+1;
filenames2 = filenames1(index1:index2);
filenames3 = filenames2(1+border:end-border);

length_names = size(filenames2,1);
length_names2 = size(filenames3,1);

% Initialize data arrays
saltxs = zeros(1102, 35, 14473);
tempxs = zeros(1102, 35, 14473);
uxs = zeros(1102, 35, 14473);
vxs = zeros(1102, 35, 14473);

% Main processing loop
for k = 1:length_names
    k  % Display current iteration
    
    path = 'D:\NMDIS-CORA2\SCS_original_grid\';
    allin = 1:length(mall);
    
    % Get indices for data extraction
    j = a99(mall,1);
    w = a99(mall,2);
    
    % Get latitude/longitude coordinates
    lat1 = depgrdzghNEW1(mall,4);
    lon1 = depgrdzghNEW1(mall,3);
    lat = table2array(lat1);
    lon = table2array(lon1);
    
    % Read CORA data file
    K_trace = strcat(path,filenames2(k));
    [ssh, u, v, temp, salt, time] = read_cora(K_trace{1,1},85,113,35,1);
    
    % Extract data for each point
    for i = allin
        saltxs(i,:,k) = salt(j(i),w(i),:);
        tempxs(i,:,k) = temp(j(i),w(i),:);
        uxs(i,:,k) = u(j(i),w(i),:);
        vxs(i,:,k) = v(j(i),w(i),:);
    end
end

% Save results
save('D:\dataseasonall.mat', 'saltxs', 'tempxs', 'uxs', 'vxs', '-v7.3');