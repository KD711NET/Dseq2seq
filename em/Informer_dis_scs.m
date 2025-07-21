clear all;
addpath 'C:\Matlab\bin\sci'
load('depgrd_zgh_NEW(1).mat');

% Define time range
time1 = datenum(2019,6,1,0,0,0);
time2 = datenum(2019,12,15,0,0,0);
border = 25; % Boundary truncation length

% Create time vectors
tt1 = time1:1/8:time2;
tt2 = tt1(1+border:end-border);

% Get file list from directory
rectotal = dir('D:\NMDIS-CORA2\SCS_original_grid');
filenames = {rectotal.name}';
filenames1 = filenames(3:end); % Remove first two entries (usually '.' and '..')

% Calculate indices based on time range
index1 = (time1-735965)*8+1;
index2 = (time2-735965)*8+1;

filenames2 = filenames1(index1:index2);
filenames3 = filenames2(1+border:end-border);

length_names = size(filenames2,1);
length_names2 = size(filenames3,1);

% Process latitude/longitude data
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
load('depth_cora2.mat')

% Define geographical bounds
lat_min = 8;
lat_max = 24;
lon_min = 108;
lon_max = 120.5;

% Get indices within specified bounds
values = getValuesWithinBounds(lat_2d, lon_2d, Index_2d, lat_min, lat_max, lon_min, lon_max);
mall = values(1:end);
Index_reshaped = squeeze(reshape(Index_2d, [],1));

% Define tidal frequencies (in Hz)
wm2 = 1/(12.420601*3600);  % M2 tidal frequency
ws2 = 1/(12*3600);         % S2 tidal frequency
wk1 = 1/(23.9345*3600);    % K1 tidal frequency
wo1 = 1/(25.8193*3600);    % O1 tidal frequency
wn2 = 1/(12.6583*3600);    % N2 tidal frequency
wq1 = 1/(26.8684*3600);    % Q1 tidal frequency
wp1 = 1/(24.0659*3600);    % P1 tidal frequency
wk2 = 1/(11.9672*3600);    % K2 tidal frequency

load('depgrd_zgh_NEW(1).mat');

% Remove file extensions from filenames
fileNamesWithoutExtension2 = cell(size(filenames2));
for i = 1:numel(filenames2)
    [~, fileName, ~] = fileparts(filenames2{i});
    fileNamesWithoutExtension2{i} = fileName;
end
newfileNames = fileNamesWithoutExtension2(1+border : end-border);
rowNames = string(newfileNames);
lat = 15;

% Create output directories if they don't exist
folder_path = 'C:\code\pythonProject9\pythonProject1\dis_scs_diur\';
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
    disp(['Directory created: ', folder_path]);
else
    disp(['Directory exists: ', folder_path]);
end

% Initialize arrays
saltx = zeros(7081,35,1577);
tempx = zeros(7081,35,1577);

% Process each file
for k = 1:length_names
    k
    
    path = 'D:\NMDIS-CORA2\SCS_original_grid\';
    K_trace = strcat(path,filenames2(k));
    [ssh,u,v,temp,salt,time] = read_cora(K_trace{1,1},85,113,35,1);
    
    q1 = filenames2{k,1}; % Get filename
    q2 = str2num(q1(1:10)); % Extract time from filename
    
    % Reshape and permute data
    salt_permuted = permute(salt, [2, 1, 3]);
    salt_reshaped_Corder = squeeze(reshape(salt_permuted, 1, [], size(salt, 3)));
    temp_permuted = permute(temp, [2, 1, 3]);
    temp_reshaped_Corder = squeeze(reshape(temp_permuted, 1, [], size(salt, 3)));
    
    saltx(:,:,k) = salt_reshaped_Corder(mall,:);
    tempx(:,:,k) = temp_reshaped_Corder(mall,:); 
end  

% Process each location
for i = 1:length(mall)
    i
    m = mall(i);
    saltx2 = redata2(squeeze(saltx(i,:,:)));
    tempx2 = redata2(squeeze(tempx(i,:,:)));
    
    mstr = num2str(m);
    lat1 = depgrdzghNEW1(m,4);
    lat = table2array(lat1);
    
    new_filename = ['C:\code\pythonProject9\pythonProject1\dis_scs_diur\',mstr, '.csv'];

    if isempty(tempx2) 
        % Create empty file if no data
        fileID = fopen(new_filename, 'w');
        fclose(fileID);       
    else
        m1 = length(tempx2(:,1)); % Depth levels
        m2 = length(tempx2(1,:));
        
        if m1 < 3
            % Skip if not enough depth levels
            fileID = fopen(new_filename, 'w');
            fclose(fileID);
        else
            dep0 = depth_cora2(1:m1);
            shen = size(tempx2,1);
            tindex = round(mean(find_thermocline(tempx2', dep0(1:shen))));

            % Calculate diurnal energy dissipation
            dis_diur = p_energy_dis2(tempx2,saltx2,lat,dep0,border);
            data = dis_diur(tindex,:);

            % Format dates for output
            dateTimes = datetime(rowNames, 'InputFormat', 'yyyyMMddHH');
            dateStrings = datestr(dateTimes, 'yyyy/mm/dd HH:MM');
            dateCellArray = cellstr(dateStrings);
      
            % Prepare output data
            outputData = [{'date'},{'OT'};dateCellArray, num2cell(data')];
            charOutputData = cellfun(@char, outputData, 'UniformOutput', false);
            
            % Write to CSV file
            writecell(outputData, new_filename);            
        end
    end
    clear saltx2 tempx2 u_baroclinic v_baroclinic dis_diur 
end