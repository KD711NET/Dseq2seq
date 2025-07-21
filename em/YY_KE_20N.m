clear all;
addpath('C:\\Matlab\\toolbox\\colorbar\\colorbar')

addpath 'C:\Matlab\bin\sci'
load precip3_16lev.txt
colormap1=precip3_16lev;

m=load("C:\code\pythonProject9\pythonProject1\rinfo\all_data_Dseq_u_diur_lz20N.mat");
mv=load("C:\code\pythonProject9\pythonProject1\rinfo\all_data_v_diur_lz20N.mat");

m1=load("C:\code\pythonProject9\pythonProject1\rdseq\all_data_Dseq_u_diur_lz20N.mat");
m1v=load("C:\code\pythonProject9\pythonProject1\rdseq\all_data_Dseq_v_diur_20N.mat");

md=load("C:\code\pythonProject9\DLinear\DL\u_diur_lz20N.mat");
mdv=load("C:\code\pythonProject9\DLinear\DL\v_diur_lz20N.mat");

mlen=length(m.data_dict);

standu=load('C:\code\pythonProject4\stand_20_diur_u.mat');

standv=load('C:\code\pythonProject4\stand_20_diur_v.mat');


% Main processing section

u_total={};
load('depth_cora2.mat')
u_seq_all=m1.data_dict;
u_info_all=m.data_dict;
u_d_all=md.data_dict;

v_seq_all=m1v.data_dict;
v_info_all=mv.data_dict;
v_d_all=mdv.data_dict;


% Initialize arrays for kinetic energy calculations
KE_info_all=zeros(mlen,1272);
KE_seq_all=zeros(mlen,1272);
KE_true_all=zeros(mlen,1272);
KE_d_all=zeros(mlen,1272);
KE_ha_all=zeros(mlen,1272);

% Initialize arrays for kinetic energy bias calculations
KEbias_d=zeros(mlen,1272);
KEbias_ha=zeros(mlen,1272);
KEbias_info=zeros(mlen,1272);
KEbias_seq=zeros(mlen,1272);

% Initialize 35x55 matrices for storing results
KE_info_all2=zeros(35,55);
KE_seq_all2=zeros(35,55);
KE_d_all2=zeros(35,55);
KE_ha_all2=zeros(35,55);


% Main processing loop for each of the 55 cases
for iii=1:55
    iii  % Display current iteration number

    % Load u-component predictions and true values
    u_seq=u_seq_all{1,iii}.preds;
    u_info=u_info_all{1,iii}.preds;
    u_true=u_info_all{1,iii}.trues;
    u_ha=u_info_all{1,iii}.prehs;
    u_d=u_d_all{1,iii}.preds;

    % Load v-component predictions and true values
    v_seq=v_seq_all{1,iii}.preds;
    v_info=v_info_all{1,iii}.preds;
    v_true=v_info_all{1,iii}.trues;
    v_ha=v_info_all{1,iii}.prehs;
    v_d=v_d_all{1,iii}.preds;

    % Get standardization parameters for u and v components
    mean_u_diur=standu.stand_dict{1,iii}.means;
    std_u_diur=standu.stand_dict{1,iii}.stds;
    mean_v_diur=standv.stand_dict{1,iii}.means;
    std_v_diur=standv.stand_dict{1,iii}.stds;

    % Inverse standardization of u-component data
    u_ha2=inverse_standardize(u_ha,mean_u_diur, std_u_diur);
    u_info2=inverse_standardize(u_info,mean_u_diur, std_u_diur);
    u_seq2=inverse_standardize(u_seq,mean_u_diur, std_u_diur);
    u_true2=inverse_standardize(u_true,mean_u_diur, std_u_diur);
    u_d2=inverse_standardize(u_d,mean_u_diur, std_u_diur);

    % Inverse standardization of v-component data
    v_ha2=inverse_standardize(v_ha,mean_v_diur, std_v_diur);
    v_info2=inverse_standardize(v_info,mean_v_diur, std_v_diur);
    v_seq2=inverse_standardize(v_seq,mean_v_diur, std_v_diur);
    v_true2=inverse_standardize(v_true,mean_v_diur, std_v_diur);
    v_d2=inverse_standardize(v_d,mean_v_diur, std_v_diur);

    % Get dimensions of the data
    m1 = length(u_info2(:,1)); % Depth index
    m2 = length(u_info2(1,:));
    depth=depth_cora2(1:m1-1); % Depth values
    bottom=depth(end);
    top=depth(1);

    % Calculate kinetic energy for different model predictions
    [KE_diur_ha,K_e_diur_ha]=k_energy2(u_ha2,v_ha2,depth);
    [KE_diur_info,K_e_diur_info]=k_energy2(u_info2,v_info2,depth);
    [KE_diur_seq,K_e_diur_seq]=k_energy2(u_seq2,v_seq2,depth);
    [KE_diur_d,K_e_diur_d]=k_energy2(u_d2,v_d2,depth);
    [KE_diur_true,K_e_diur_true]=k_energy2(u_true2,v_true2,depth);

    % Store kinetic energy results
    KE_ha_all(iii,:)=KE_diur_ha;
    KE_info_all(iii,:)=KE_diur_info;
    KE_seq_all(iii,:)=KE_diur_seq;
    KE_true_all(iii,:)=KE_diur_true;
    KE_d_all(iii,:)=KE_diur_d;

    % Calculate and store mean absolute errors (padded to 35 rows)
    KE_ha_all2(:,iii)=padMatrixTo35Rows(mean(abs(K_e_diur_ha-K_e_diur_true),2));
    KE_info_all2(:,iii)=padMatrixTo35Rows(mean(abs(K_e_diur_info-K_e_diur_true),2));
    KE_seq_all2(:,iii)=padMatrixTo35Rows(mean(abs(K_e_diur_seq-K_e_diur_true),2));
    KE_d_all2(:,iii)=padMatrixTo35Rows(mean(abs(K_e_diur_d-K_e_diur_true),2));

    % Calculate and store absolute biases
    KEbias_info(iii,:)=abs(KE_diur_info-KE_diur_true);
    KEbias_seq(iii,:)=abs(KE_diur_seq-KE_diur_true);
    KEbias_ha(iii,:)=abs(KE_diur_ha-KE_diur_true);
    KEbias_d(iii,:)=abs(KE_diur_d-KE_diur_true);
end

% Define tidal frequencies (in Hz)
wm2 = 1/(12.420601*3600);  % M2 tidal frequency
ws2 = 1/(12*3600);         % S2 tidal frequency
wk1 = 1/(23.9345*3600);    % K1 tidal frequency
wo1 = 1/(25.8193*3600);    % O1 tidal frequency
wn2 = 1/(12.6583*3600);    % N2 tidal frequency
wq1 = 1/(26.8684*3600);    % Q1 tidal frequency
wp1 = 1/(24.0659*3600);    % P1 tidal frequency
wk2 = 1/(11.9672*3600);    % K2 tidal frequency

% Load bathymetry data
load('depgrd_zgh_NEW(1).mat');

% Extract longitude/latitude and other data from the loaded file
a8 = depgrdzghNEW1(:,3:4);  % Contains longitude and latitude
a9 = depgrdzghNEW1(:,1:2);  % Contains other data
a88 = table2array(a8);      % Convert to array
a99 = table2array(a9);      % Convert to array

% Define time range for analysis
time1 = datenum(2019,6,1,0,0,0);  % Start date
time2 = datenum(2019,12,15,0,0,0); % End date

% Reshape data into 2D grids
lon_2d = reshape(a88(:,1),113,85);  % Longitude grid
lat_2d = reshape(a88(:,2),113,85);  % Latitude grid
Index_2d = reshape(1:9605,113,85);  % Index grid

% Get grid dimensions
rows = size(Index_2d, 1);
cols = size(Index_2d, 2);

% Load depth data
load('depth_cora2.mat')

% Define analysis region boundaries
lat_min = 20;  % Minimum latitude
lat_max = 20;  % Maximum latitude
lon_min = 112; % Minimum longitude
lon_max = 120; % Maximum longitude

% Get values within specified bounds
values = getValuesWithinBounds(lat_2d, lon_2d, lon_2d, lat_min, lat_max, lon_min, lon_max);
depp = depth_cora2;
lon = values;

% Create meshgrid for plotting
[X,Y] = meshgrid(lon,depp);

%% Plot kinetic energy time series
% Calculate mean kinetic energy across all cases
PEdiur_ha = mean(KE_ha_all(:, 1:1272),1);    % HA model
PEdiur_info = mean(KE_info_all(:, 1:1272),1); % Informer model
PEdiur_seq = mean(KE_seq_all(:, 1:1272),1);   % Dseq2seq model
PEdiur_d = mean(KE_d_all(:, 1:1272),1);       % DLinear model
PEdiur_true = mean(KE_true_all(:, 1:1272),1); % True values

% Define time vectors
time1 = datenum(2019,6,1,0,0,0);
time2 = datenum(2019,12,15,0,0,0);
tt1 = time1:1/8:time2;            % Full time vector with 1/8 day resolution
border = 25;                      % Number of points to trim from boundaries
tt2 = tt1(1+border:end-border);   % Trimmed time vector

% Create first subplot for time series
subplot('position',[0.05 0.84 0.27 0.1]);
tim = tt2(1:1272);
plot(tim, PEdiur_ha/1000, 'g', 'LineWidth', 2, 'DisplayName', 'HA prediction');
hold on
plot(tim, PEdiur_seq/1000, 'b-', 'LineWidth', 2, 'DisplayName', 'Dseq2seq prediction');
hold on
plot(tim, PEdiur_true/1000, 'r--', 'LineWidth', 2, 'DisplayName', 'true value');
title('KE','FontSize', 14, 'FontWeight', 'bold');
ylabel('(KJ)');
datetick('x', 'mm/dd', 'keepticks', 'keeplimits'); % Format x-axis as dates

% Add legend and labels
legend('show','FontSize', 10);
xlabel('date', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Set plot appearance
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');

%% Plot MAE by longitude
% Calculate mean absolute error for each model
info_pos = mean(KEbias_info,2);  % Informer model
seq_pos = mean(KEbias_seq,2);    % Dseq2seq model
ha_pos = mean(KEbias_ha,2);      % HA model
d_pos = mean(KEbias_d,2);        % DLinear model

% Create second subplot for MAE by longitude
subplot('position',[0.05 0.54 0.27 0.1]);
plot(lon, d_pos/1000, 'g', 'LineWidth', 1.5);   % DLinear
hold on
plot(lon, info_pos/1000, 'b', 'LineWidth', 1.5); % Informer
hold on;
plot(lon, seq_pos/1000, 'r', 'LineWidth', 1.5);  % Dseq2seq

% Add legend and labels
legend({'DLinear','Informer', 'Dseq2seq'}, 'Location', 'best', 'FontSize', 10);
ylabel('MAE(KJ)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Set plot appearance
set(gca, 'FontSize', 10);
set(gca, 'LineWidth', 1.5);
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');

%% Calculate and display global MAE values
info_pos1 = mean(KEbias_info/1000,'all')  % Informer global MAE
seq_pos1 = mean(KEbias_seq/1000,'all')    % Dseq2seq global MAE
ha_pos1 = mean(KEbias_ha/1000,'all')      % HA global MAE
d_pos1 = mean(KEbias_d/1000,'all')        % DLinear global MAE

% Create bar plot of global MAE values
values = [ha_pos1, d_pos1, info_pos1, seq_pos1];  % MAE values
labels = {'HA', 'DLinear', 'Informer', 'Dseq2seq'}; % Model labels

% Create third subplot for MAE comparison
subplot('position',[0.355 0.52 0.2 0.43]);
bar(values);

% Customize bar plot
set(gca, 'XTickLabel', labels);
set(gca, 'LineWidth', 1.5);
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');
grid on;
set(gca, 'LineWidth', 1.5);
ylabel('MAE(KJ)');
title('MAE','FontSize', 14, 'FontWeight', 'bold');

% Add value labels on bars
for i = 1:length(values)
    text(i, values(i), sprintf('%.3f', values(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end



%% Calculate absolute differences between model predictions and true values
for ii = 1:55
    % Extract kinetic energy values for each model
    KEdiur_ha = KE_ha_all(ii, 1:1272);
    KEdiur_info = KE_info_all(ii, 1:1272);
    KEdiur_seq = KE_seq_all(ii, 1:1272);
    KEdiur_d = KE_d_all(ii, 1:1272);
    KEdiur_true = KE_true_all(ii, 1:1272);

    % Calculate absolute differences from true values
    KEdiur_ha_diff1(ii,:) = abs(KEdiur_ha - KEdiur_true);
    KEdiur_info_diff1(ii,:) = abs(KEdiur_info - KEdiur_true);
    KEdiur_seq_diff1(ii,:) = abs(KEdiur_seq - KEdiur_true);
    KEdiur_d_diff1(ii,:) = abs(KEdiur_d - KEdiur_true);
end

% Calculate mean differences across all cases
KEdiur_ha_diff = mean(KEdiur_ha_diff1,1);
KEdiur_info_diff = mean(KEdiur_info_diff1,1);
KEdiur_seq_diff = mean(KEdiur_seq_diff1,1);
KEdiur_d_diff = mean(KEdiur_d_diff1,1);

%% Reshape data for hourly analysis
numRows = 24; % Number of hours in a day
numCols = length(PEdiur_ha) / numRows;

% Reshape arrays into 24-hour bins
KEdiur_ha_diff_reshaped = reshape(KEdiur_ha_diff, numRows, numCols)';
KEdiur_info_diff_reshaped = reshape(KEdiur_info_diff, numRows, numCols)';
KEdiur_seq_diff_reshaped = reshape(KEdiur_seq_diff, numRows, numCols)';
KEdiur_d_diff_reshaped = reshape(KEdiur_d_diff, numRows, numCols)';

% Calculate hourly mean differences
KEdiur_ha_diff_mean = mean(KEdiur_ha_diff_reshaped);
KEdiur_info_diff_mean = mean(KEdiur_info_diff_reshaped);
KEdiur_seq_diff_mean = mean(KEdiur_seq_diff_reshaped);
KEdiur_d_diff_mean = mean(KEdiur_d_diff_reshaped);

%% Plot hourly MAE comparison
subplot('position',[0.05 0.69 0.27 0.1]);
tim = 1:24;

% Plot each model's MAE
plot(tim, KEdiur_d_diff_mean/1000, 'g', 'LineWidth', 2, 'DisplayName', 'DLinear');
hold on
plot(tim, KEdiur_info_diff_mean/1000, 'b', 'LineWidth', 2, 'DisplayName', 'Informer');
hold on;
plot(tim, KEdiur_seq_diff_mean/1000, 'r', 'LineWidth', 2, 'DisplayName', 'Dseq2seq');
hold off;

% Set axis labels and appearance
xlabel('Step Length', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('MAE(KJ)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([1 24]);
ylim auto;

% Add legend and grid
legend('Location', 'northeast', 'FontSize', 10);
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');
grid on;
set(gca, 'LineWidth', 1.5);

%% Plot depth-longitude MAE distributions
% First subplot: HA model
subplot('position',[0.06 0.27 0.23 0.18]);
KE_ha_all2(KE_ha_all2>=3) = 3; % Cap values at 3 for visualization

contourf(X, -Y, KE_ha_all2, 'edgecolor', 'none');
hold on;
plot20N; % Custom function to plot 20N reference
colormap(colormap1);
xticks(linspace(min(X(:)), max(X(:)), 5)); 
cb2 = colorbar;
ylabel(cb2, '(J)', 'FontSize', 10, 'FontWeight', 'bold');
title('KE MAE HA','FontSize', 14, 'FontWeight', 'bold');
ylabel('Depth(m)');
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');
grid on;
set(gca, 'LineWidth', 1.5);

% Second subplot: Informer model
KE_info_all2(KE_info_all2>=1) = 1;
subplot('position',[0.34 0.27 0.23 0.18]);

contourf(X, -Y, KE_info_all2, 'edgecolor', 'none');
hold on;
plot20N;
colormap(colormap1);
cb2 = colorbar;
ylabel(cb2, '(J)', 'FontSize', 10, 'FontWeight', 'bold');
title('KE MAE Informer','FontSize', 14, 'FontWeight', 'bold');
xticks(linspace(min(X(:)), max(X(:)), 5)); 
ylabel('Depth(m)');
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');
grid on;
set(gca, 'LineWidth', 1.5);

% Third subplot: Dseq2seq model
KE_seq_all2(KE_seq_all2>=1) = 1;
subplot('position',[0.34 0.03 0.23 0.18]);
contourf(X, -Y, KE_seq_all2, 'edgecolor', 'none');
hold on;
plot20N;
colormap(colormap1);
cb2 = colorbar;
ylabel(cb2, '(J)', 'FontSize', 10, 'FontWeight', 'bold');
title('KE MAE Dseq2seq','FontSize', 14, 'FontWeight', 'bold');
ylabel('Depth(m)');
xticks(linspace(min(X(:)), max(X(:)), 5)); 
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');
grid on;
set(gca, 'LineWidth', 1.5);

% Fourth subplot: DLinear model
KE_d_all2(KE_d_all2>=1) = 1;
subplot('position',[0.06 0.03 0.23 0.18]);
contourf(X, -Y, KE_d_all2, 'edgecolor', 'none');
hold on;
plot20N;
colormap(colormap1);
cb2 = colorbar;
ylabel(cb2, '(J)', 'FontSize', 10, 'FontWeight', 'bold');
title('KE MAE DLinear','FontSize', 14, 'FontWeight', 'bold');
ylabel('Depth(m)');
xticks(linspace(min(X(:)), max(X(:)), 5)); 
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on', 'GridLineStyle', '--');
grid on;
set(gca, 'LineWidth', 1.5);


