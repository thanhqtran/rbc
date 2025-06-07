addpath /Applications/Dynare/6.2-arm64/matlab

%% =====================================================
%% Run the recalibrate model
%% =====================================================
dynare rbc_log.mod

%% =====================================================
%% Update params 
%% Run the recalibrated model
%% =====================================================

dynare rbc_log_recalibrated.mod

%% =====================================================
%% extract the parameters for state-space representation
%% =====================================================
% variable order: Y I C L W R K A
% state vars: 7, 8

p_Y = 1;
p_I = 2;
p_C = 3;
p_L = 4;
p_W = 5;
p_R = 6;
p_K = 7;
p_A = 8;

% create matrices for the state-space representation
% S(t) = A*S(t-1) + B*e(t)
% X(t) = C*S(t-1) + D*e(t)

A = [   oo_.dr.ghx(oo_.dr.inv_order_var(p_K),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_A),:)
    ];
B = [   oo_.dr.ghu(oo_.dr.inv_order_var(p_K),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_A),:)
    ];
C = [   oo_.dr.ghx(oo_.dr.inv_order_var(p_Y),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_I),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_C),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_R),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_W),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_L),:);
    ];
D = [   oo_.dr.ghu(oo_.dr.inv_order_var(p_Y),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_I),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_C),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_R),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_W),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_L),:);
    ];

%% =====================================================
%% ======== IRF=========================================
%% =====================================================

% compute the impulse reponses by hand
sigma_e = 0.01;
H = 20;
Sirf = zeros(2,H);
Xirf = zeros(6,H);

Sirf(:,1) = B*sigma_e;
Xirf(:,1) = D*sigma_e;

for j = 2:H
    Sirf(:, j) = A*Sirf(:, j-1);
    Xirf(:, j) = C*Sirf(:, j-1);
end

%% Plotting
% Time axis
t = 1:H;

% Variable names for display (including K and A)
% variable order: Y I C L W R K A
var_names = {'Y', 'I', 'C', 'L', 'W', 'R', 'K', 'A'};

% Combine Xirf and Sirf into one matrix for plotting
IRFs_all = [Xirf; Sirf];

% Plot all IRFs
figure;
for i = 1:8
    subplot(3,3,i);
    plot(t, IRFs_all(i,:), 'LineWidth', 1, 'Color','black');
    hold on;
    yline(0, '-k', 'LineWidth', 1, 'Color','r'); % Add horizontal zero line
    hold off;
    title([var_names{i}]);
    %xlabel('Periods');
    %ylabel('Response');
    %grid on;
end

sgtitle('Impulse Response Functions (Manual State-Space IRFs)');

%% =====================================================
%% ========= RANDOM Simulation =========================
%% =====================================================

% compute a simulation. First draw shocks
randn('seed', 666);
T = 200;
e = sigma_e * randn(1,T);

Ssim = zeros(2,T);
Xsim = zeros(6,T);

% assume initial state is SS
Ssim(:,1) = B*e(1,1);
for j = 2:T
    Ssim(:,j) = A*Ssim(:,j-1) + B*e(1,j);
    Xsim(:,j) = C*Ssim(:,j-1) + D*e(1,j);
end

% Time axis
t = 1:T;
% Combine Xirf and Sirf into one matrix for plotting
simulated = [Xsim; Ssim];

% Plot all simulated
figure;
for i = 1:8
    subplot(3,3,i);
    plot(t, simulated(i,:), 'LineWidth', 1, 'Color','black');
    title([var_names{i}]);
end


%% =====================================================
%% ====== CALIBRATION and DATASET ======================
%% =====================================================

%% Load in real data
dat = readtable("us_test.csv");  %the data is for illustrative purposes only

% Extract only numeric columns
cols = dat(:, vartype('numeric'));

% Compute standard deviations
data_std = varfun(@std, cols);

% Convert to array
std_array = table2array(data_std)';
var_names = data_std.Properties.VariableNames';

% Reference variable (first column is Y)
ref_std = std_array(1);  

% Compute relative standard deviation
relative_std_array = std_array / ref_std;

% Combine into table
data_stats = table(var_names, std_array, relative_std_array, ...
    'VariableNames', {'Variable', 'StdDev', 'RelStdDev'});

disp(data_stats);

%% =====================================================
%% ====== EYEBALL SIMULATION ======================
%% =====================================================

% Multiply scalar std by tfp vector
e = transpose(dat.eA);

% time
T = length(dat.DATE);
Ssim = zeros(2,T);
Xsim = zeros(6,T);

% assume initial state is SS
Ssim(:,1) = B*e(1,1);
for j = 2:T
    Ssim(:,j) = A*Ssim(:,j-1) + B*e(1,j);
    Xsim(:,j) = C*Ssim(:,j-1) + D*e(1,j);
end

% Time axis
t = 1:T;

% Variable names for display (including K and A)
var_names = {'Y', 'I', 'C', 'L'};

% Combine Xirf and Sirf into one matrix for plotting
simulated = [Xsim; Ssim];

% Plot model's simulated against real data
% Table variable names to match each var_name
data_vars = {'y', 'i', 'c', 'l'};

figure;
for i = 1:4
    subplot(2,2,i);
    plot(t, simulated(i,:), 'k-', 'LineWidth', 1); % model
    hold on;
    plot(t, dat{:, data_vars{i}}, 'r-', 'LineWidth', 1); % data
    hold off;
    title(var_names{i});
    xlabel('Time');
    ylabel('Deviation');
    grid on;
    legend('Model', 'Data');
end

%% =====================================================
%% ====== MODEL BASIC STATISTICS =======================
%% =====================================================

% Variable names (must match your .mod file variables)
vars = {'Y', 'I', 'C', 'L', 'W', 'R', 'K', 'A'};

% Convert Dynare's char matrix to cell array
all_vars = cellstr(M_.endo_names);
var_idx = cellfun(@(v) find(strcmp(all_vars, v)), vars);

% Standard deviations
std_devs = sqrt(diag(oo_.var(var_idx, var_idx)));
relative_std = std_devs / std_devs(strcmp(vars, 'Y'));

% First-order autocorrelations
autocorr1 = oo_.autocorr{1};  % Full autocorr matrix
autocorr1_vec = diag(autocorr1(var_idx, var_idx));

% Contemporaneous correlations with Y
corr_matrix = oo_.contemporaneous_correlation(var_idx, var_idx);
corr_with_Y = corr_matrix(:, strcmp(vars, 'Y'));

% Create and display table
T = table(string(vars'), std_devs, relative_std, autocorr1_vec, corr_with_Y, ...
    'VariableNames', {'Variable', 'Std. Dev', 'Relative Std. Dev', 'First-order AR', 'Contemporaneous corr with output'});
disp(T);
