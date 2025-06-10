%% EYEBALL SIMULATION
% Multiply scalar std by tfp vector
shock_dat = readtable("rbc_log_decomp/Output/rbc_log_decomp_shock_decomposition.xls");
dat = readtable("usdat.csv")
e = transpose(shock_dat.SmootVar);

% time
T = length(shock_dat.SmootVar);
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
data_vars = {'Y', 'I', 'C', 'L'};
first_obs = 8;
% Extract time axis starting from 8th row
% Convert 'YYYYQX' strings to datetime (only once at beginning of your script)
dat.DATES = datetime(dat.DATES, 'InputFormat', 'yyyy''Q''Q', 'Format', 'yyyyQQQ');
t_plot = dat.DATES(first_obs:end);

figure;
for i = 1:4
    subplot(2,2,i);
    plot(t_plot, simulated(i,:), 'k-', 'LineWidth', 1); % model (already aligned)
    hold on;
    plot(t_plot, dat{first_obs:end, data_vars{i}}, 'r-', 'LineWidth', 1); % skip first 7 obs
    hold off;
    title(var_names{i});
    xlabel('Time');
    ylabel('Deviation');
    grid on;
    legend('Model', 'Data');
end
