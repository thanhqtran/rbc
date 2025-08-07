% kalman_shock_sim.m
%
% This script uses a Kalman smoother to extract both technology (supply)
% and preference (demand) shocks from the data. It then simulates the model
% using this full set of historical shocks and plots the result against the
% actual data to evaluate the model's fit.
%
% 1. It loads data and the model's estimated parameters.
% 2. It runs a Kalman filter and smoother to estimate the unobserved states.
% 3. It generates the model's observable variables directly from the smoothed states.
% 4. It plots the simulated data against the actual data for comparison.


%% 1. Load Data and Model Parameters
% =========================================================================
% Make other functions available
% Ensure state_space_matrices.m is in the MATLAB path.
global lct lyt lht; % These are needed for state_space_matrices
global param;

% Use the final estimated parameters (from main.m or stoch_simul.m)
paramstar = [
    pstar(1);    % thetab
    pstar(2);    % rhotheta
    pstar(3);    % sigmae
    pstar(4);    % rhoa
    pstar(5);    % sigmaa
    pstar(6);    % alpha
    pstar(7)     % gamma
];
alpha = paramstar(6);
rho_theta = paramstar(2);
rho_a = paramstar(4);

fprintf('Model parameters loaded.\n');

%% 2. Kalman Filter and Smoother
% =========================================================================
% Get state-space matrices from the model solution
[Ax, Bx, Cx_all, V] = state_space_matrices(paramstar);
Q = V*V; % Shock variance-covariance matrix (required by filter)

% The observables for the filter are consumption and hours
Cx = [Cx_all(1, :); Cx_all(3, :)] ; % Observation matrix for [c; h]

% The observed data for the filter should be the demeaned series.
observed_data = [lct; lht];

% --- Kalman Filter (Forward Pass) ---
state_filtered = zeros(3, 1);
P_uncond = dlyap(Ax, Bx * Q * Bx'); % Unconditional covariance of states
P_filtered = P_uncond;

states_t_t = zeros(3, T);       % s_t|t
Ps_t_t = zeros(3, 3, T);      % P_t|t
states_t_t1 = zeros(3, T);      % s_t|t-1
Ps_t_t1 = zeros(3, 3, T);     % P_t|t-1

fprintf('Running Kalman filter and smoother...\n');
for t = 1:T
    states_t_t1(:, t) = state_filtered;
    Ps_t_t1(:, :, t) = P_filtered;
    prediction_error = observed_data(:, t) - Cx * state_filtered;
    F = Cx * P_filtered * Cx' ;
    K = P_filtered * Cx' * inv(F);
    state_updated = state_filtered + K * prediction_error;
    states_t_t(:, t) = state_updated;
    P_updated = (eye(3) - K * Cx) * P_filtered;
    Ps_t_t(:, :, t) = P_updated;
    state_filtered = Ax * state_updated;
    P_filtered = Ax * P_updated * Ax' + Bx * Q * Bx';
end

% --- Kalman Smoother (Backward Pass) ---
states_smoothed = zeros(3, T);
states_smoothed(:, T) = states_t_t(:, T);

for t = T-1:-1:1
    J = Ps_t_t(:, :, t) * Ax' * inv(Ps_t_t1(:, :, t+1));
    states_smoothed(:, t) = states_t_t(:, t) + J * (states_smoothed(:, t+1) - states_t_t1(:, t+1));
end
fprintf('Smoothed states calculated.\n');

%% 3. Generate Model Variables from Smoothed States
% =========================================================================
% The previous code extracted shocks and re-simulated the model, which is
% unnecessary and introduces a small error.
%
% The CORRECT method is to use the 'states_smoothed' vector directly, as
% it represents the model's best estimate of the state path. This ensures
% a perfect fit for the observed variables (consumption and hours).

simulated_vars = Cx_all * states_smoothed;
fprintf('Model variables generated from smoothed states.\n');

%% 4. Plot Data vs. Simulation
% =========================================================================
fprintf('Plotting results...\n');

y_sim = simulated_vars(2, :);
c_sim = simulated_vars(1, :);
h_sim = simulated_vars(3, :);

% The data for plotting is the same demeaned data used in the filter.
y_data_plot = lyt;
c_data_plot = lct;
h_data_plot = lht;

figure('Name', 'Model Simulation (Kalman Shocks) vs. Actual Data');

% Define ticks for the x-axis to avoid clutter
tick_interval = 3; % Show a label every 3 quarters

xticks_pos = 1:tick_interval:T;
xticks_labels = date_labels(xticks_pos);

% Plot Output (y)
subplot(3, 1, 1);
plot(1:T, y_data_plot, 'm--', 'LineWidth', 1.5);
hold on;
plot(1:T, y_sim, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:T, zeros(T), 'r:', 'LineWidth', 1.5);
title('GDP per worker: Model vs. Data');
xlabel('Date');
ylabel('Log Deviation');
legend('Data', 'Model Simulation');
grid on;
set(gca, 'XTick', xticks_pos, 'XTickLabel', xticks_labels);
xtickangle(90);

% Plot Consumption (c)
subplot(3, 1, 2);
plot(1:T, c_data_plot, 'r-', 'LineWidth', 1.5);
hold on;
plot(1:T, c_sim, 'k--', 'LineWidth', 1.5);
title('Consumption: Model (Supply & Demand Shocks) vs. Data');
xlabel('Date');
ylabel('Log Deviation');
legend('Data', 'Model Simulation');
grid on;
set(gca, 'XTick', xticks_pos, 'XTickLabel', xticks_labels);
xtickangle(45);

% Plot Hours (h)
subplot(3, 1, 3);
plot(1:T, h_data_plot, 'r-', 'LineWidth', 1.5);
hold on;
plot(1:T, h_sim, 'k--', 'LineWidth', 1.5);
title('Hours: Model (Supply & Demand Shocks) vs. Data');
xlabel('Date');
ylabel('Log Deviation');
legend('Data', 'Model Simulation');
grid on;
set(gca, 'XTick', xticks_pos, 'XTickLabel', xticks_labels);
xtickangle(45);

fprintf('Done.\n');
