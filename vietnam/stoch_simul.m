% random_shock_sim.m
%
% This script performs a stochastic simulation of the RBC model based on
% randomly generated technology and preference shocks.
% 1. It loads the model's calibrated and estimated parameters.
% 2. It obtains the state-space representation of the model.
% 3. It generates random series for the technology shock (epsilon_t) and
%    the preference shock (xi_t).
% 4. It simulates the model's response to these shocks.
% 5. It plots the log-deviations from the steady state for key variables
%    and the generated shock series.


%% 1. Load Model Parameters
% =========================================================================
% These parameters must be consistent with the estimation in main.m

% Make other functions available
% Ensure state_space_matrices.m is in the MATLAB path.
global param;

% Set calibrated parameters (from main.m)
% annual_r = 9.113996795/100;
% annual_dep = 0.057428208;
% rss = (1+annual_r)^(1/4)-1;
% delta = 1-(1-annual_dep)^(1/4);
% beta = 1/(rss+1-delta);
% param = [beta delta]; % Store in global for other functions

% Use the final estimated parameters (from main.m or stoch_simul.m)
% NOTE: This replaces the dependency on 'pstar' from main.m
paramstar = [
    pstar(1);    % thetab
    pstar(2);    % rhotheta
    pstar(3);    % sigmae
    pstar(4);    % rhoa
    pstar(5);    % sigmaa
    pstar(6);    % alpha
    pstar(7)     % gamma
];
fprintf('Model parameters loaded.\n');

%% 2. Get State-Space Representation
% =========================================================================
[Ax, Bx, Cx_all, ~] = state_space_matrices(paramstar);
% Ax: State transition matrix
% Bx: Shock impact matrix
% Cx_all: Observation matrix for all variables
fprintf('State-space matrices computed.\n');

%% 3. Generate Random Shocks
% =========================================================================
T = 200; % Number of periods to simulate
sigma_e_shock = paramstar(3); % Standard deviation of the technology shock
sigma_a_shock = paramstar(5); % Standard deviation of the preference shock

% Generate random shocks from a normal distribution
tech_shocks = randn(T, 1) * sigma_e_shock; % epsilon_t
pref_shocks = randn(T, 1) * sigma_a_shock; % xi_t
shocks_sim = [tech_shocks, pref_shocks];

fprintf('Generated %d periods of random technology and preference shocks.\n', T);

%% 4. Perform Stochastic Simulation
% =========================================================================
fprintf('Running stochastic simulation...\n');

% Initialize simulation
% The state vector is [k_t, theta_t, a_t]'
state_sim = zeros(3, 1); % Start from the steady state (zero log-deviation)
simulated_states = zeros(3, T);

% Run simulation loop
for t = 1:T
    % Get the shock for the current period
    current_shock = shocks_sim(t, :)';

    % Evolve state one period forward using the state-space equation
    state_sim = Ax * state_sim + Bx * current_shock;

    % Store the new state
    simulated_states(:, t) = state_sim;
end

% Calculate observable variables from the simulated states
% Cx_all maps states to [c, y, h, i, w, r]'
simulated_vars = Cx_all * simulated_states;
fprintf('Simulation complete.\n');

%% 5. Plot Simulation Results
% =========================================================================
fprintf('Plotting results...\n');

% Extract the relevant series from the simulation results
y_sim = simulated_vars(2, :);
c_sim = simulated_vars(1, :);
h_sim = simulated_vars(3, :);
i_sim = simulated_vars(4, :);

% Create the plot
figure('Name', 'Model Simulation with Random Shocks');

% Plot Output (y_t)
subplot(3, 2, 1);
plot(1:T, y_sim, 'r-', 'LineWidth', 1.5);
title('Output (y_t)');
xlabel('Quarters');
ylabel('Log Deviation from SS');
grid on;
hold on;
plot(1:T, zeros(1, T), 'k--'); % Add steady state line
hold off;

% Plot investment
subplot(3, 2, 2);
plot(1:T, i_sim, 'b-', 'LineWidth', 1.5);
title('Investment (i_t)');
xlabel('Quarters');
ylabel('Log Deviation from SS');
grid on;
hold on;
plot(1:T, zeros(1, T), 'k--'); % Add steady state line
hold off;

% Plot Consumption (c_t)
subplot(3, 2, 3);
plot(1:T, c_sim, 'g-', 'LineWidth', 1.5);
title('Consumption (c_t)');
xlabel('Quarters');
ylabel('Log Deviation from SS');
grid on;
hold on;
plot(1:T, zeros(1, T), 'k--'); % Add steady state line
hold off;

% Plot Hours (h_t)
subplot(3, 2, 4);
plot(1:T, h_sim, 'm-', 'LineWidth', 1.5);
title('Hours (h_t)');
xlabel('Quarters');
ylabel('Log Deviation from SS');
grid on;
hold on;
plot(1:T, zeros(1, T), 'k--'); % Add steady state line
hold off;

% Plot Technology Shocks (epsilon_t)
subplot(3, 2, 5);
plot(1:T, tech_shocks, 'Color', [0 0.4470 0.7410], 'LineWidth', 1);
title('Technology Shocks (\epsilon_t)');
xlabel('Quarters');
ylabel('Shock Size');
grid on;
hold on;
plot(1:T, zeros(1, T), 'k--'); % Add steady state line
hold off;

% Plot Preference Shocks (xi_t)
subplot(3, 2, 6);
plot(1:T, pref_shocks, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1);
title('Preference Shocks (\xi_t)');
xlabel('Quarters');
ylabel('Shock Size');
grid on;
hold on;
plot(1:T, zeros(1, T), 'k--'); % Add steady state line
hold off;

fprintf('Done.\n');
