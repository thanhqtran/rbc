clear all;

% Replicate Figure 1.2 in ABCs
% specify parameters
A_tilde = 1;    % steady-state technology
n = 0.02;       % pop growth
delta = 0.1;    % depreciation rate
theta = 0.36;   % capital share (alpha in standard literature)
s = 0.2;        % saving rate for solow

% shock parameters
mu_A = 0;       % mean
sigma_A = 0.2;  % standard deviation

% the function
phi = log(s*A_tilde/(1+n));
kss = (s/(n+delta))^(1/(1-theta));
yss = kss^theta;

% results
numit = 1000;   % no. of simulations
timeit = 10000;   % no. of periods

k = zeros(numit, timeit+1);       % capital stock
y = zeros(numit, timeit+1);       % output
gamma = zeros(numit, timeit);       % growth rate of capital stock
epsilon = zeros(numit, timeit); % stochastic process

% time path
for j = 1:numit
    k(j,1) = kss;
    for t = 1:timeit
    % shock is realized
        epsilon(j,t) = normrnd(mu_A, sigma_A);
        y(j, t) = A_tilde * exp(epsilon(j,t)) * (k(j,t)^theta);
        k(j, t+1) = ((1-delta)*k(j,t) + s*y(j,t)) / (1+n);
        gamma(j,t) = exp(phi - (1-theta)*log(k(j,t)) + epsilon(j,t)) + (1-delta)/(1+n) - 1;
    end
end

% Plot 3 time path of capital stock
figure(1);
for j = 1:3
    plot(k(j,1:end-1));
    hold on;
    yline(kss, '--b');
end
yline(kss, '--b', 'LineWidth', 1.5); % Steady-state level
xlabel('Time');
ylabel('Capital Stock');
hold off;


% Deviations from steady state
k_dev = log(k(:, 1:end-1)) - log(kss);  % Deviations of k from kss
y_dev = log(y(:, 1:end-1)) - log(yss);  % Deviations of y from yss
epsilon_dev = epsilon;        % Deviations of epsilon (steady state is 0)

% Initialize storage for variance, std dev, and autocorrelation
k_var = zeros(1, numit);       % Variance of k
y_var = zeros(1, numit);       % Variance of y
epsilon_var = zeros(1, numit); % Variance of epsilon

% Compute variances, standard deviations, and autocorrelations
for j = 1:numit
    % Variances
    k_var(j) = var(k_dev(j, :));
    y_var(j) = var(y_dev(j, :));
    epsilon_var(j) = var(epsilon_dev(j, :));
end

% Averages across simulations
avg_k_var = mean(k_var);       % Average variance of k
avg_y_var = mean(y_var);       % Average variance of y
avg_epsilon_var = mean(epsilon_var); % Average variance of epsilon

% Display results
disp('Results:');
disp(['Average Variance of Capital Stock (k): ', num2str(avg_k_var)]);
disp(['Average Variance of Output (y): ', num2str(avg_y_var)]);
disp(['Average Variance of Shocks (epsilon): ', num2str(avg_epsilon_var)]);
disp(['Variance of Capital relative to Variance of Tech Shocks: ', num2str(avg_k_var/avg_epsilon_var)]);
disp(['Variance of Output relative to Variance of Tech Shocks: ', num2str(avg_y_var/avg_epsilon_var)]);

