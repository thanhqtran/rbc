clear all;

% Replicate 1 simulation in figure 1.2 in ABCs
% specify parameters
A_tilde = 1;
n = 0.02;
delta = 0.1;
theta = 0.36;
s = 0.2;

% shock parameters
mu_A = 0;
sigma_A = 0.2;

% the function
phi = log(s*A_tilde/(1+n));
kss = (s/(n+delta))^(1/(1-theta));

% results
k = zeros(1,101);       % capital stock
y = zeros(1,101);       % output
gamma = zeros(1,100);   % growth rate of capital stock
epsilon = zeros(1,100); % stochastic process
k(1) = kss;

% time path
for t = 1:100
    % shock is realized
    epsilon(t) = normrnd(mu_A, sigma_A);
    y(t) = A_tilde * exp(epsilon(t)) * (k(t)^theta);
    k(t+1) = ((1-delta)*k(t) + s*y(t)) / (1+n);
    gamma(t) = exp(phi - (1-theta)*log(k(t)) + epsilon(t)) + (1-delta)/(1+n) - 1;
end

% Plot the time path of capital stock
figure(1);
plot(k(1:end-1), 'b', 'LineWidth', 1.5); % Exclude the last element
hold on;
yline(kss, '--r', 'LineWidth', 1.5); % Steady-state level
xlabel('Time');
ylabel('Capital Stock');
title('Time Path of Capital Stock');
legend('Capital Stock', 'Steady State');
hold off;

% Calculate the standard deviation of capital stock and output
k_sd = std(k(1:end-1)); % Exclude the last element
y_sd = std(y(1:end-1)); % Exclude the last element

% Calculate the first-order autocorrelation
k_ac = autocorr(k(1:end-1), 1); % First lag autocorrelation
y_ac = autocorr(y(1:end-1), 1); % First lag autocorrelation

% Display results
disp('Results:');
disp(['Standard Deviation of Capital Stock: ', num2str(k_sd)]);
disp(['Standard Deviation of Output: ', num2str(y_sd)]);
disp(['First-Order Autocorrelation of Capital Stock: ', num2str(k_ac)]);
disp(['First-Order Autocorrelation of Output: ', num2str(y_ac)]);
