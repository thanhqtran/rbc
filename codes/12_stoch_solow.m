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

% results
numit = 1000;   % no. of simulations
timeit = 100;   % no. of periods

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

% Calculate and report the average standard deviation and autocorrelation
k_sd = zeros(1, numit);
y_sd = zeros(1, numit);
k_ac = zeros(1, numit);
y_ac = zeros(1, numit);

for j = 1:numit
    % Exclude the last column of k and y when computing
    k_sd(j) = std(k(j,1:end-1)); % Std dev of capital stock
    y_sd(j) = std(y(j,1:end-1)); % Std dev of output
    
    % First-order autocorrelation using corr
    k_ac(j) = corr(k(j,1:end-2)', k(j,2:end-1)'); % Capital stock
    y_ac(j) = corr(y(j,1:end-2)', y(j,2:end-1)'); % Output
end

% Compute averages across simulations
avg_k_sd = mean(k_sd);
avg_y_sd = mean(y_sd);
avg_k_ac = mean(k_ac);
avg_y_ac = mean(y_ac);

% Display results
disp('Results:');
disp(['Average Standard Deviation of Capital Stock: ', num2str(avg_k_sd)]);
disp(['Average Standard Deviation of Output: ', num2str(avg_y_sd)]);
disp(['Average First-Order Autocorrelation of Capital Stock: ', num2str(avg_k_ac)]);
disp(['Average First-Order Autocorrelation of Output: ', num2str(avg_y_ac)]);
