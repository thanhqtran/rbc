clear all;

% Replicate Figure 1.3 in ABCs
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
numit = 1;   % no. of simulations
timeit = 120;   % no. of periods

k = zeros(numit, timeit+1);       % capital stock
y = zeros(numit, timeit+1);       % output
gamma = zeros(numit, timeit);       % growth rate of capital stock
epsilon = zeros(numit, timeit); % stochastic process

% log-linearized
B = (1 + theta * n - delta * (1-theta))/(1+n);
C = (delta+n)/(1+n);
tildek = zeros(numit, timeit+1);
k_approx = zeros(numit, timeit+1);

% time path
for j = 1:numit
    k(j,1) = kss;
    k_approx(j,1) = kss;
    for t = 1:timeit
    % shock is realized
        epsilon(j,t) = normrnd(mu_A, sigma_A);
        y(j, t) = A_tilde * exp(epsilon(j,t)) * (k(j,t)^theta);
        k(j, t+1) = ((1-delta)*k(j,t) + s*y(j,t)) / (1+n);
        gamma(j,t) = exp(phi - (1-theta)*log(k(j,t)) + epsilon(j,t)) + (1-delta)/(1+n) - 1;
        % log-linearized version
        tildek(j, t+1) = B * (log(k(j,t)) - log(kss)) + C * epsilon(j,t);
        % convert back to level
        k_approx(j, t+1) = kss * exp(tildek(j,t+1));
    end
end

% Plot 3 time path of capital stock
figure(1);
plot(k(1,1:end-1), 'LineWidth', 1);
hold on;
plot(k_approx(1, 1:end-1), 'LineWidth', 1.5, LineStyle='--');
xlabel('Time');
ylabel('Capital Stock');
legend('nonlinear', 'log-linearized','Location', 'Best');
hold off;
