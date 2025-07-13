% Theoretical moments calculation
% simulate epsilon 1000 times and calculate the sample average

clear all;

%params
alpha = 0.35;
beta = 0.985;
eta = 2;
phi = 1.5;
delta = 0.025;
rhoa = 0.95;

% Step 1: Compute the steady-state of all variables
rss = 1 / beta + delta - 1;
wss = (1-alpha) * ((alpha/rss)^(alpha/(1-alpha)));
nss = ((wss^(1/eta))/( wss/(1-alpha) - delta*((wss/(1-alpha))^(1/alpha) ) ))^(1/(phi/eta + 1));
iss = delta*((alpha/rss)^(1/(1-alpha)))*nss;
css = (wss/(nss^phi))^(1/eta);
kss = ((alpha/rss)^(1/(1-alpha)))*nss;
yss = kss^alpha * nss^(1-alpha);
disp([rss, wss, yss, iss, css, nss, kss]);


%% Define matrices
A = [1; 0; 0; 0; 0; 0];
B = [-(1-delta); -alpha; 0; 0; 1; 0];
C = [0, 0, 0, 0, 0, -delta; ...
     1, 0, -(1-alpha), 0, 0, 0; ...
     0, eta, phi, -1, 0, 0; ...
     -1, 0, 1, 1, 0, 0; ...
     -1, 0, 0, 0, 1, 0; ...
     -yss, css, 0, 0, 0, iss];
D = [0; -1; 0; 0; 0; 0];
F = [0]; G = [0]; H = [0];
J = [0, eta/beta, 0, 0, -rss, 0];
K = [0, -eta/beta, 0, 0, 0, 0];
L = [0]; 
M = [0];
N = [rhoa];

% Solve for P (quadratic equation)
C_inv = inv(C);
a = F - J * C_inv*A;
b = - (J * C_inv * B - G + K * C_inv * A);
c = - K * C_inv * B + H;
DELTA = b^2 - 4 * a * c;
P1 = (-b + sqrt(DELTA))/(2*a);
P2 = (-b - sqrt(DELTA))/(2*a);
P = min(abs(P1), abs(P2));

% Solve for R
R = -C_inv * (A * P + B);

% Solve for Q
k = 1; % Number of columns in Q, based on the dimension of z_t
I_k = eye(k);
% LHS matrix for the system
LHS = kron(N', F - J * C_inv * A) + kron(I_k, J * R + F * P + G - K * C_inv * A);
% RHS vector for the system
RHS = (J * C_inv * D - L) * N + K * C_inv * D - M;
% Solve for vectorized Q
Q_vec = inv(LHS) * vec(RHS);
%Q_vec = RHS/LHS;
% Since Q is 1x1, assign directly
Q = Q_vec; % No need to reshape for scalar values

% Solve for S
S = -C_inv * (A * Q + D);


% Model parameters
rhoa = .95;
sigmae = 0.01;
T = 1000;
burnin = 1000;
n_sim = 1000;

% Storage for moments
num_vars = 8; % y, c, n, w, r, i, k, A
stddev_all = zeros(n_sim, num_vars);
var_all = zeros(n_sim, num_vars);
autocorr_all = zeros(n_sim, num_vars);
mean_all = zeros(n_sim, num_vars);  % To store means


% Simulation loop
for sim = 1:n_sim
    % Initialize variables
    tilde_k = zeros(1,T);
    tilde_y = zeros(1,T);
    tilde_c = zeros(1,T);
    tilde_n = zeros(1,T);
    tilde_w = zeros(1,T);
    tilde_r = zeros(1,T);
    tilde_i = zeros(1,T);
    tilde_A = zeros(1,T);

    % Random shocks
    e = sigmae * randn(1,T);

    % Simulate
    for t = 1:T-1
        tilde_A(t+1) = rhoa * tilde_A(t) + e(t);
        tilde_k(t+1) = P * tilde_k(t) + Q * tilde_A(t);
        tilde_y(t) = R(1) * tilde_k(t) + S(1) * tilde_A(t);
        tilde_c(t) = R(2) * tilde_k(t) + S(2) * tilde_A(t);
        tilde_n(t) = R(3) * tilde_k(t) + S(3) * tilde_A(t);
        tilde_w(t) = R(4) * tilde_k(t) + S(4) * tilde_A(t);
        tilde_r(t) = R(5) * tilde_k(t) + S(5) * tilde_A(t);
        tilde_i(t) = R(6) * tilde_k(t) + S(6) * tilde_A(t);
    end

    % Store data
    variables = {tilde_y, tilde_c, tilde_n, tilde_w, tilde_r, tilde_i, tilde_k, tilde_A};
    data_mat = cell2mat(variables');
    data_mat = data_mat(:, burnin+1:end);  % Remove burn-in
    data_mat = data_mat';                  % T x 8

    % Compute stddev and variance
    mean_all(sim, :) = mean(data_mat);      % Store mean
    stddev_all(sim, :) = std(data_mat);     % Store standard deviation
    var_all(sim, :) = var(data_mat);        % Store variance

    % Compute AR(1) for each variable
    for i = 1:num_vars
        acf = autocorr(data_mat(:,i), 'NumLags', 4);
        autocorr_all(sim, i) = acf(2); % lag-1 autocorrelation
    end
end

% Compute averages over all simulations
avg_mean = mean(mean_all);
avg_stddev = mean(stddev_all);
avg_var = mean(var_all);
avg_autocorr = mean(autocorr_all);

% Optional: Display
disp('Average Means:');
disp(avg_mean);
disp('Average Std. Devs:');
disp(avg_stddev);
disp('Average Variances:');
disp(avg_var);
disp('Average AR(1) Autocorrelations:');
disp(avg_autocorr);
