% Uhlig and Christiano (2002)'s method of undermined coefficients
clear all;

%=========================================================
% ======= PARAMETERS =====================================
%% normally, we calibrate beta, delta and estimate the rest
beta = 0.985;          %intertemporal discount
delta = 0.025;         %depreciation rate
alpha = 0.35;          %capital share
eta = 2;               %Coefficient of Relative Risk Aversion, intertemporal elasticity of substitution is 1/eta
phi = 1.5;             %Inverse Frisch Elasticity of Labor Supply, Frisch elasticity is 1/phi
rhoa = 0.95;           %persistence of shocks

% Step 1: Compute the steady-state of all variables
%==================================================
rss = 1 / beta + delta - 1;
wss = (1-alpha) * ((alpha/rss)^(alpha/(1-alpha)));
nss = ((wss^(1/eta))/( wss/(1-alpha) - delta*((wss/(1-alpha))^(1/alpha) ) ))^(1/(phi/eta + 1));
iss = delta*((alpha/rss)^(1/(1-alpha)))*nss;
css = (wss/(nss^phi))^(1/eta);
kss = ((alpha/rss)^(1/(1-alpha)))*nss;
yss = kss^alpha * nss^(1-alpha);
disp([rss, wss, yss, iss, css, nss, kss]);

%% The model in linearized form
%==================================================
%% x_t = [ \tilde{k}_t ]
%% y_t = \begin{bmatrix}
%%		\tilde{y}_t \\
%%		\tilde{c}_t \\
%%		\tilde{n}_t \\
%%		\tilde{w}_t \\
%%		\tilde{r}_t \\
%%		\tilde{i}_t
%%	\end{bmatrix}
%% The mapping matrices
%% \begin{align}
%%	\label{matrix_1} 0 &= A x_t + B x_{t-1} + C y_t + D z_t, \\
%%	\label{matrix_2} 0 &= E_t [ F x_{t+1} + G x_t + H x_{t-1} + J y_{t+1} + K y_t + L z_{t+1} + M z_t ]
%% \label{matrix_3} z_{t+1} &= N z_t + \epsilon_{t+1}, \ E_t (\epsilon_{t+1}) = 0.
%% \end{align}

%%=================================================
% Define matrices and solve the model (linearized)
%%=================================================

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

% Solving auxiliary matrices
%==================================================
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

% Display results
disp('P = '); disp(P);
disp('R = '); disp(R);
disp('Q = '); disp(Q);
disp('S = '); disp(S);

%%=============================
% Impulse Response Functions
%%=============================

% A one-time shock with the maginitude of 1 standard deviation
T = 80; %time horizon

tilde_k = zeros(1,T);
tilde_y = zeros(1,T);
tilde_c = zeros(1,T);
tilde_n = zeros(1,T);
tilde_w = zeros(1,T);
tilde_r = zeros(1,T);
tilde_i = zeros(1,T);
tilde_A = zeros(1,T);

%% IRF
% one-time shock
tilde_A(1) = 0.01; %standard error of shock

for t = 1:T-1
    tilde_A(t+1) = rhoa * tilde_A(t);
    tilde_k(t+1) = P * tilde_k(t) + Q * tilde_A(t);
    tilde_y(t) = R(1) * tilde_k(t) + S(1) * tilde_A(t);
    tilde_c(t) = R(2) * tilde_k(t) + S(2) * tilde_A(t);
    tilde_n(t) = R(3) * tilde_k(t) + S(3) * tilde_A(t);
    tilde_w(t) = R(4) * tilde_k(t) + S(4) * tilde_A(t);
    tilde_r(t) = R(5) * tilde_k(t) + S(5) * tilde_A(t);
    tilde_i(t) = R(6) * tilde_k(t) + S(6) * tilde_A(t);
end

variables = {tilde_y,tilde_c,tilde_n,tilde_w,tilde_r,tilde_i, tilde_k, tilde_A};
labels = {'y', 'c', 'n', 'w', 'r', 'i', 'k', 'A'};
horizon = s+1:T-1; %choose the horizon after the shock

figure;
for i = 1:length(variables)
    subplot(3,3,i);
    plot(horizon, variables{i}(horizon), 'b', 'LineWidth', 1); %
    hold on;
    title(labels{i});
    yline(0, 'Color','r', 'LineWidth', 1); % Zero line in red
    grid on;
end

%%===========================
%% Stochastic simulation
%%===========================
sigmae = 0.01;

T = 1000; %nbumber of periods

tilde_k = zeros(1,T);
tilde_y = zeros(1,T);
tilde_c = zeros(1,T);
tilde_n = zeros(1,T);
tilde_w = zeros(1,T);
tilde_r = zeros(1,T);
tilde_i = zeros(1,T);
tilde_A = zeros(1,T);

% a series of random shocks
randn('seed',666);
e = sigmae*randn(1,T);
% the time series
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

variables = {tilde_y,tilde_c,tilde_n,tilde_w,tilde_r,tilde_i, tilde_k, tilde_A, e};
labels = {'y', 'c', 'n', 'w', 'r', 'i', 'k', 'A', 'e'};
horizon = s:200; %choose the horizon after the shock

figure;
for i = 1:length(variables)
    subplot(3,3,i);
    plot(horizon, variables{i}(horizon), 'b', 'LineWidth', 1); %
    hold on;
    title(labels{i});
    yline(0, 'Color','r', 'LineWidth', 1); % Zero line in red
    grid on;
end
