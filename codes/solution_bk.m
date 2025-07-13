%% Blanchard-Kahn (1980) solution method
%% requirements
%% qzschur.m
%% the schur function decomposes A,B into S,T,Q,Z
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

% Define fundamental matrices
% === Fundamental matrices B and A ===
%% FORM: \begin{align} \label{bk_1}
%	B \begin{bmatrix}
%		x_{t+1} \\
%		E_t y_{t+1}
%	\end{bmatrix}
%	=
%	A
%	\begin{bmatrix}
%		x_t \\ y_t
%	\end{bmatrix}
%	+ G \epsilon_t
% \end{align}
%%=========================
%% specify the matrices

B = [
    -kss, 0, yss, 0, 0;
    0, 1, 0, 0, 0;
    0, 1, (1-alpha)/(1+phi)-1, 0, 0;
    0, 0, 1, 0, 0;
    0, 0, 0, -eta/beta, rss
];

A = [
    -(1-delta)*kss, 0, 0, css, 0;
    0, rhoa, 0, 0, 0;
    -alpha, 0, 0, (1-alpha)*eta/(1+phi), 0;
    1, 0, 0, 0, 1;
    0, 0, 0, -eta/beta, 0 
];
G = [0; 1; 0; 0; 0];



%% PARTITIONING
% == Step 3: Partitioning Information ==
% We need to identify the number of predetermined (state) variables (nx)
% and non-predetermined (jump) variables (ny).
% State variables (x_t): k_t, a_t
% Jump variables (y_t): y_t, c_t, r_t
nx = 3; % Number of non-expectation variables
ny = 2; % Number of expectation variables

%% == Step 4: Generalized Schur (QZ) Decomposition ==
% This decomposition provides A = Q*S*Z' and B = Q*T*Z', where the
% eigenvalues are sorted by increasing magnitude.
[Q, Z, S, T] = qzschur(A, B);

% --- Blanchard-Kahn Condition Check ---
% The number of unstable eigenvalues (abs > 1) must equal the number of
% non-predetermined (jump) variables for a unique, stable solution to exist.
eigenvalues = eig(A, B);
num_unstable = sum(abs(eigenvalues) > 1);
disp(eigenvalues);

fprintf('--- Blanchard-Kahn Conditions ---\n');
fprintf('Number of unstable eigenvalues: %d\n', num_unstable);
fprintf('Number of jump variables (ny): %d\n', nx);

if num_unstable ~= ny
    error('Blanchard-Kahn conditions are NOT satisfied. A unique stable solution does not exist.');
else
    fprintf('Blanchard-Kahn conditions are satisfied.\n');
end
fprintf('---------------------------------\n\n');


%% == Step 5: Partition Decomposed and Original Matrices ==
% We partition the matrices according to the number of stable (nx) and
% unstable (ny) roots.

% Transposes are used frequently in the solution formula
Q_prime = Q';
Z_prime = Z';

% Partition Z'
Z11p = Z_prime(1:nx, 1:nx);
Z12p = Z_prime(1:nx, nx+1:end);
Z21p = Z_prime(nx+1:end, 1:nx);
Z22p = Z_prime(nx+1:end, nx+1:end);

% Partition S and T (from the stable/unstable blocks)
S11 = S(1:nx, 1:nx);
S22 = S(nx+1:end, nx+1:end);

% Partition Q'
Q21p = Q_prime(nx+1:end, 1:nx);
Q22p = Q_prime(nx+1:end, nx+1:end);

% Partition original A, B, and G matrices
A11 = A(1:nx, 1:nx);
A12 = A(1:nx, nx+1:end);
B11 = B(1:nx, 1:nx);
B12 = B(1:nx, nx+1:end);
G1 = G(1:nx, :);
G2 = G(nx+1:end, :);


%% == Step 6: Solve for Policy Function Matrices (N, L, C, D) ==
% These matrices define the complete solution to the model:
% y_t = -N*x_t - L*epsilon_t
% x_{t+1} = C*x_t + D*epsilon_t

% Solve for N, which maps states (x_t) to controls (y_t)
% Formula: N = inv(Z'_{22}) * Z'_{21}
N = Z22p \ Z21p; % Using backslash is more stable than inv()

% Solve for L, which maps shocks (epsilon_t) to controls (y_t)
% Formula: L = inv(Z'_{22}) * inv(S_{22}) * [Q'_{21}*G1 + Q'_{22}*G2]
L = inv(Z22p) * inv(S22) * (Q21p * G1 + Q22p * G2);

% Solve for C, the state transition matrix
% Formula: C = inv(B11 - B12*N) * (A11 - A12*N)
inv_term_C = (B11 - B12 * N);
C = inv_term_C \ (A11 - A12 * N);

% Solve for D, the impulse response of states to shocks
% Formula: D = inv(B11 - B12*N) * (G1 - A12*L)
D = inv_term_C \ (G1 - A12 * L);


%% == Step 7: Display Results ==
fprintf('--- SOLVED POLICY FUNCTION MATRICES ---\n');
fprintf('Matrix N (relates controls to states, y_t = -N*x_t + ...):\n');
disp(N);
fprintf('Matrix L (relates controls to shocks, y_t = ... -L*eps_t):\n');
disp(L);
fprintf('Matrix C (state transition, x_{t+1} = C*x_t + ...):\n');
disp(C);
fprintf('Matrix D (state impulse response, x_{t+1} = ... + D*eps_t):\n');
disp(D);
fprintf('---------------------------------------\n');

%% ============================
%% reconstruct the state space
%% ============================
R = [C;-N];
S = [D;-L];
