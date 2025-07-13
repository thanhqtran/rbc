function [Q, Z, S, T] = generalizedSchurDecomposition(A, B)
% generalizedSchurDecomposition: Performs a REAL generalized Schur decomposition
% ordered by eigenvalue magnitude.
%
% This function computes the real QZ decomposition, returning real matrices
% Q, Z, S, and T such that A = Q*S*Z' and B = Q*T*Z'.
%
% -- IMPORTANT --
% To ensure real-valued outputs, S is QUASI-UPPER TRIANGULAR (it may have
% 2x2 blocks on its diagonal), while T is upper triangular. Q and Z are
% orthogonal.
%
% The output is ordered so that the absolute values of the generalized
% eigenvalues increase down the diagonal.
%
% Syntax:
%   [Q, Z, S, T] = generalizedSchurDecomposition(A, B)

% --- Input Validation ---
if any(size(A) ~= size(B)) || (size(A, 1) ~= size(A, 2))
    error('Input matrices A and B must be square and of the same size.');
end
if ~isreal(A) || ~isreal(B)
    error('Input matrices A and B must be real for a real decomposition.');
end

n = size(A, 1);

% --- QZ Decomposition (Real) ---
% Use the 'real' flag to force real-valued orthogonal Q and Z.
% This results in a quasi-upper triangular S and an upper triangular T.
[S_init, T_init, Q_m, Z_m] = qz(A, B, 'real');

% --- Eigenvalue Sorting ---
% To correctly sort, we get the generalized eigenvalues directly from A and B.
% This is the most reliable way, as extracting them from a quasi-upper
% triangular S is complex.
eigenvalues = eig(A, B);

% Get the sorting permutation based on ascending absolute value.
[~, sorted_indices] = sort(abs(eigenvalues));

% Create a 'clusters' vector for ordqz.
clusters = zeros(n, 1);
clusters(sorted_indices) = n:-1:1;

% --- Reorder the Decomposition ---
% ordqz correctly handles the 2x2 blocks in the quasi-upper triangular S.
[S, T, Q_m_ord, Z_m_ord] = ordqz(S_init, T_init, Q_m, Z_m, clusters);

% --- Format Output Matrices ---
% For the form A = Q*S*Z', we define Q = Q_m_ord' and Z = Z_m_ord.
Q = Q_m_ord';
Z = Z_m_ord;

end
