import numpy as np
from numpy.linalg import inv
from numpy import kron
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


# Parameters
alpha = 0.35
beta = 0.985
eta = 2
phi = 1.5
delta = 0.025
rhoa = 0.95
sigmae = 0.01

# === STEADY STATES ===
# Step 1: Compute the steady state
rss = 1 / beta + delta - 1
wss = (1 - alpha) * ((alpha / rss) ** (alpha / (1 - alpha)))
nss = ((wss ** (1 / eta)) / (wss / (1 - alpha) - delta * ((wss / (1 - alpha)) ** (1 / alpha)))) ** (1 / (phi / eta + 1))
iss = delta * ((alpha / rss) ** (1 / (1 - alpha))) * nss
css = (wss / (nss ** phi)) ** (1 / eta)
kss = ((alpha / rss) ** (1 / (1 - alpha))) * nss
yss = kss ** alpha * nss ** (1 - alpha)

print("Steady states:")
print([rss, wss, yss, iss, css, nss, kss])

# Define matrices
A = np.array([[1], [0], [0], [0], [0], [0]])
B = np.array([[-(1 - delta)], [-alpha], [0], [0], [1], [0]])
C = np.array([
    [0, 0, 0, 0, 0, -delta],
    [1, 0, -(1 - alpha), 0, 0, 0],
    [0, eta, phi, -1, 0, 0],
    [-1, 0, 1, 1, 0, 0],
    [-1, 0, 0, 0, 1, 0],
    [-yss, css, 0, 0, 0, iss]
])
D = np.array([[0], [-1], [0], [0], [0], [0]])
F = np.array([[0]])
G = np.array([[0]])
H = np.array([[0]])
J = np.array([[0, eta / beta, 0, 0, -rss, 0]])
K = np.array([[0, -eta / beta, 0, 0, 0, 0]])
L = np.array([[0]])
M = np.array([[0]])
N = np.array([[rhoa]])

# Solve for P (scalar)
C_inv = inv(C)
a = F - J @ C_inv @ A
b = - (J @ C_inv @ B - G + K @ C_inv @ A)
c = - K @ C_inv @ B + H
DELTA = b**2 - 4 * a * c
P1 = (-b + np.sqrt(DELTA)) / (2 * a)
P2 = (-b - np.sqrt(DELTA)) / (2 * a)
P = min(abs(P1[0, 0]), abs(P2[0, 0]))

# Solve for R
R = -C_inv @ (A * P + B)

# Solve for Q
k = 1  # Number of shocks
I_k = np.eye(k)

LHS = kron(N.T, F - J @ C_inv @ A) + kron(I_k, J @ R + F * P + G - K @ C_inv @ A)
RHS = (J @ C_inv @ D - L) @ N + K @ C_inv @ D - M
Q_vec = inv(LHS) @ RHS
Q = Q_vec  # Scalar, no reshape needed

# Solve for S
S = -C_inv @ (A * Q + D)

# Display results
print("P =", P)
print("R =\n", R)
print("Q =", Q)
print("S =\n", S)

# Set parameters
T = 200  # Time horizon

# Initialize arrays
tilde_k = np.zeros(T)
tilde_y = np.zeros(T)
tilde_c = np.zeros(T)
tilde_n = np.zeros(T)
tilde_w = np.zeros(T)
tilde_r = np.zeros(T)
tilde_i = np.zeros(T)
tilde_A = np.zeros(T)

# One-time shock
s = 1  # shock at period 1 (Python uses 0-based indexing)
tilde_A[s] = 0.01  # standard deviation of shock

# Generate IRFs
for t in range(s, T - 1):
    tilde_A[t + 1] = rhoa * tilde_A[t]
    tilde_k[t + 1] = P * tilde_k[t] + Q.item() * tilde_A[t]
    tilde_y[t] = R[0, 0] * tilde_k[t] + S[0, 0] * tilde_A[t]
    tilde_c[t] = R[1, 0] * tilde_k[t] + S[1, 0] * tilde_A[t]
    tilde_n[t] = R[2, 0] * tilde_k[t] + S[2, 0] * tilde_A[t]
    tilde_w[t] = R[3, 0] * tilde_k[t] + S[3, 0] * tilde_A[t]
    tilde_r[t] = R[4, 0] * tilde_k[t] + S[4, 0] * tilde_A[t]
    tilde_i[t] = R[5, 0] * tilde_k[t] + S[5, 0] * tilde_A[t]

# Collect results
variables = [tilde_y, tilde_c, tilde_n, tilde_w, tilde_r, tilde_i, tilde_k, tilde_A]
labels = ['y', 'c', 'n', 'w', 'r', 'i', 'k', 'A']
horizon = np.arange(s + 1, T - 1)  # same as s+1:T-1 in MATLAB

# Plot IRFs
plt.figure(figsize=(12, 8))
for i, (var, label) in enumerate(zip(variables, labels)):
    plt.subplot(3, 3, i + 1)
    plt.plot(horizon, var[horizon], 'b', linewidth=1)
    plt.axhline(0, color='r', linewidth=1)
    plt.title(label)
    plt.grid(True)

plt.tight_layout()
plt.savefig('irf.png', dpi=300)

## STOCHASTIC SIMULATION
# Parameters
T = 9000
burnin = 1000
s = 1  # index of shock

# Initialize arrays
tilde_k = np.zeros(T)
tilde_y = np.zeros(T)
tilde_c = np.zeros(T)
tilde_n = np.zeros(T)
tilde_w = np.zeros(T)
tilde_r = np.zeros(T)
tilde_i = np.zeros(T)
tilde_A = np.zeros(T)

#np.random.seed(666)
e = sigmae * np.random.randn(T)

# Simulate
for t in range(T - 1):
    tilde_A[t + 1] = rhoa * tilde_A[t] + e[t]
    tilde_k[t + 1] = P * tilde_k[t] + Q.item() * tilde_A[t]
    tilde_y[t] = R[0, 0] * tilde_k[t] + S[0, 0] * tilde_A[t]
    tilde_c[t] = R[1, 0] * tilde_k[t] + S[1, 0] * tilde_A[t]
    tilde_n[t] = R[2, 0] * tilde_k[t] + S[2, 0] * tilde_A[t]
    tilde_w[t] = R[3, 0] * tilde_k[t] + S[3, 0] * tilde_A[t]
    tilde_r[t] = R[4, 0] * tilde_k[t] + S[4, 0] * tilde_A[t]
    tilde_i[t] = R[5, 0] * tilde_k[t] + S[5, 0] * tilde_A[t]

# Collect data
variables = [tilde_y, tilde_c, tilde_n, tilde_w, tilde_r, tilde_i, tilde_k, tilde_A, e]
labels = ['y', 'c', 'n', 'w', 'r', 'i', 'k', 'A', 'e']
horizon = np.arange(s, 200)

# Plot
plt.figure(figsize=(12, 8))
for i, (var, label) in enumerate(zip(variables, labels)):
    plt.subplot(3, 3, i + 1)
    plt.plot(horizon, var[horizon], 'b', linewidth=1)
    plt.axhline(0, color='r', linewidth=1)
    plt.title(label)
    plt.grid(True)
plt.tight_layout()
plt.savefig('stoch.png', dpi=300)

# === THEORETICAL MOMENTS ===

# Stack and trim data: shape = (T-burnin) x 9
data_mat = np.stack(variables, axis=0)[:, burnin:].T

# 1. Means
means = np.mean(data_mat, axis=0)

# 2. Standard deviations
stddevs = np.std(data_mat, axis=0)

# 3. Variances
variances = np.var(data_mat, axis=0)

# 4. Correlation matrix
correlations = np.corrcoef(data_mat, rowvar=False)

# 5. First-order autocorrelations
autocorrs = []
for i in range(data_mat.shape[1]):
    acf_vals = acf(data_mat[:, i], nlags=4, fft=True)
    autocorrs.append(acf_vals[1])  # lag-1

# Display results
print("=== THEORETICAL MOMENTS ===")
print("\n{:<6s} {:>10s} {:>12s} {:>12s}".format("Var", "Mean", "Std Dev", "Variance"))
for i, label in enumerate(labels):
    print("{:<6s} {:>10.4f} {:>12.4f} {:>12.4f}".format(label, means[i], stddevs[i], variances[i]))

print("\n=== CORRELATION MATRIX ===")
print(np.round(correlations, 4))

print("\n=== FIRST-ORDER AUTOCORRELATIONS ===")
for i, label in enumerate(labels):
    print("{:<6s}: {:.4f}".format(label, autocorrs[i]))
