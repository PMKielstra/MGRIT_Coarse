% Finds an optimum weight vector to find an optimum coarse-grid operator
% to solve an MGRIT/Parareal problem
% Based on https://arxiv.org/pdf/1910.03726.pdf ("Krzysik").
clear all;
clc;
%% Size of system
space_steps = 2^8;
time_steps = 2^9;

%% Eigenvectors
dx = 2/space_steps;
dt = 1/time_steps;
vec = zeros(space_steps, 1);
vec(1) = 3;
vec(2) = -6;
vec(3) = 1;
vec(space_steps) = 2;
L = circulant(-vec, 1);

Phi = calc_Phi(L, 0.85*dt/dx, space_steps);

%% Restriction matrix
nz = nnz(Phi(:, 1));
R = zeros(nz, space_steps);
j = 1;
for i=1:space_steps
    if Phi(i, 1) ~= 0
        R(j, i) = 1;
        j = j + 1;
    end
end

Phi = Phi^2;
plot(eig(Phi), 'o r');
hold on;
pbaspect([1 1 1]);


%% Fixed DFT matrix (to enforce an ordering of eigenvalues)
F = dftmtx(space_steps);
mu_calculation_matrix = F * R';
lambda = F * Phi(:, 1);

%% Set up and solve main problem
f = @(w) solve_with_w(w, mu_calculation_matrix, lambda, space_steps);
weighting = @(l) 1/(1 - norm(l) + 1e-6)^2;
w_original = arrayfun(weighting, lambda);

ps_eigs = mu_calculation_matrix * lsqlin(diag(w_original)^(1/2) * mu_calculation_matrix, diag(w_original)^(1/2) * lambda);
plot(ps_eigs, 'x b');

[w, f_val, exit_flag, output] = patternsearch(f, w_original, [], [], ...
    [], [], zeros(space_steps, 1), zeros(space_steps, 1) + Inf, ...
    optimoptions(@patternsearch, 'Display', 'iter', 'MaxIterations', 15));

new_ps_eigs = mu_calculation_matrix * lsqlin(diag(w)^(1/2) * mu_calculation_matrix, diag(w)^(1/2) * lambda);
plot(new_ps_eigs, '+ g');

%% Subordinate problem taken from Krzysik
function maxerr = solve_with_w(w, mu_calculation_matrix, lambda, M)
    Psi = lsqlin(diag(w)^(1/2) * mu_calculation_matrix, diag(w)^(1/2) * lambda); % Equation 8 in Krzysik
    mu = mu_calculation_matrix * Psi;
    E = zeros(M, 1);
    for i=1:M
        E(i) = norm(lambda(i)) * norm(lambda(i) - mu(i)) / (1 - norm(mu(i))) * (1 - norm(mu(i))^M); % Equation 7 in Krzysik
    end
    maxerr = max(E);
end

%% Runge-Kutta ERK3
function Phi = calc_Phi(L, k, M)
    U1 = eye(M) + k * L;
    U2 = (3/4) * eye(M) + (1/4) * U1 + (k/4) * L * U1;
    Phi = (1/3) * eye(M) + (2/3) * U2 + (2*k/3) * L * U2;
end
