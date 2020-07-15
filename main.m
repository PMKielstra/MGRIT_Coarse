% Finds an optimum weight vector to find an optimum coarse-grid operator
% to solve an MGRIT/Parareal problem
% Based on https://arxiv.org/pdf/1910.03726.pdf ("Krzysik").

%% Size of system
space_steps = 2^8;
time_steps = 2^9;

%% Eigenvectors
dx = 2/space_steps;
dt = 2/time_steps;
L = mat_from_four_elt_upwind_stencil(1/(6 * dx) * [1 -6 3 2], space_steps);
Phi = calc_Phi(L, dt, space_steps);
disp(eig(Phi));
plot(eig(Phi), 'o');
pbaspect([1 1 1]);

%% Randomly generate a circulant matrix with all eigenvalues <1
vec = zeros(space_steps, 1);
vec(1) = rand(1);
vec(2) = rand(1);
vec(3) = rand(1);
vec(space_steps - 1) = rand(1);
vec(space_steps) = rand(1);
A = circulant(vec);
vec = vec / (max(arrayfun(@(x) norm(x), eig(A))) + 0.2); % Without the 0.2, this would set some eigenvalues strictly equal to 1.
A = circulant(vec);

%% Restriction matrix
R = zeros(3, space_steps);
R(1, 1) = 1;
R(2, 2) = 1;
R(3, space_steps) = 1;

%% Fixed DFT matrix (to enforce an ordering of eigenvalues)
F = dftmtx(space_steps);
mu_calculation_matrix = F * R';
lambda = F * vec;

%% Set up and solve main problem
f = @(w) solve_with_w(w, mu_calculation_matrix, lambda, space_steps);
weighting = @(l) 1/(1 - norm(l) + 1e-6)^2;
w_original = arrayfun(weighting, lambda);
[w, f_val, exit_flag, output] = patternsearch(f, w_original, [], [], ...
    [], [], zeros(space_steps, 1), zeros(space_steps, 1) + Inf, ...
    optimoptions(@patternsearch, 'Display', 'iter'));

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

%% For use in U3 upwind discretization
function L = mat_from_four_elt_upwind_stencil(S, M)
    L = zeros(M);
    L(1, 1:2) = S(3:4);
    L(2, 1:3) = S(2:4);
    for i=3:M-1
        L(i, (i-2):(i+1)) = S;
    end
    L(M, M-2:M) = S(1:3);
end