% Finds an optimum weight vector to find an optimum coarse-grid operator
% to solve an MGRIT/Parareal problem
% Based on https://arxiv.org/pdf/1910.03726.pdf ("Krzysik").

%% Size of system
M = 100;

%% Randomly generate a circulant matrix with all eigenvalues <1
vec = zeros(M, 1);
vec(1) = rand(1);
vec(2) = rand(1);
vec(3) = rand(1);
vec(M - 1) = rand(1);
vec(M) = rand(1);
A = circulant(vec);
vec = vec / (max(arrayfun(@(x) norm(x), eig(A))) + 0.2); % Without the 0.2, this would set some eigenvalues strictly equal to 1.
A = circulant(vec);

%% Restriction matrix
R = zeros(3, M);
R(1, 1) = 1;
R(2, 2) = 1;
R(3, M) = 1;

%% Fixed DFT matrix (to enforce an ordering of eigenvalues)
F = dftmtx(M);
mu_calculation_matrix = F * R';
lambda = F * vec;

%% Set up and solve main problem
f = @(w) solve_with_w(w, mu_calculation_matrix, lambda, M);
weighting = @(l) 1/(1 - norm(l) + 1e-6)^2;
w_original = arrayfun(weighting, lambda);
[w, f_val, exit_flag, output] = patternsearch(f, w_original, [], [], ...
    [], [], zeros(M, 1), zeros(M, 1) + Inf, ...
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
