%% Initialization
close all;
clear all;

addpath('lib/')

% Setting for Figures
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultTextFontSize', 10);
set(0, 'DefaultUicontrolFontSize', 10);
set(0, 'DefaultAxesLinewidth', 1.5);
set(0, 'DefaultLineLinewidth', 1.5);
set(0, 'DefaultPatchLinewidth', 1.5);

%% Calibration

params.rho      = 0.05;       % discount factor (?)
params.gamma    = 1.05;       % => per-line profit pi_val = (gamma-1)/gamma
params.eta      = 2.0;        % curvature in cost function z_i^eta
params.zeta     = 0.5;        % cost parameter
params.psi      = 0.5;        % free-entry parameter
params.L        = 2.0;        % used in labor-market clearing


%% Iteration Parameters
Nmax     = 50;         % Maximum number of product lines (grid size)
tol_v    = 0.01;       % Tolerance for inner loop on V(n)
tol_x    = 0.01;       % Tolerance for outer loop on x
maxIterV = 100;       % Maximum iterations for inner loop
maxIterX = 100;         % Maximum iterations for outer loop

%% Initial Values
w_init   = 0.5;         % Initial guess for wage
x = 0.9;     % Initial guess for aggregate innovation
x_init = repmat(x, Nmax, 1);


% Static Profits
pi = (params.gamma-1)/params.gamma;

%% Plot Value Function (Sanity check)

% --- Define the Grid ---
% The value function V(n) is defined on a grid for n = 1, 2, ..., Nmax
n_grid = (1:Nmax)';

% --- Initial Guess for the Value Function ---
% Set v0 as ((pi_val/rho) * n)^1.5 (Nmax x 1 vector)
v0 = ((pi / params.rho) .* n_grid).^1.5;

%% Inner Loop --- Call the Value Function Iteration Function ---
%[v,v_vec, z_i_vec, x_i_vec, profit_vec] = VF_iter_v(params, x_init, w_init, maxIterV, v0, tol_v);


%% Outer Loop --- Call the VFI within optimizing x

[x_hist, v_final, v_vec_final, z_i_final, z_hist, w_hist] = VF_iter_x(params, x_init, w_init, maxIterV, tol_v, maxIterX, tol_x, v0)

% Plot 

colIdx_v = find(any(v_vec, 1), 1, 'last');
v_converged = v_vec(:, colIdx_v);
plot(v_converged); 