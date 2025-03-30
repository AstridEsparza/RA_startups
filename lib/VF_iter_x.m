function [x_hist, v_final, v_vec_final, z_i_final, z_hist, w_hist] = VF_iter_x(params, x_init, w_init, T, tol_v, maxOuterIter, tol_x, v0)
% VF_iter_x computes the optimal state-dependent decision variable x, 
% the converged value function v, the optimal innovation decisions z_i,
% and records the wage history w.
%
% The algorithm uses an outer loop that:
%   1) Initializes x and w.
%   2) Runs an inner loop (VF_iter_v) to compute v and z_i given current x and w.
%   3) Updates w using the first element of the converged v and parameter psi.
%   4) Obtains finite differences (dV_forward, dV_backward) directly from VF_iter_v.
%   5) Updates x using the rearranged Bellman equation in one expression.
%   6) Repeats until x converges.
%
% This version is more robust: it uses the convergence iteration index (iter_conv)
% returned by VF_iter_v so that we know exactly which iteration produced the 
% converged value function, rather than guessing from the last column.
%
% Inputs:
%   params       - structure with fields: gamma, rho, eta, zeta, psi.
%   x_init       - initial guess for x, an n×1 vector.
%   w_init       - initial guess for w (scalar).
%   T            - maximum number of iterations for the inner loop.
%   tol_v        - tolerance for convergence in the inner loop.
%   maxOuterIter - maximum number of outer loop iterations.
%   tol_x        - tolerance for convergence of x.
%   v0           - initial guess for the value function (n×1 vector).
%
% Outputs:
%   x_hist       - history matrix of x over outer iterations (n×#iterations).
%   v_final      - final converged value function (n×1 vector).
%   v_vec_final  - matrix of value function iterates from the inner loop.
%   z_i_final    - final innovation decision vector (n×1).
%   z_hist       - history matrix of innovation decisions over outer iterations.
%   w_hist       - vector of wage values (w_current) over outer iterations.

% Ensure initial w is positive.
if w_init <= 0
    error('w_init must be greater than 0.');
end

% Initialize outer loop variables.
x_current = x_init;    % Current decision variable vector.
w_current = w_init;    % Current wage parameter (scalar).
N = length(v0);        % Number of states (grid points).

% Preallocate history matrices.
x_hist = zeros(N, maxOuterIter);  % x history (each column corresponds to an iteration).
z_hist = zeros(N, maxOuterIter);  % z_i history.
w_hist = zeros(1, maxOuterIter);  % w history as a row vector.

% Begin outer loop iterations.
for outer_iter = 1:maxOuterIter
    fprintf('Outer iteration %d: updating x...\n', outer_iter);
    
    x_hist(:, outer_iter) = x_current;  % Save current x into history.
    w_hist(outer_iter) = w_current; % Save the current wage into the wage history.

    
    % ----- INNER LOOP -----
    % Run the inner loop with current x and w to get the converged value function v.
    % We assume VF_iter_v returns an additional output 'iter_conv' indicating convergence iteration.
    [v, v_vec, z_i_vec, x_i_vec, profit_vec, dV_forward, dV_backward, iter_conv] = ...
        VF_iter_v(params, x_current, w_current, T, v0, tol_v);
    
    % Extract the converged value function and innovation decision using iter_conv.
    v_converged = v_vec(:, iter_conv);
    z_i_star = z_i_vec(:, iter_conv);
    
    % ----- UPDATE w -----
    % Update w_current based on the first element of the converged value function
    % and the parameter psi.
    w_current = v_converged(1) / params.psi;
    
    
    % ----- SAFETY CHECK FOR z_i -----
    if any(z_i_star < 0)
        error('Safety check failed: Some elements of z_i_star are negative.');
    end
    
    % Save current z_i into history.
    z_hist(:, outer_iter) = z_i_star;
    
    % ----- UPDATE x USING THE REARRANGED BELLMAN EQUATION -----
    % For each interior state (n = 2 to N-1), update x with a single-line formula.
    x_new = zeros(N,1);
    for n = 2:(N-1)
        % The update equation is:
        %   x_new(n) = [ n*((gamma-1)/gamma) - w_current*zeta*(z_i_star(n)^eta) 
        %                 + n*z_i_star(n)*dV_forward(n) - rho*v_converged(n) ] / (n*dV_backward(n))
        x_new(n) = ( n*((params.gamma - 1)/params.gamma) ...
                     - w_current*params.zeta*(z_i_star(n)^params.eta) ...
                     + n*z_i_star(n)*dV_forward(n) ...
                     - params.rho*v_converged(n) ) / ( n*dV_backward(n) );
        % Enforce a strict positive lower bound on x_new(n) to avoid infeasibility.
        x_new(n) = max(x_new(n), eps);
    end
    
    % ----- CONVERGENCE CHECK FOR x -----
    if max(abs(x_new - x_current)) < tol_x
        fprintf('Outer loop converged at iteration %d.\n', outer_iter);
        x_current = x_new;
        break;
    end
    
    % Update x for the next outer iteration.
    x_current = x_new;
    
    % Update v0 for the next inner loop to be the current converged value function.
    v0 = v_converged;
end

% Truncate history matrices to the number of iterations performed.
x_hist = x_hist(:, 1:outer_iter);
z_hist = z_hist(:, 1:outer_iter);
w_hist = w_hist(1:outer_iter);

% Final outputs.
x_vec = x_current;          % Final x.
v_final = v_converged;      % Final converged value function.
v_vec_final = v_vec;        % Full value function iteration history.
z_i_final = z_i_star;       % Final innovation decisions.

end
