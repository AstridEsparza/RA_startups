function [v, v_vec, z_i_vec, x_i_vec, profit_vec, dV_forward, dV_backward, iter_conv] = VF_iter_v(params, x, w, T, v0, tol_v)
% VF_iter_v uses value function iteration to solve the functional Bellman
% Equation.
%
% Inputs:
%   params - structure with model parameters:
%            params.gamma : parameter for per-line profit
%            params.rho   : discount rate
%            params.eta   : curvature parameter in the cost function
%            params.zeta  : cost parameter
%   x      - current value of the endogenous parameter x (scalar or vector).
%   w      - current guess for omega (w) (scalar).
%   T      - maximum number of iterations.
%   v0     - initial guess for the value function (n×1 vector).
%   tol_v  - tolerance for convergence.
%
% Outputs:
%   v           - converged value function (n×1 vector).
%   v_vec       - matrix of value function iterates (n×(T+1) matrix).
%   z_i_vec     - matrix of optimal innovation decisions from each iteration (n×(T+1) matrix).
%   x_i_vec     - matrix of endogenous innovation rates (n×(T+1) matrix) [here constant, equal to input x].
%   profit_vec  - matrix of aggregated static profits (n×(T+1) matrix).
%   dV_forward  - forward finite differences (n×1 vector) computed in the final iteration.
%   dV_backward - backward finite differences (n×1 vector) computed in the final iteration.
%   iter_conv   - iteration index (column number in v_vec) at which convergence was achieved.
%
% The Bellman update is given by:
%   v(n) = { n*((gamma-1)/gamma) - w*zeta*(z_i)^eta - n*x*(v(n-1)-v(n)) + n*z_i*(v(n+1)-v(n)) } / rho,
% where the optimal innovation decision is determined by solving
%   z_i*(n) = argmax_{0 ≤ z ≤ x} [ dV_forward(n)*z - w*zeta*z^eta ].
%   or 
%   z_i*(n) = (v(n+1)-v(n))/eta*zeta*omega]^(1/(eta-1))
%
% Note: The grid vector is defined assuming N grid points.

    % Extract parameters.
    gamma = params.gamma;
    rho   = params.rho;
    eta   = params.eta;
    zeta  = params.zeta;
    
    % Set N using the number of rows of v_vec.
    N = length(v0);
    
    % Preallocate matrices to store iterations of v and z_i.
    v_vec = zeros(N, T+1);
    v_vec(:, 1) = v0;
    
    % In this model, we assume x is given and does not change during the inner loop.
    x_i_vec = repmat(x, 1, T+1);
    
    % Preallocate for innovation decisions. Initially, set to zero.
    z_i_vec = zeros(N, T+1);
    
    % Preallocate for the aggregated static profits.
    profit_vec = zeros(N, T+1);
    
    % Define grid vector (each state n corresponds to an index in 1:N).
    n_vec = (1:N)';
    
    % Initialize iter_conv to T+1 (default, if convergence is not reached).
    iter_conv = T + 1;
    
    % Main iteration cycle.
    for iter = 1:T
        % 1. Extract current value function iterate.
        v_old = v_vec(:, iter);
        
        % 2. Compute the optimal innovation decision z_i and the finite differences.
        %    The Innovation function computes:
        %      - z_i based on v_old,
        %      - dV_forward as v(n+1) - v(n) (with proper handling at boundaries),
        %      - dV_backward as v(n-1) - v(n) (with dV_backward(1)=0).
        [z_i, dV_forward, dV_backward] = Innovation(params, x, w, v_old);
        
        % 3. Compute the profit components for each state.
        cost = w * zeta * (z_i.^ eta);
        % Profit if the firm does not change its number of product lines:
        profit_no_change = n_vec.* ((gamma - 1) / gamma - cost);

        % Profit if the firm loses a product line (using the backward difference):
        profit_loss = (n_vec).*(dV_backward);
        
        % Profit if the firm innovates (using the forward difference):
        profit_innovate = (n_vec).*(dV_forward); 
      

        % 4. Compute the aggregated static profit v_new.
        %    The idea is that, with probability (1 - x - z_i), the firm stays;
        %    with probability x, it loses a product line; and with probability z_i, it gains one.
        v_new = (1/rho).* (profit_no_change + x.* profit_loss + z_i.* profit_innovate);
        
        % 5. Save the new iterate and the innovation decision.
        v_vec(:, iter+1) = v_new;
        z_i_vec(:, iter+1) = z_i;
        profit_vec(:, iter+1) = v_new;
        
        % 6. Check convergence: compare the change between the new and old value functions.
        d = max(abs(v_new - v_old));
        fprintf('Inner loop iteration %d, distance: %f\n', iter+1, d);
        
        if d < tol_v
            % Convergence is reached. Set iter_conv to the current iteration index.
            iter_conv = iter + 1;
            v = v_new;
            break;
        end
    end
    
    % If convergence is not reached within T iterations, set v to the last iterate.
    if iter == T && d >= tol_v
        v = v_vec(:, end);
    end

end
