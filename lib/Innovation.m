function [z_i, dV_forward, dV_backward] = Innovation(params, x, w, v)
% Innovation computes the optimal innovation decision z_i for each state 
% and the finite differences of the value function v.
%
% For each state n (except n = 1), the closed‐form expression is used:
%
%   z_i*(n) = [ (v(n+1)-v(n)) / (eta * w * zeta) ]^(1/(eta-1)),
%
% and then z_i(n) is bounded above by x(n).
%
% Finite differences are defined as:
%   dV_forward(n)  = v(n+1) - v(n)   (with special handling at the boundaries)
%   dV_backward(n) = v(n-1) - v(n)   (with dV_backward(1)=0)
%
% Inputs:
%   params - structure with fields: gamma, rho, eta, zeta, psi.
%   x      - vector of maximum allowed innovation decisions for each state.
%   w      - scalar, current guess for w.
%   v      - current value function (n×1 vector).
%
% Outputs:
%   z_i         - optimal innovation decision (n×1 vector), bounded by x.
%   dV_forward  - forward finite differences (n×1 vector).
%   dV_backward - backward finite differences (n×1 vector).

% Extract parameters.
gamma = params.gamma;
eta   = params.eta;
zeta  = params.zeta;

N = length(v);       % Number of states.
n_vec = (1:N)';      % Grid indices.

%% Compute finite differences
% --- Forward differences ---
dV_forward = zeros(N,1);
if N >= 2
    % For state 1, use v(2)-v(1).
    dV_forward(1) = v(2) - v(1);
end
if N > 2
    % For interior states (n = 2 to N-1): use v(n+1)-v(n).
    dV_forward(2:N-1) = v(3:N) - v(2:N-1);
end
if N >= 2
    % For state N, approximate dV_forward as v(N)-v(N-1).
    dV_forward(N) = v(N) - v(N-1);
end

% --- Backward differences ---
dV_backward = zeros(N,1);
% For state 1, no previous state so set to 0.
dV_backward(1) = 0;
if N >= 2
    % For states n >= 2: use v(n-1)-v(n).
    dV_backward(2:N) = v(1:N-1) - v(2:N);
end

%% Compute optimal innovation decision z_i for each state.
% Initialize z_i.
z_i = zeros(N,1);

for i = 2:N
    % If the forward difference is negative, set z_i(i)=0.
    if dV_forward(i) < 0
        z_i(i) = 0;
    else
        % Compute the closed-form solution for the optimal innovation decision.
        % Formula: 
        %   z_i*(i) = [ i*(v(i+1)-v(i)) / (eta * w * zeta) ]^(1/(eta-1))
        computed_z = (dV_forward(i) / (eta * w * zeta) )^(1/(eta-1));
        
        % Bound the computed innovation decision by the maximum allowed x(i).
        % This ensures that z_i(i) does not exceed x(i).
        z_i(i) = min(computed_z, x(i));
    end
end

