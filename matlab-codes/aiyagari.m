% This file replicates the paper 'Uninsured Idiosyncratic Risk and Aggregate Saving - QJE - 1994'.
% To solve the Bellman equation, this file uses the discretized state space method.

%% Step 1: Set Up Parameters
clear all

% Step 1.1: Economic Environment
param_beta = 0.96;                 % discount factor
param_alpha = 0.36;                % capital share in the Cobb-Douglas production function
param_delta = 0.08;                % depreciation rate
param_mu = [1, 3, 5];              % CRRA coefficient in the utility function
param_rho = [0.0, 0.3, 0.6, 0.9];  % serial correlation of income AR(1) process
param_sigma = [0.2, 0.4];          % coefficient of variation of income AR(1) process

% Step 1.2: Computational Constants
parcon_markovN = 7;                 % number of states in the Markov process to approximate the income AR(1) process 
parcon_markovm = 3;                 % the largest state in estimated Markov process is equal to m*sigma*sqrt(1-rho^2)
parcon_tol1 = 1e-5;                 % tolerance for inner calculation (value function iteration)
parcon_tol2 = 1e-3;                 % tolerance for outer calculation (price determination)
parcon_gridN = 1000;                % number of discretized state space
parcon_minCons = 1e-3;
% r_ss=1/param_beta-1;
% K_ss=((r_ss+param_delta)/param_alpha)^(1/(param_alpha-1));
parcon_asset_min = 1e-5;
parcon_asset_max = 100;

% Step 1.3: Set up the Makrov Chain for Income Process
rho_v = reshape(repmat(param_rho, length(param_sigma), 1), [], 1);
sigma_v = reshape(repmat(param_sigma', length(param_rho), 1), [], 1);
var_e = (sigma_v.^2) .* (1 - rho_v.^2);

for i=1:8
  [log_l{i}, P_logl{i}] = fun_Tauchen(parcon_markovN, parcon_markovm, rho_v(i), var_e(i));
end


%% Step 2: Obtain the Asset Demand Function Given Interest Rate r

% The asset demand function maps current period asset holdings to next period asset holdings, given the interest rate r.
% To solve the policy function, we need to solve the infinite horizon dynamic programming problem.
% 

% First, we need to link all other parameters to r.
% To derive them, note that Y = K^alpha * L^(1-alpha), r = Y_{K} - delta, and w = Y_{L}

param_mu = 3;
fun_utility = @(c) c.^(1-param_mu)./(1-param_mu);
log_l = log_l{3};
P_logl = P_logl{3};

r = 0.03; % r is the rental price of capital net of depreciation
w = (1-param_alpha) * ((r+param_delta)/param_alpha)^(param_alpha/(param_alpha-1));

x_asset_grid = linspace(parcon_asset_min, parcon_asset_max, parcon_gridN)'; % parcon_gridN by 1 vector
y_value_old = zeros(parcon_gridN, parcon_markovN);
y_value_new = repmat(fun_utility((1+r)*x_asset_grid), 1, parcon_markovN); 
asset_policy = zeros(parcon_gridN, parcon_markovN);
it_vfi = 0;
while max( abs(y_value_new-y_value_old), [], 'all') > parcon_tol1
  it_vfi = it_vfi + 1; 
  fprintf("iteration = %d", it_vfi)
  
  y_value_old = y_value_new;
  for indx_asset = 1:parcon_gridN
    for indx_l = 1:parcon_markovN
      p_l_tplus1 = P_logl(indx_l,:)'; % change the transition probability into a column vector
      ev_old = y_value_old * p_l_tplus1; % gridN by 1 vector
      rhs = fun_utility(w*exp(log_l(indx_l))+(1+r)*x_asset_grid(indx_asset)-x_asset_grid) + param_beta * ev_old; % gridN by 1
      [y_value_new(indx_asset, indx_l), indx_max] = max(rhs);
      asset_policy(indx_asset, indx_l) = x_asset_grid(indx_max);

      max_asset_policy = w*exp(log_l(indx_l))+(1+r)*x_asset_grid(indx_asset)-parcon_minCons;

      if asset_policy(indx_asset, indx_l) > max_asset_policy
        [~, indx_max] = min(abs(max_asset_policy - x_asset_grid));
        asset_policy(indx_asset, indx_l) = x_asset_grid(indx_max);
        y_value_new(indx_asset, indx_l) = fun_utility(w*exp(log_l(indx_l))+(1+r)*x_asset_grid(indx_asset)-x_asset_grid(indx_max)) + param_beta * ev_old(indx_max);
      end
    end
  end
  fprintf("     max(V_new-V_old) = %f\n", max(abs(y_value_new-y_value_old),[], 'all'))

end

%% Step 3: Obtain Invariant Distribution

y_dist_old = zeros(parcon_gridN, parcon_markovN);
y_dist_old(1,1) = 1;
y_dist_new = 1/(parcon_gridN * parcon_markovN) * ones(parcon_gridN, parcon_markovN);
it_dist = 0;
while any(max(abs(y_dist_new - y_dist_old), [], 'all') > parcon_tol1)
  it_dist = it_dist + 1;
  fprintf("iteration (for distribution) = %d", it_dist)
  y_dist_old = y_dist_new;
  for indx_asset = 1:300
    for indx_l = 1:3
      [~, indx_policy] = find( asset_policy(indx_asset, indx_l) == x_asset_grid );
      y_dist_new(indx_policy, :) = y_dist_new(indx_policy, :) + P_logl(indx_l, :) * y_dist_old(indx_asset, indx_l);
      y_dist_new(indx_asset, indx_l) = 0;
    end
  end
  fprintf("     max(dist_new-dist_old) = %f\n", max(abs(y_dist_new - y_dist_old), [], 'all'))

end

[~, indx_policy] = min( abs(asset_policy - x_asset_grid), [], 1 )



