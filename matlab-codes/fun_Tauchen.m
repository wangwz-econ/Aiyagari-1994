function [y, P] = fun_Tauchen(markovN, markovm, rho, var)
% This function implements finite-state Markov chain approximation to a univariate autoregression, a method proposed 
%   in paper 'Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions' by Tauchen (1986).
% The univariate AR(1) process is y_t+1 = rho*y_t + epsilon_t, where epsilon_t ~ Normal(0, var).

% Inputs:
%   markovN, scalar, number of states in the approximated Markov chain.
%   markovm, scalar, controls the largest state (the largest state value is m*sqrt(var/(1-rho^2))) (m*std(y)).
%   rho, scalar, the serial correlation coefficient.
%   var, scalar, variance of normally distributed white noise process.
% Outputs:
%   y, markovN by 1 vector, states of the approximated Markov chain.
%   P, markovN by markovN, transition matrix of the approximated Markov chain.

std = sqrt(var);
sigmay = sqrt(var/(1-rho^2));
ymax = markovm * sigmay;

y = linspace(-ymax, ymax, markovN)'; % markovN by 1 vector

% Calculate the transition probabilities
P = NaN(markovN, markovN);
step = y(2) - y(1);
for j=1:markovN
  for k = 1:markovN
    if k == 1
      P(j, k) = normcdf((y(k) - rho*y(j) + step/2), 0, std);
    elseif k == markovN
      P(j, k) = 1 - normcdf((y(k) - rho*y(j) - step/2), 0, std);
    else
      P(j, k) = normcdf((y(k) - rho*y(j) + step/2), 0, std) - normcdf((y(k) - rho*y(j) - step/2), 0, std);
    end
  end
end

% Another vectorized way to calculate the transition probabilities

% sigma = sqrt(var);
% sigmay = sqrt(var/(1-rho^2));
% 
% y = linspace(-markovm*sigmay, markovm*sigmay, markovN)';
% step = y(2)-y(1);
% 
% zi = y * ones(1,markovN);
% zj = 0 * ones(markovN, markovN) + ones(markovN, 1) * y';
% 
% P_part1 = normcdf(zj + step/2 - rho*zi, 0, sigma);
% P_part2 = normcdf(zj - step/2 - rho*zi, 0, sigma);
% 
% P             = P_part1 - P_part2;
% P(:, 1)       = P_part1(:, 1);
% P(:, markovN) = 1 - P_part2(:, markovN);


end