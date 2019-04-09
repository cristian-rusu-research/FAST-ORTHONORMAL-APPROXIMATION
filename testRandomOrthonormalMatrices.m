close all
clear
clc

%% dimension of the matrix
d = 100;

%% generate random orthogonal matrix
[Q, R] = qr(randn(d));

%% make sure the matrix is from the Haar measure
for i = 1:d
    if (R(i,i) < 0)
        Q(:, i) = -Q(:, i);
    end
end

%% number of Givens transformations
g = round(d*log2(d));

%% create the fast approximation
[positions, values, approx_error, time] = orthogonal_approximation(Q, g);

%% save results
%save(['approximation random orthogonal d = ' num2str(d) ' g = ' num2str(g) '.mat']);
