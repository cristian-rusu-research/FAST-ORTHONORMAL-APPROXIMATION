function [U, X, positions, values, tus] = fast_pca_transform(Up, m, S)
%% Demo code for paper Fast PCA projections by generalized Givens transformations

%% Input:
% Up - orthogonal projections, size d x p, p <= d
% m - the number of generalized Givens transformations to use for the approximation
% S - the spectrum

%% Output:
% The g generalized Givens transformations:
% U - the explicit fast projection
% X - the spectrum
% positions - the two indices (i,j) where the transformation operates
% values - the four values of the transformations
% tus - the total running time

tic;
[n, p] = size(Up);

if (nargin < 3)
    S = eye(p);
end

X = speye(n); X = X(:, 1:p); X(1:p, 1:p) = S;

Up = Up*S;

positions = zeros(2, m);
values = zeros(4, m);

% number of iterations
K = 5;

%% the initialization
% compute all the scores
Z = Up*X';
scores_nuclear = zeros(n, n);
for i = 1:n
    for j = i+1:n
        T = Z([i j], [i j]);
        c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
        scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
    end
end

% initialize all the Givens transformations
for kk = 1:m
    [~, index_nuc] = max(scores_nuclear(:));
    [i_nuc, j_nuc] = ind2sub([n n], index_nuc);

    [Uu, ~, Vv] = svd(Z([i_nuc j_nuc], [i_nuc j_nuc]));
    GG = Uu*Vv';

    positions(1, kk) = i_nuc;
    positions(2, kk) = j_nuc;
    values(:, kk) = vec(GG);
    
    Z = applyGTransformOnRightTransp(Z, i_nuc, j_nuc, values(:, kk));

    for i = [i_nuc j_nuc]
        for j = i+1:n
            T = Z([i j], [i j]);
            c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
            scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
        end
    end

    for j = [i_nuc j_nuc]
        for i = 1:j-1
            T = Z([i j], [i j]);
            c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
            scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
        end
    end
end

%% the iterative process
for k = 1:K
    Z = Up*X';
    for h = m:-1:1
        Z = applyGTransformOnLeftTransp(Z, positions(1, h), positions(2, h), values(:, h));
    end
    
    for kk = 1:m
        Z = applyGTransformOnLeft(Z, positions(1, kk), positions(2, kk), values(:, kk));
        
        if (kk == 1)
            %% first time compute all the scores from scratch
            scores_nuclear = zeros(n, n);
            for i = 1:n
                for j = i+1:n
                    T = Z([i j], [i j]);
                    c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
                    scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
                end
            end
        else
            %% update only indices that were selected and previously used
            for i = [positions(1, kk), positions(2, kk), positions(1, max(kk-1, 1)), positions(2, max(kk-1, 1))]
                for j = i+1:n
                    T = Z([i j], [i j]);
                    c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
                    scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
                end
            end
            
            for j = [positions(1, kk), positions(2, kk), positions(1, max(kk-1, 1)), positions(2, max(kk-1, 1))]
                for i = 1:j-1
                    T = Z([i j], [i j]);
                    c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
                    scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
                end
            end
        end
        
        [~, index_nuc] = max(scores_nuclear(:));
        [i_nuc, j_nuc] = ind2sub([n n], index_nuc);
        
        [Uu, ~, Vv] = svd(Z([i_nuc j_nuc], [i_nuc j_nuc]));
        GG = Uu*Vv';
        
        positions(1, kk) = i_nuc;
        positions(2, kk) = j_nuc;
        values(:, kk) = vec(GG);
        
        Z = applyGTransformOnRightTransp(Z, positions(1, kk), positions(2, kk), values(:, kk));
    end
end

%% the explicit projection, can be avoided
U = eye(n);
for h = 1:m
    U = applyGTransformOnLeft(U, positions(1, h), positions(2, h), values(:, h));
end

tus = toc;
