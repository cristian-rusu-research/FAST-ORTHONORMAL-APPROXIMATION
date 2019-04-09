function [positions, values, approx_error, tus] = orthogonal_approximation(U, g)
%% Demo code for paper Fast PCA projections by generalized Givens transformations

%% Input:
% U - an orthonormal matrix of size dxd
% g - the number of generalized Givens transformations to use for the approximation

%% Output:
% The g generalized Givens transformations:
% positions - the two indices (i,j) where the transformation operates
% values - the four values of the transformations
% approx_error - the approximation error, as defined in the paper
% tus - the total running time

tic;
[d, ~] = size(U);

%% basic sanity check
if (d <= 1) || (g < 1)
    positions = []; values = []; tus = toc;
    return;
end
if norm(U'*U - eye(d)) >= 10e-7
    error('U has to be orthogonal');
end

%% make sure we have a positive integer
g = round(g);

%% vector that will store the indices (i,j) and the values of the transformations for each of the g Givens transformations
positions = zeros(2, g);
values = zeros(4, g);

%% number of iterations
K = 3;

%% compute all scores C_{ij}
scores_nuclear = zeros(d);
for i = 1:d
    for j = i+1:d
        T = U([i j], [i j]);
        c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
        scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
    end
end

%% initialization of each Givens transformation
Z = U;
for kk = 1:g
    %% check where the maximum scores is, to find the optimum indices
    [~, index_nuc] = max(scores_nuclear(:));
    [i_nuc, j_nuc] = ind2sub([d d], index_nuc);

    %% compute the optimum orthogonal transformation on the optimum indices
    [Uu, ~, Vv] = svd(Z([i_nuc j_nuc], [i_nuc j_nuc]));
    GG = Uu*Vv';

    %% save the Givens transformation
    positions(1, kk) = i_nuc;
    positions(2, kk) = j_nuc;
    values(:, kk) = vec(GG);
    
    %% update the working matrix
    Z = applyGTransformOnRightTransp(Z, i_nuc, j_nuc, values(:, kk));

    %% update the scores only for the coordinates that were selected, everything else is the same
    for i = [i_nuc j_nuc]
        for j = i+1:d
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

%% iterative process to refine the initialization
for k = 1:K
    Z = U;
    for k = g:-1:1
        Z = applyGTransformOnLeftTransp(Z, positions(1, k), positions(2, k), values(:, k));
    end
    
    for kk = 1:g
        Z = applyGTransformOnLeft(Z, positions(1, kk), positions(2, kk), values(:, kk));
   
        if (kk == 1)
            %% first time compute all the scores from scratch
            scores_nuclear = zeros(d);
            for i = 1:d
                for j = i+1:d
                    T = Z([i j], [i j]);
                    c1 = norm(T, 'fro')^2/2; c1_2 = c1^2; c2_2 = det(T)^2;
                    scores_nuclear(i, j) = sqrt(c1 + sqrt(c1_2 - c2_2)) + sqrt(c1 - sqrt(c1_2 - c2_2)) - trace(T);
                end
            end
        else
            %% update only indices that were selected and previously used
            for i = [positions(1, kk), positions(2, kk), positions(1, max(kk-1, 1)), positions(2, max(kk-1, 1))]
                for j = i+1:d
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
        
        %% check where the maximum scores is, to find the optimum indices
        [~, index_nuc] = max(scores_nuclear(:));
        [i_nuc, j_nuc] = ind2sub([d d], index_nuc);
        
        %% compute the optimum orthogonal transformation on the optimum indices
        [Uu, ~, Vv] = svd(Z([i_nuc j_nuc], [i_nuc j_nuc]));
        GG = Uu*Vv';
        
        %% save the Givens transformation
        positions(1, kk) = i_nuc;
        positions(2, kk) = j_nuc;
        values(:, kk) = vec(GG);
        
        %% update the working matrix
        Z = applyGTransformOnRightTransp(Z, positions(1, kk), positions(2, kk), values(:, kk));
    end
end

%% the explicit approximation, can be avoided
Ubar = eye(d);
for k = 1:g
    Ubar = applyGTransformOnLeft(Ubar, positions(1, k), positions(2, k), values(:, k));
end
approx_error = 1/2*norm(U - Ubar,'fro')^2/norm(U,'fro')^2;

%% time everything
tus = toc;
