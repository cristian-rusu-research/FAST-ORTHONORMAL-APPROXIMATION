close all
clear
clc

% load the dataset
load('USPS.mat');

% number of principal components
p = 15;

% K, the number of neighbors for K-NN
K = 15;

% N is the size of the dataset and n is the dimension
[n, N] = size(X);

% normalize: remove the means
X = bsxfun(@minus, X, mean(X));
% X = bsxfun(@rdivide, X, sqrt(sum(X.^2)));

% train and test size
N_train = 8000;
N_test = N - N_train;

% create train and test datasets
support_train = randsample(1:N, N_train);
support_test = setdiff(1:N, support_train);
Xtr = X(:, support_train);
Xts = X(:, support_test);
Ytr = Y(support_train);
Yts = Y(support_test);

% we do not need this anymore
clear X Y;

%% perform singular value decomposition, get p principal components
[Up, Sp, Vp] = svds(Xtr, p);
projts = Up'*Xts;
projtr = Up'*Xtr;
accuracy = getAccuracy_KNN(projts, projtr, Ytr, Yts, K);

% number of generalized Givens transformations in the factorization
m = round(2*p*log2(256));

%% call on the algorithm
[Us, Xs, positionss, valuess, tuss] = fast_pca_transform(Up, m);

%% see Remark 1 of the paper
% check if all calculation are necessary to the final projection
operation_necessary = zeros(2, m);
projtr = Xtr;
projts = Xts;
nop = 0; % number of operations the projection takes
for h = m:-1:1
    A = projtr(positionss(1, h), :); C = projts(positionss(1, h), :);
    B = projtr(positionss(2, h), :); D = projts(positionss(2, h), :);
    if (positionss(1, h) <= p) || (any(find(positionss(1, 1:h-1)==positionss(1, h)))) || (any(find(positionss(2, 1:h-1)==positionss(1, h))))
        projtr(positionss(1, h), :) = valuess(1, h)*A + valuess(2, h)*B;
        projts(positionss(1, h), :) = valuess(1, h)*C + valuess(2, h)*D;
        nop = nop + 3;
        
        operation_necessary(1, h) = 1;
    end
        
    if (positionss(2, h) <= p) || (any(find(positionss(1, 1:h-1)==positionss(2, h)))) || (any(find(positionss(2, 1:h-1)==positionss(2, h))))
        projtr(positionss(2, h), :) = valuess(3, h)*A + valuess(4, h)*B;
        projts(positionss(2, h), :) = valuess(3, h)*C + valuess(4, h)*D;
        nop = nop + 3;
        
        operation_necessary(2, h) = 1;
    end
end
%% the new projections
projtr = projtr(1:p, :); % projtr is now Us(:, 1:p)'Xtr
% norm(projtr - Us(:, 1:p)'*Xtr) % double check corect result

projts = projts(1:p, :); % projts is now Us(:, 1:p)'Xts
% norm(projts - Us(:, 1:p)'*Xts) % double check corect result

%% get new accuracy
accuracy_fast = getAccuracy_KNN(projts, projtr, Ytr, Yts, K);

%% explicitly compute speedup
disp(['Accuracy of PCA + K-NN is ' num2str(accuracy) '%']);
disp(['Accuracy of fastPCA + K-NN is ' num2str(accuracy_fast) '%']);
disp(['Speedup is x' num2str(2*n*p/nop)]);

%% save results
save(['fast pca projections usps n = ' num2str(n) ' p = ' num2str(p) ' m = ' num2str(m) '.mat']);
