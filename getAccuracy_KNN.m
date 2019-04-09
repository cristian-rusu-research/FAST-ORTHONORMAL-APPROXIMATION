function accuracy = getAccuracy_KNN(newZ, newX, labels_train, labels_test, k)
% newZ = U'*Z;

numTest = length(labels_test);

idx = knnsearch(newX', newZ', 'K', k);

for i = 1:numTest
    for j = 1:k
        idx(i,j) = labels_train(idx(i,j));
    end
end

scores = zeros(numTest, k);
for i = 1:numTest
    for j = 1:k
        scores(i, j) = length(find(idx(i,:)==(j-1)));
    end
end

guess = zeros(numTest, 1);
for i = 1:numTest
    [max_v, max_i] = max(scores(i,:));
    guess(i) = max_i - 1;
end

accuracy = length(find(guess==labels_test))/length(labels_test)*100;
