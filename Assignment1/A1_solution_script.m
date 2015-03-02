% Authors: Thom Hurks and Vincent de Vos
% Group 27, 1BM56 Business Intelligence, march 2, 2015.

% For reproducibility:
rng(0,'twister');
% D is the input table, provided by the teachers, it's inside A1.mat.
load('A1.mat');
predictors = table2array(D(:,1:3));
[input_length,~] = size(D);
labels = table2array(D(:,4));
% Train-Validation-Test partition
% 80% is in train-val, 20% is final test set
TVTPartition = cvpartition(labels,'HoldOut', 0.2);
% Find returns indices of nonzero entries.
final_test_indices = find(TVTPartition.test);
final_test_predictors = table2array(D(final_test_indices,1:3));
final_test_labels = table2array(D(final_test_indices,4));
train_validation_data = TVTPartition.training;
% Train-Test partition for 10-fold cross validation training.
TTPartition = cvpartition(train_validation_data, 'KFold', 10);
% Final-Train partition for final training on whole (80%) TT data.
FTPartition = cvpartition(train_validation_data,'resubstitution');
% Find out which amount of minimum leafs gives the lowest loss.
leafs = linspace(1,800,100); % alternative is logspace(1,3,100);
leaf_tests = numel(leafs);
losses = zeros(leaf_tests,1);
for n=1:leaf_tests
    tree = fitctree(predictors, labels, 'CVPartition', TTPartition,...
                                        'CrossVal', 'on',...
                                        'MinLeaf', leafs(n));
    losses(n) = kfoldLoss(tree);
end
plot(leafs,losses);
xlabel('Min Leaf Size');
ylabel('Cross-Validated Error');
[min_avg_loss, min_avg_loss_index] = min(losses);
best_leaf_count = leafs(min_avg_loss_index);
% Use the discovered best minimum leaf count to train a tree on the whole
% training data (80% of D).
best_tree_model = fitctree(predictors, labels, 'CVPartition', FTPartition,...
                                               'MinLeaf', best_leaf_count);
best_tree = best_tree_model.Trained{1};
view(best_tree, 'mode', 'graph');
% Use the best tree to create predictions based on final test test (20%).
predictions = predict(best_tree, final_test_predictors);
% Compare predictions with reality and create confusion matrix.
confusion = confusionmat(final_test_labels, predictions);
% Calculate outcome of predictions using the tree.
true_positives = confusion(2,2);
true_negatives = confusion(1,1);
false_positives = confusion(1,2);
false_negatives = confusion(2,1);
overall_success_rate = (true_positives + true_negatives)/(input_length);
error = 1 - overall_success_rate;
precision = true_positives/(true_positives + false_positives);
recall = true_positives/(true_positives + false_negatives);
F1 = 2*(precision * recall)/(precision + recall);
% Done!