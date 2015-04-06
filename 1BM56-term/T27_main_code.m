% Authors: Thom Hurks and Vincent de Vos
% Group 27, 1BM56 Business Intelligence, april 6, 2015.

% For reproducibility:
rng(0,'twister');

% Data is the input table, provided by the teachers, it's inside 1BM56-term.mat.
load('1BM56-term.mat');

% Create the empty matrix that will contain the normalized data.
normalized_data = zeros(size(data, 1),0);

% Normalization:
% Loop over all columns, column 17 (outcome) we do separately.
for i=1:16
    curArray = table2array(data(:,i));
    % Normalize categorical data by turning them into logical vectors.
    if iscategorical(curArray)
        categories = unique(curArray);
        % Remove <undefined>s from categories.
        % <undefined> then becomes a zero-vector.
        categories = categories(~isundefined(categories));
        % Category matrix: each row will become a vector of zeroes and
        % ones, where the 1 indicates the category of that row.
        cat_matrix = zeros(size(curArray, 1), size(categories, 1));
        % For each entry in the column, create the logical vector and
        % put it in the matrix.
        for j=1:size(curArray, 1)
            cat_matrix(j,:) = categories == curArray(j);
        end;
        % Concatenate to the matrix containing the normalized data.
        normalized_data = horzcat(normalized_data, cat_matrix);
    % Normalize numerical data by numerical normalization; range to [0-1]
    elseif isnumeric(curArray)
        largest = max(curArray);
        smallest = min(curArray);
        range = largest - smallest;
        curArray = (curArray - smallest) / range;
        % Concatenate to the matrix containing the normalized data.
        normalized_data = horzcat(normalized_data, curArray);
    else
        display(strcat('Error normalizing column ', num2str(i)))
    end;
end;
% Now do the last column (outcome), convert yes/no to 1 and 0 (binary).
curArray = table2array(data(:,17));
normalized_data = horzcat(normalized_data, curArray == '"yes"');
display('Finished normalizing data');

% Create final test set, training set and validation set:

% Sample 10% observations uniformly at random, without replacement, for the
% final test set. We want the final test set to have the same imbalance
% as the original test set, to make the test more representative.
final_test_set_size = floor(size(normalized_data, 1) * 0.1);
[final_test_set, final_test_idx] = datasample(normalized_data, final_test_set_size ,...
                                   'Replace', false);
% Separate the final test data from the validation/train data for now.
train_val_data = removerows(normalized_data, final_test_idx);
% Divide remaining training/validation set in set of subscribed and 
% unsubscribed data.
last_column = size(train_val_data, 2);
set_subscribed = train_val_data(train_val_data(:,last_column) == 1,:);
set_unsubscribed = train_val_data(train_val_data(:,last_column) == 0,:);
% Get nr of subscribed entries.
balanced_size_subscribed = size(set_subscribed, 1);
% Calculate number of unsubscribed samples to get desired 70/30 ratio.
balanced_size_unsubscribed = floor((70 * balanced_size_subscribed) / 30);
% Sample k observations uniformly at random, without replacement.
balanced_unsubscribed = datasample(set_unsubscribed, balanced_size_unsubscribed,...
                                   'Replace', false);
% Concatenate the subscribed and sampled unsubcribed matrices to create
% a 70/30 balanced dataset.
balanced_data = vertcat(set_subscribed, balanced_unsubscribed);
% Shuffle the rows of the matrix by doing a random permutation from
% 1 to the row count of the matrix.
balanced_data = balanced_data(randperm(size(balanced_data, 1)),:);
train_val_size = size(balanced_data, 1);
% Add the final test set back to the end of the dataset, since the neural
% network expects one contiguous dataset.
balanced_data = vertcat(balanced_data, final_test_set);
display('Finished balancing data');

% Split the balanced dataset into inputs and labels for the neural network.
% We need to transpose the input/labels since the neural network
% expects that each input entry is a column vector and the labels is a row.
nn_Inputs = transpose(balanced_data(:,(1:(last_column - 1))));
nn_Labels = transpose(balanced_data(:,last_column));

% Create feedforward neural network with 2 hidden layers.
% First hidden layer has 10 neurons, and second 2 neurons.
% Training function is the default.
% alternative: net = patternnet([100 20], 'trainlm');
net = fitnet([10, 2], 'trainlm');

% Specify train, validation and test sets on the neural network.
% The data is divided by index, the indices for the three subsets are
% defined by the division parameters trainInd, valInd and testInd.
% Train size is 80% of the balanced train/validation set.
trainSize = floor(train_val_size * 0.8);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:trainSize;
net.divideParam.valInd = (trainSize + 1):train_val_size;
% The final test set is the unbalanced last part of the balanced_data
% array that we concatenated back in line 80.
net.divideParam.testInd = (train_val_size + 1):size(balanced_data, 1);

% Train the neural network with the data and get the output for the test
% set.
[net, train_record] = train(net, nn_Inputs, nn_Labels);
display('Finished training neural network');

% Show the trained neural network.
view(net);
% Test the performance of the neural networks using the final test set.
results = net(nn_Inputs(train_record.testInd));
nn_perf = perform(net, nn_Labels(train_record.testInd), results);
% Save the neural network to disk, in a file called "<name>_<timestamp>.m"
timestamp = strcat(datestr(clock,'yyyy-mm-dd-HHMM'),'m',datestr(clock,'ss'),'s');
save(strcat('T27_neural_net_', timestamp), 'net');

% Data set for Fuzzy Inference Model 

% TODO %

display('Done!');
