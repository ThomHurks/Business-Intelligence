% Authors: Thom Hurks and Vincent de Vos
% Group 27, 1BM56 Business Intelligence, april 6, 2015.

% For reproducibility:
rng(0,'twister');

% data is the input table, provided by the teachers, it's inside 1BM56-term.mat.
load('1BM56-term.mat');

% Divide original data set in set of subscribed and unsubscribed data
set_subscribed = data(table2array(data(:,17)) == '"yes"',:);
set_unsubscribed = data(table2array(data(:,17)) == '"no"',:);

% Get nr of rows (dimension is 1) in matrix.
balanced_size_subscribed = size(set_subscribed,1);
% Calculate number of unsubscribed samples to get desired 70/30 ratio.
balanced_size_unsubscribed = (70*balanced_size_subscribed)/30;
% Sample k observations uniformly at random, without replacement.
balanced_unsubscribed = datasample(set_unsubscribed, balanced_size_unsubscribed,...
                                   'Replace', false);
% Concatenate the subscribed and sampled unsubcribed matrices to create
% a 70/30 balanced dataset.
balanced_data = vertcat(set_subscribed, balanced_unsubscribed);
% Shuffle the rows of the matrix by doing a random permutation from
% 1 to the row count of the matrix.
balanced_data = balanced_data(randperm(size(balanced_data,1)),:);

% Create the empty matrix that will contain the normalized data.
normalized_data = zeros(size(balanced_data, 1),0);

% Normalization:
% Loop over all columns, column 17 (outcome) we do separately.
for i=1:16
    curArray = table2array(balanced_data(:,i));
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
curArray = table2array(balanced_data(:,17));
normalized_data = horzcat(normalized_data, curArray == '"yes"');

% TODO: finish code below this comment.

% Specify train, validation and test sets on the neural network.
% The data is divided by index, the indices for the three subsets are
% defined by the division parameters trainInd, valInd and testInd.
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:400; % 79% of data
net.divideParam.valInd = 401:450; % 10% of data
net.divideParam.testInd = 451:506; % 11% of data


nnInputs = normalized_data(:,1:16);
nnLabels = normalized_data(:,17); % we need to convert yes/no to 1/0

% Create ranges for different set sizes: train, validate, test
nncount = size(nnInputs,1);
trainRange = 1:(nncount*0.6); % 60%
validateRange = (floor(nncount*0.6)+1):floor(nncount*0.8); % 20%
testRange = (floor(nncount*0.8)+1):floor(nncount*1.0); % 20%

% Setup our Neural Network
nnet = patternnet([100 20], 'trainlm');
nnet.divideFcn = 'divideind';
nnet.divideParam.trainInd = trainRange;
nnet.divideParam.valInd = validateRange;
nnet.divideParam.testInd = testRange;

% train the neural network and get the output for the test set
[net, tr] = train(nnet, nnInputs, nnLabels);
output = nnet(nnInputs(:,testRange));

% Data set for Fuzzy Inference Model 

% TODO %



