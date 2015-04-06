% Authors: Thom Hurks and Vincent de Vos
% Group 27, 1BM56 Business Intelligence, april 6, 2015.

% For reproducibility:
rng(0,'twister');

% data is the input table, provided by the teachers, it's inside 1BM56-term.mat.
load('1BM56-term.mat');

% Get the unique values in each column, so they can be evaluated.
% Store categorical values separately for easy evaluation.
unique_columns = cell(17,2);
for i=1:17
    curTable = unique(data(:,i));
    unique_columns(i,1) = {curTable};
    curArray = table2array(curTable);
    if iscategorical(curArray)
        unique_columns(i,2) = {categorical(curArray)};
    end;
end;

% Get the nr of missing values in each column, so they can be evaluated.
missing_columns = cell(17,4);
for i=1:17
    curData = data(:,i);
    curArray = table2array(curData);
    % Count NaN's
    if isnumeric(curArray)
        missing_columns(i,1) =  {sum(isnan(curArray))};
        missing_columns(i,2) = {length(find(curArray==-1))};
    end;
    if iscategorical(curArray)
        % Count "unknown" entries.
        missing_columns(i,3) = {length(find(curArray=='"unknown"'))};
        % Count <undefined> entries
        missing_columns(i,4) =  {sum(isundefined(curArray))};
    end;
    % Count the total of missing values
    missing_values = { NaN, '', '<undefined>', '"unknown"' };
    missing_columns(i,5) = {sum(ismissing(data(:,i), missing_values))};
end;

% Get the value ranges in each column, so they can be evaluated.
range_columns = cell(17,2);
for i=1:17
    curArray = table2array(data(:,i));
    if isnumeric(curArray)
        range_columns(i,1) =  {min(curArray)};
        range_columns(i,2) =  {max(curArray)};
    end;
end;

% Question 2 - Balance data sets 
% Subscribed 30% vs Unsubscribed 70%

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
bdata = vertcat(set_subscribed, balanced_unsubscribed);
% Shuffle the rows of the matrix by doing a random permutation from
% 1 to the row count of the matrix.
bdata = bdata(randperm(size(bdata,1)),:);

% Normalize the data into vectors
ndata = cell(size(bdata));

% loop over every column
for i=1:17 
    % get all the row data for the column:c
    cdata = bdata(:,i);
    
    % TODO: alleen hebben we voor elke NaN value een entry in onze vector
    % deze kunnen we met het cleanen eruit halen dan zal het goed gaan
    udata = table2array(unique(cdata));
    usize = size(udata,1);
    
    % loop over every row (size 1) in the column
    for c=1:size(cdata,1)
        % create empty zeros vector
        vector = zeros(usize,1);
        % find index for value in the zero vector
        vindex = find(udata == cdata{c,1});
        % set this index 1
        vector(vindex) = 1;
        % store this vector in our new normalized data table
        % row:c and column:i 
        % vec2str = strtrim(cellstr(num2str(vector)));
        ndata(c,i) = { vector };
    end;
end;

% Q7 repare the data set, so that it is in the proper form for the neural
% network and fuzzy inference modeling.
% Data set for Neural Networks

% TODO: alles staat nu dus in een CellTable 
% Hoe gaan we deze genormalizeerde vector data in onze NN gebruiken?
nnInputs = ndata(:,1:16);
nnLabels = ndata(:,17); % we need to convert yes/no to 1/0

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



