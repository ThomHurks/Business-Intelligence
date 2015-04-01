% Authors: Thom Hurks and Vincent de Vos
% Group 27, 1BM56 Business Intelligence, <date>, 2015.

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
    end;
    if iscategorical(curArray)
        % Count "unknown" entries.
        missing_columns(i,2) = {length(find(curArray=='"unknown"'))};
        % Count <undefined> entries
        missing_columns(i,3) =  {sum(isundefined(curArray))};
    end;
    % Count the total of missing values
    missing_values = { NaN, '', '<undefined>', '"unknown"' };
    missing_columns(i,4) = {sum(ismissing(data(:,i), missing_values))};
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
set_subscribed = data(find(table2array(data(:,17)) == '"yes"'),:);
set_unsubscribed = data(find(table2array(data(:,17)) == '"no"'),:);

balanced_size_subscribed = size(set_subscribed,1);
balanced_size_unsubscribed = (70*balanced_size_subscribed)/30;
balanced_unsubscribed = datasample(set_unsubscribed, balanced_size_unsubscribed);

% Create a the balanced dataset where 30% is subscribed and 70% is unsubscribed data
bdata = vertcat(set_subscribed, balanced_unsubscribed);
% Shuffle our rows 
bdata = bdata(randperm(size(bdata,1)),:);

% Data set for Neural Networks
nninput = bdata(:,1:16);
nnlabels = bdata(:,17);
 
nnet = patternnet(17, 'trainlm');

% Data set for Fuzzy Inference Model 


