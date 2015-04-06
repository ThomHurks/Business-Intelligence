% Authors: Thom Hurks and Vincent de Vos
% Group 27, 1BM56 Business Intelligence, april 6, 2015.

% ANALYSIS OF TERM ASSIGNMENT DATA.

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