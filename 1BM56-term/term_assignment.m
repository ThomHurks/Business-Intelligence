% Authors: Thom Hurks and Vincent de Vos
% Group 27, 1BM56 Business Intelligence, <date>, 2015.

% For reproducibility:
rng(0,'twister');
% data is the input table, provided by the teachers, it's inside 1BM56-term.mat.
load('1BM56-term.mat');
% Get the unique values in each column, so they can be evaluated.
unique_columns = cell(17,1);
for i=1:17
unique_columns(i) =  {unique(data(:,i))};
end;
% Get the nr of missing values in each column, so they can be evaluated.
missing_columns = cell(17,3);
for i=1:17
    curArray = table2array(data(:,i));
    if isnumeric(curArray)
        missing_columns(i,1) =  {sum(isnan(curArray))};
    end;
    missing_columns(i,2) =  {sum(strncmp('unknown',curArray, 7))};
    if iscategorical(curArray)
        missing_columns(i,3) =  {sum(isundefined(curArray))};
    end;
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