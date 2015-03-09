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