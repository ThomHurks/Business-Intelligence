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
% array that we concatenated back in line 81.
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
save(strcat('T27_neural_net_', timestamp), 'net', 'train_record');

% Data set for Fuzzy Inference Model

% Variables from our original data set
% age - column 1
% education - column - 4
% balance - column 6
% mortgage - column 7
% loan - column 8
% campaign - column 13
% pdays - number of days since last contact - column 14
%

% Select the choosen variables for our FIS
% TODO: choose the correct column index from normalized data 
% fis_data_input = normalized_data(:,[1 4 6 7 8 13 14]);
fis_data_input = normalized_data(:,[1 4]);

% Make sure to add the actual labels as the last column 
fis_data_labels = table2array(data(:,17)) == '"yes"';
fis_data_train = horzcat(fis_data_input, fis_data_labels);

% For now select a subset for testing (large sets can take quite some time)
fis_data_train = fis_data_train(1:100,:);

% Generate FIS structure with linear membership functions
fis_linear = genfis1(fis_data_train, 2, 'gbellmf', 'linear');
fis = anfis(fis_data_train, fis_linear, size(fis_data_train));
gensurf(fis);

% For every input variable define a set of fuzzy rules 

index = 1;

% age - column 1 (#1)
fis_linear = addvar(fis_linear,'input','age',[18 95]); % the actual column range
fis_linear = addmf(fis_linear,'input',index,'young adult','gaussmf',[18 28]);
fis_linear = addmf(fis_linear,'input',index,'adult','gaussmf',[29 40]);
fis_linear = addmf(fis_linear,'input',index,'older adult','gaussmf',[41 60]);
fis_linear = addmf(fis_linear,'input',index,'old','gaussmf',[61 95]);

index = index + 1;

% education - column 4 (#2)
%fis_linear = addvar(fis_linear,'input','education',[0 20]); 
%fis_linear = addmf(fis_linear,'input',2,'low educated','gaussmf',[0 8]);
%fis_linear = addmf(fis_linear,'input',2,'med educated','gaussmf',[8 14]);
%fis_linear = addmf(fis_linear,'input',2,'highly educated','gaussmf',[14 18]);
%fis_linear = addmf(fis_linear,'input',2,'academic','gaussmf',[18 20]);
%index = index + 1;

% balance - column 6 (#3)
fis_linear = addvar(fis_linear,'input','y',[-8019 102127]);
fis_linear = addmf(fis_linear,'input',index,'very poor','gaussmf',[-8019 -5000]);
fis_linear = addmf(fis_linear,'input',index,'poor','gaussmf',[-5000 0]);
fis_linear = addmf(fis_linear,'input',index,'moderate','gaussmf',[1 5000]);
fis_linear = addmf(fis_linear,'input',index,'rich','gaussmf',[5001 10000]);
fis_linear = addmf(fis_linear,'input',index,'very rich','gaussmf',[10001 102127]);
index = index + 1;

% TODO
% mortgage - column 7
% loan - column 8
% campaign - column 13

% pdays - number of days since last contact - column 14
% input error als ik deze column (column 14 uit data variable) wil meenemen
%fis_linear = addvar(fis_linear,'input','contact',[0 871]);
%fis_linear = addmf(fis_linear,'input',index, 'days','gaussmf',[0 7]);
%fis_linear = addmf(fis_linear,'input',index, 'weeks','gaussmf',[8 30]);
%fis_linear = addmf(fis_linear,'input',index, 'months','gaussmf',[31 90]);
%fis_linear = addmf(fis_linear,'input',index, 'many months','gaussmf',[90 365]);
%fis_linear = addmf(fis_linear,'input',index, 'years','gaussmf',[365 871]);

fis_linear = addvar(fis_linear,'output','label', [0 1]);
fis_linear = addmf(fis_linear,'output',1,'no','gaussmf',[0 0]);
fis_linear = addmf(fis_linear,'output',1,'yes','gaussmf',[1 1]);

%plotmf(fis_linear, 'input', 1)
%plotmf(fis_linear, 'input', 2)
%plotmf(fis_linear, 'input', 3)
%plotmf(fis_linear, 'ouput', 1)

timestamp = strcat(datestr(clock,'yyyy-mm-dd-HHMM'),'m',datestr(clock,'ss'),'s');
save(strcat('T27_fuzzy_inference_', timestamp), 'fis', 'train_record');

display('Done!');