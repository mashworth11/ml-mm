%% Script based on the python script of the same name to predict the rate
% of change of average matrix pressure using machine learning methods. 
% Purely out of laziness I will just process data files using the Python 
% workflow and the import them into Matlab. 

%% Import data
processed_data_set = readtable('../data/processed_diffusionData2D.csv', 'PreserveVariableNames', true);
% m = processed_data_set.time <= 10;
% processed_data_set = processed_data_set(m, :);

%% Split data
p_i = unique(processed_data_set.p_i);
p_i_train = p_i(mod(1:length(p_i), 3) ~= 2);
p_i_test = p_i(mod(1:length(p_i), 3) == 2);
X_train = cell2table(cell(0,9), 'VariableNames', processed_data_set.Properties.VariableNames);
y_train = cell2table(cell(0,1), 'VariableNames', {'target'});
X_test = cell2table(cell(0,9), 'VariableNames', processed_data_set.Properties.VariableNames);
y_test = cell2table(cell(0,1), 'VariableNames', {'target'});

for p = p_i_train
    X_train = [X_train;processed_data_set(ismember(...
                                            round(table2array(processed_data_set(:,{'p_i'})),7),... 
                                            round(p,7)),:)];
    y_train = [y_train;processed_data_set(ismember(...
                                            round(table2array(processed_data_set(:,{'p_i'})),7),...
                                            round(p,7)),{'target'})]; 
end

for p = p_i_test
    X_test = [X_test;processed_data_set(ismember(...
                                            round(table2array(processed_data_set(:,{'p_i'})),7),...
                                            round(p,7)),:)];
    y_test = [y_test;processed_data_set(ismember(...
                                            round(table2array(processed_data_set(:,{'p_i'})),7),...
                                            round(p,7)),{'target'})];
end


%% Train models 
% sort out data
X_tr = removevars(X_train, {'p_i', 'time', 'target'});
X_te = removevars(X_test, {'p_i', 'time', 'target'});

% scale data
[X_tr_scaled, mu_x, sigma_x] = zscore(table2array(X_tr));
X_te_scaled = (table2array(X_te) - mu_x)./sigma_x;
[y_tr_scaled, mu_y, sigma_y] = zscore(table2array(y_train));
y_te_scaled = (table2array(y_test) - mu_y)./sigma_y;
scaler = struct('mu_y', mu_y, 'sigma_y', sigma_y);

% import neural net and use model
nn_model = importKerasLayers('../data/nn_model.h5','ImportWeights',true);
nn_model(1) = sequenceInputLayer(6, 'Name', 'SequenceInputLayer',...
                                    'Normalization', 'zscore',...
                                    'Mean', mu_x',...
                                    'StandardDeviation', sigma_x');
lgraph = layerGraph(nn_model);                       
nn_model = assembleNetwork(lgraph);
p_nn_tr = predict(nn_model, table2array(X_tr)').*sigma_y+mu_y;
p_nn_te = predict(nn_model, table2array(X_te)').*sigma_y+mu_y;
RMSE_nn_tr = sqrt(mean((table2array(y_train) - p_nn_tr').^2));
RMSE_nn_te = sqrt(mean((table2array(y_test) - p_nn_te').^2));


%% Results 
fprintf('\nNeural network training RMSE: %f\n', RMSE_nn_tr);
fprintf('\nNeural network testing RMSE: %f\n', RMSE_nn_te);



