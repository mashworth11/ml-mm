%% Script to train polynomial regression model using processed data coming 
% from python. 

%% Import data
processed_data_set = readtable('./data/processed_data_set.csv', 'PreserveVariableNames', true);


%% Split data
p_i = unique(processed_data_set.p_i);
p_i_train = p_i(mod(1:length(p_i), 3) ~= 2);
p_i_test = p_i(mod(1:length(p_i), 3) == 2);
X_train = cell2table(cell(0,8), 'VariableNames', processed_data_set.Properties.VariableNames);
y_train = cell2table(cell(0,1), 'VariableNames', {'target'});
X_test = cell2table(cell(0,8), 'VariableNames', processed_data_set.Properties.VariableNames);
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
X_tr = removevars(X_train, {'p_i', 'p_m', 'target'});
X_te = removevars(X_test, {'p_i', 'p_m', 'target'});

% linear regression
pr_model = fitlm(table2array(X_tr), table2array(y_train), 'quadratic', 'Intercept',false); 
dp_tr = predict(pr_model, table2array(X_tr));
dp_te = predict(pr_model, table2array(X_te));
RMSE_tr = sqrt(mean((table2array(y_train) - dp_tr).^2));
RMSE_te = sqrt(mean((table2array(y_test) - dp_te).^2));


%% Results 
fprintf('Polynomial regression training RMSE: %f\n', RMSE_tr);
fprintf('Polynomial regression testing RMSE: %f\n', RMSE_te);


%% Save coefficients table
writetable(pr_model.Coefficients, '../Data/MATLAB_coefficients.csv', 'WriteRowNames',true);


