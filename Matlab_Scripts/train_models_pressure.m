%% Script based on the python script of the same name to predict the rate
% of change of average matrix pressure using machine learning methods. 
% Purely out of laziness I will just process data files using the Python 
% workflow and the import them into Matlab. 

%% Import data
processed_data_set = readtable('../data/processed_diffusionData2D.csv', 'PreserveVariableNames', true);


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
X_tr = removevars(X_train, {'p_i', 'target'});
X_te = removevars(X_test, {'p_i', 'target'});

% Scaler
[X_tr_scaled, mu_x, sigma_x] = zscore(table2array(X_tr));
sigma_x(:,5) = inf; % change to ensure no NaNs
X_te_scaled = (table2array(X_te) - mu_x)./sigma_x;
[y_tr_scaled, mu_y, sigma_y] = zscore(table2array(y_train));
y_te_scaled = (table2array(y_test) - mu_y)./sigma_y;
scaler = struct('mu_x', mu_x, 'sigma_x', sigma_x, 'mu_y', mu_y, 'sigma_y', sigma_y);

% Ridge
pr_model = fitlm(X_tr_scaled, y_tr_scaled, 'quadratic', 'Intercept',false);
feature_model = pr_model.Formula.Terms(:, 1:6);
X_tr_poly = x2fx(X_tr_scaled, feature_model);
X_te_poly = x2fx(X_te_scaled, feature_model);
k = 1E-6; % scaling parameter
B = (X_tr_poly'*X_tr_poly +  diag(ones(1, size(X_tr_poly,2))*k)) \ (X_tr_poly'*y_tr_scaled);

% Ridge model 
rr_model = @(x) ridge_predictor(x, scaler, feature_model, B);
p_tr = rr_model(table2array(X_tr));
p_te = rr_model(table2array(X_te));

RMSE_tr = sqrt(mean((table2array(y_train) - p_tr).^2));
RMSE_te = sqrt(mean((table2array(y_test) - p_te).^2));


%% Results 
fprintf('\nLinear regression training RMSE: %f\n', RMSE_tr);
fprintf('\nLinear regression training RMSE: %f\n', RMSE_te);



