function p_m = ridge_predictor(X, scaler, feature_model, B)
% Function for the ridge regression predictor. 
% NOTE: Requires train_models_pressure.m to have been run already. 
% Inputs:
%   X - input vector
%   scaler - scaler for input vector and output prediction
%   feature model - poly. transform
%   B - ridge regression weights
% Returns: 
%   p_m - pressure prediction at t+1

X_scaled = (X - scaler.mu_x)./scaler.sigma_x;
X_poly = x2fx(X_scaled, feature_model);

p_m = (X_poly*B).*scaler.sigma_y + scaler.mu_y;

end

