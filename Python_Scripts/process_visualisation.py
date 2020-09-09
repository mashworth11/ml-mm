#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for processing MATLAB data and some data viz.
"""
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from Utilities import D_creator
from Utilities import msa_outer_loop
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


#%% Import and arrange data
base_data = pd.read_csv('../Data/diffusionData.csv')
data_obj = D_creator(base_data, 2, 3, 1E6)
data_set = data_obj.data_set
target = data_set.target
MATLAB_data_set = data_obj.D_4_MATLAB()
MATLAB_data_set.to_csv('../Data/processed_data_set.csv', index = False)


#%% Split training and testing data, and visualise
times = np.unique(data_set['time']); N = len(times)  
p_i = np.unique(data_set['p_i']); K = len(p_i)
p_i_train = p_i[np.arange(0,len(p_i))%3 != 1]; K_train = len(p_i_train)
p_i_test = p_i[np.arange(0,len(p_i))%3 == 1]; K_test = len(p_i_test)
X_train = pd.DataFrame(columns = list(data_set.columns))
X_test = pd.DataFrame(columns = list(data_set.columns))

for p in p_i_train:
   X_train = X_train.append(data_set.loc[np.round(data_set['p_i'], 7) 
                                           == np.round(p, 7)], ignore_index = False)
y_train = target.loc[X_train.index]
for p in p_i_test:
   X_test = X_test.append(data_set.loc[np.round(data_set['p_i'], 7) 
                                           == np.round(p, 7)], ignore_index = False)
y_test = target.loc[X_test.index]

# visualisation - full data set
fig, ax = plt.subplots(figsize=(12, 7))
start = 80000; stop = 81000;
for i in range(K-80):
    data = data_set['target'].iloc[start:stop].to_numpy()
    ax.loglog(times, data, color = 'k',  marker = 'x', markersize = 5, alpha = 0.2)
    start += N; stop += N
ax.set_xlim([1E-1, 100]);  
ax.set_xlabel('Time (s)', fontsize = 12); 
ax.set_ylim([1E0, 1E6]);
ax.set_ylabel(r'$\Delta\overline{p}_m/\Delta t$ (Pa$\cdot$s$^{-1})$', fontsize = 12)
# =============================================================================
# ax.axis('off')
# ax.tick_params(labelsize=9)
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(5/2.54,4.5/2.54)
# plt.savefig('full_data.png', dpi = 600)
# =============================================================================

# visualisation - train/test split
blue = (0.1294, 0.4000, 0.6745)
red = (0.6980, 0.0941, 0.1686)
fig, ax = plt.subplots(figsize=(12, 7))
start = 0; stop = N
for i in range(K_train):
    ax.loglog(times, X_train['target'].iloc[start:stop], color = blue, 
              marker = 'x', markersize = 4, alpha = 0.5)
    start += N; stop += N
train_targets =  mlines.Line2D([], [], color = blue, marker='x', alpha = 0.5,
                              markersize=4, label='Train')

start = 0; stop = N
for i in range(K_test):
    ax.loglog(times, X_test['target'].iloc[start:stop], color = red, 
              fillstyle = 'none', marker = 'o', markersize = 4, alpha = 0.5)
    start += N; stop += N
test_targets =  mlines.Line2D([], [], color = red, marker='o', alpha = 0.5,
                              fillstyle = 'none', markersize=4, label='Test')
ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([10E0, 1E6])
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.tick_params(labelsize=14)
ax.legend(handles=[train_targets, test_targets], fontsize = 12, loc = 'upper right')
# =============================================================================
# ax.axis('off')
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(11/2.54,11/2.54)
# plt.savefig('train_test_data.png', dpi = 600)
# =============================================================================


#%% Train, test one-step ahead mode - Linear regression
pr_model = LinearRegression()
#Ridge_model = Ridge(alpha = 0.001, fit_intercept = 0)
X_tr = X_train.drop(['time', 'dt^(n+1)', 'p_i', 'p_m', 'target', 'p^(n-2)', 'p^(n-1)', 'p^(n)'], axis = 1)
X_tr = X_tr.reindex(['diff_p^(n-2)', 'dp^(n-1)', 'diff_p^(n-1)', 'dp^(n)', 'diff_p^(n)'], axis=1)
X_te = X_test.drop(['time', 'dt^(n+1)', 'p_i', 'p_m', 'target', 'p^(n-2)', 'p^(n-1)', 'p^(n)'], axis = 1)
X_te = X_te.reindex(['diff_p^(n-2)', 'dp^(n-1)', 'diff_p^(n-1)', 'dp^(n)', 'diff_p^(n)'], axis=1)
poly = PolynomialFeatures(2, include_bias = False).fit(X_tr)

## scalers
sc_x = StandardScaler()
X_tr_scaled = sc_x.fit_transform(X_tr)
X_te_scaled = sc_x.transform(X_te)
sc_y = StandardScaler()
y_tr_scaled = sc_y.fit_transform(y_train.to_numpy().reshape((-1,1)))
y_te_scaled = sc_y.transform(y_test.to_numpy().reshape((-1,1)))
scaler = {'sc_x':sc_x, 'sc_y':sc_y}

## poly features
X_tr_poly = pd.DataFrame(poly.transform(X_tr), columns = poly.get_feature_names(X_tr.columns))
X_te_poly = pd.DataFrame(poly.transform(X_te), columns = poly.get_feature_names(X_te.columns))

## fitting 
pr_model.fit(X_tr_poly, y_train)
MATLAB_pr_coeffs = pd.read_csv('../Data/MATLAB_coefficients.csv')
newindex = [0,1,2,3,4,15,5,6,7,8,16,9,10,11,17,12,13,18,14,19]
MATLAB_pr_coeffs = MATLAB_pr_coeffs.reindex(newindex)
MATLAB_pr_coeffs = MATLAB_pr_coeffs.Estimate.to_numpy()
pr_model.coef_ = MATLAB_pr_coeffs
#Ridge_model.fit(X_tr_poly, y_tr_scaled)
#dp_tr = sc_y.inverse_transform(Ridge_model.predict(X_tr_poly))
dp_tr = pr_model.predict(X_tr_poly)
RMSE_tr = np.sqrt(mean_squared_error(y_train, dp_tr))
#dp_te = sc_y.inverse_transform(Ridge_model.predict(X_te_poly))
dp_te = pr_model.predict(X_te_poly)
RMSE_te = np.sqrt(mean_squared_error(y_test, dp_te))
print(f'Train SSA RMSE: {RMSE_tr}, Test SSA RMSE: {RMSE_te}')


#%% Test multi-step ahead mode
init_test_inputs = X_te[['diff_p^(n-2)', 'dp^(n-1)', 'diff_p^(n-1)', 'dp^(n)', 'diff_p^(n)']].to_numpy()
init_test_inputs = init_test_inputs[0::N]
dp_msa = msa_outer_loop(pr_model, init_test_inputs, 0.1, N, poly)
RMSE_msa = np.sqrt(mean_squared_error(y_test, dp_msa))
print(f'MSA RMSE: {RMSE_msa}')


#%% Visualise results
# training: one-step ahead
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title('Training: SSA', fontsize=14)
start = 0; stop = N
for i in range(K_train):
    ax.loglog(times, X_train['target'].iloc[start:stop], color = 'k', 
              marker = 'x', markersize = 4, alpha = 0.5)
    start += N; stop += N
train_targets =  mlines.Line2D([], [], color='k', marker='x', alpha = 0.5,
                          markersize=4, label='Target')
start = 0; stop = N
for i in range(K_train):
    ax.loglog(times, dp_tr[start:stop], color = blue, marker = 'o', 
              fillstyle = 'none', markersize = 4, alpha = 0.4)
    start += N; stop += N
prediction =  mlines.Line2D([], [], color = blue, marker='o', alpha = 0.4, 
                            fillstyle = 'none', markersize=4, label='Prediction')
ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([10E0, 1E6])
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.tick_params(labelsize=14)
ax.legend(handles=[train_targets, prediction], fontsize = 12, loc = 'lower left')
# =============================================================================
# ax.axis('off')
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(5/2.54,4.5/2.54)
# plt.savefig('training_ssp.png', dpi = 600)
# =============================================================================

# testing: one-step ahead
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title('Testing: SSA', fontsize=14)
start = 0; stop = N
for i in range(K_test):
    ax.loglog(times, X_test['target'].iloc[start:stop], color = 'k', 
              marker = 'x', markersize = 4, alpha = 0.5)
    start += N; stop += N
test_targets =  mlines.Line2D([], [], color='k', marker='x', alpha = 0.5,
                          markersize=4, label='Target')
start = 0; stop = N
for i in range(K_test):
    ax.loglog(times, dp_te[start:stop], color = 'tab:green', 
              marker = 'o', markersize = 4, alpha = 0.5, fillstyle = 'none')
    start += N; stop += N
prediction =  mlines.Line2D([], [], color='tab:green', marker='o', alpha = 0.5,
                          fillstyle = 'none', markersize=4, label='Prediction')
ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([10E0, 1E6])
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.tick_params(labelsize=14)
ax.legend(handles=[test_targets, prediction])

# testing: multi-step ahead
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title('Testing: MSA', fontsize=14)
start = 0; stop = N
for i in range(K_test):
    ax.loglog(times, X_test['target'].iloc[start:stop], color = 'k', 
              marker = 'x', markersize = 4, alpha = 0.6)
    start += N; stop += N
test_targets =  mlines.Line2D([], [], color='k', marker='x', alpha = 0.6,
                          markersize=4, label='Target')
start = 0; stop = N
for i in range(K_test):
    ax.loglog(times, dp_msa[start:stop], color = red, marker = 'o', 
              fillstyle = 'none', markersize = 4, alpha = 0.4)
    start += N; stop += N
prediction =  mlines.Line2D([], [], color = red, marker='o', alpha = 0.4, 
                            fillstyle = 'none', markersize=4, label='Prediction')
ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([10E0, 1E6])
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.tick_params(labelsize=14)
ax.legend(handles=[test_targets, prediction], fontsize = 12, loc = 'lower left')
# =============================================================================
# ax.axis('off')
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(5/2.54,4.5/2.54)
# plt.savefig('testing_msp.png', dpi = 600)
# =============================================================================
