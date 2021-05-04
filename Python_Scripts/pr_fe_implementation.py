#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for polynomial regression using engineered features i.e. P_f - P_m. 
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


#%% Import and process data
base_data = pd.read_csv('../Data/diffusionData2D.csv')
data_obj = D_creator(base_data, 2, 3, 1E6)
data_set = data_obj.data_set
data_set['target'] = data_set['p_m']
target = data_set['target']
data_set = data_set.drop(['dp^(n)', 'dp^(n-1)', 'p_m'], axis = 1)
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


#%% Train, test one-step ahead mode - Linear regression
pr_model = LinearRegression()
X_tr = X_train.reindex([ 'diff_p^(n-2)', 'diff_p^(n-1)', 'diff_p^(n)'], axis=1)
X_te = X_test.reindex([ 'diff_p^(n-2)', 'diff_p^(n-1)', 'diff_p^(n)'], axis=1)
poly = PolynomialFeatures(2, include_bias = False).fit(X_tr)

## poly features
X_tr_poly = pd.DataFrame(poly.transform(X_tr), columns = poly.get_feature_names(X_tr.columns))
X_te_poly = pd.DataFrame(poly.transform(X_te), columns = poly.get_feature_names(X_te.columns))

## fitting 
pr_model.fit(X_tr_poly, y_train)
p_tr = pr_model.predict(X_tr_poly)
RMSE_tr = np.sqrt(mean_squared_error(y_train, p_tr))
p_te = pr_model.predict(X_te_poly)
RMSE_te = np.sqrt(mean_squared_error(y_test, p_te))
print(f'Train SSA RMSE: {RMSE_tr}, Test SSA RMSE: {RMSE_te}')


#%% Test multi-step ahead mode
init_test_inputs = X_te[['diff_p^(n-2)', 'diff_p^(n-1)', 'diff_p^(n)']].to_numpy()
init_test_inputs = init_test_inputs[0::N]
p_msa = msa_outer_loop(pr_model, init_test_inputs, N, poly, scaler = None, diffs = True)
RMSE_msa = np.sqrt(mean_squared_error(y_test, p_msa))
print(f'MSA RMSE: {RMSE_msa}')
 

#%% Visualise results
blue = (0.1294, 0.4000, 0.6745)
red = (0.6980, 0.0941, 0.1686)

# training: one-step ahead
fig, ax = plt.subplots(figsize=(12, 7))
#ax.set_title('Training: SSA', fontsize=14)
start = 0; stop = N
for i in range(K_train):
    ax.semilogx(times, X_train['target'].iloc[start:stop], color = 'k', 
                alpha = 0.5, linewidth = 1)
    start += N; stop += N
train_targets =  mlines.Line2D([], [], color='k', alpha = 0.5, linewidth = 1, 
                            label='Target')
start = 0; stop = N
for i in range(K_train):
    ax.semilogx(times, p_tr[start:stop], color = blue, alpha = 0.7, linewidth = 1, 
               linestyle = '--')
    start += N; stop += N
prediction =  mlines.Line2D([], [], color = blue, alpha = 0.7, linewidth = 1, 
                            linestyle = '--', label='Prediction')
ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([1.5E4, 1E6])
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.tick_params(labelsize=14)
ax.legend(handles=[train_targets, prediction], fontsize = 8, loc = 'lower right')
# =============================================================================
# ax.axis('off')
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(5.5/2.54, 5.5/2.54)
# plt.savefig('train_pr.png', dpi = 600)
# =============================================================================

# testing: one-step ahead
fig, ax = plt.subplots(figsize=(12, 7))
#ax.set_title('Testing: SSA', fontsize=14)
start = 0; stop = N
for i in range(K_test):
    ax.semilogx(times, X_test['target'].iloc[start:stop], color = 'k', 
                marker = 'x', markersize = 4, alpha = 0.5)
    start += N; stop += N
test_targets =  mlines.Line2D([], [], color='k', marker='x', alpha = 0.5,
                          markersize=4, label='Target')
start = 0; stop = N
for i in range(K_test):
    ax.semilogx(times, p_te[start:stop], color = 'tab:green', 
                marker = 'o', markersize = 4, alpha = 0.5, fillstyle = 'none')
    start += N; stop += N
prediction =  mlines.Line2D([], [], color='tab:green', marker='o', alpha = 0.5,
                          fillstyle = 'none', markersize=4, label='Prediction')
ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([1.5E4, 1E6])
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.tick_params(labelsize=14)
ax.legend(handles=[test_targets, prediction])

# testing: multi-step ahead
fig, ax = plt.subplots(figsize=(12, 7))
#ax.set_title('Testing: MSA', fontsize=14)
start = 0; stop = N
for i in range(K_test):
    ax.semilogx(times, X_test['target'].iloc[start:stop], color = 'k', 
                alpha = 0.5, linewidth = 1)
    start += N; stop += N
test_targets =  mlines.Line2D([], [], color='k', alpha = 0.5, linewidth = 1, 
                             label='Target')
start = 0; stop = N
for i in range(K_test):
    ax.semilogx(times, p_msa[start:stop], color = red, linewidth = 1, 
                linestyle = '--', alpha = 0.7)
    start += N; stop += N
prediction =  mlines.Line2D([], [], color = red, alpha = 0.7, linewidth = 1, 
                            linestyle = '--', label='Prediction')
ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([1.5E4, 1E6])
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.tick_params(labelsize=14)
ax.legend(handles=[test_targets, prediction], fontsize = 8, loc = 'lower right')
# =============================================================================
# ax.axis('off')
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(5.5/2.54, 5.5/2.54)
# plt.savefig('test_pr.png', dpi = 600)
# =============================================================================