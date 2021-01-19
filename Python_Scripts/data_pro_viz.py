#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for processing and visualising MATLAB data
"""
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
from Utilities import D_creator

#%% Import and process data
base_data = pd.read_csv('../Data/diffusionData2D.csv')
data_obj = D_creator(base_data, 2, 0, 1E6)
data_set = data_obj.data_set
data_set['target'] = data_set['p_m']
target = data_set['target']
data_set = data_set.drop(['dp^(n)', 'dp^(n-1)', 'p_m'], axis = 1)
MATLAB_data_set = data_obj.D_4_MATLAB()
MATLAB_data_set.to_csv('../Data/processed_diffusionData2D.csv', index = False)


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
start = 0; stop = N;
for i in range(K):
    data = data_set['target'].iloc[start:stop].to_numpy()
    ax.semilogx(times, data, color = 'k', alpha = 0.5, linewidth = 1)
    start += N; stop += N
data_targets =  mlines.Line2D([], [], color = 'k', alpha = 0.5,
                              linewidth = 1, label='Test')

ax.semilogx(times, np.ones(len(times))*1E6, color = 'tab:green', alpha = 0.7, linewidth = 1.5)
f_pressure =  mlines.Line2D([], [], color = 'tab:green', alpha = 0.7,
                         linewidth = 1.5, label='Fracture pressure')

ax.set_xlim([min(times), 100]);  
ax.set_xlabel('Time (s)', fontsize = 12); 
ax.set_ylim([1.5E4, 1.05E6]);
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.legend(handles=[f_pressure, data_targets], fontsize = 8, loc = 'lower right')
plt.show()
# =============================================================================
# ax.axis('off')
# ax.tick_params(labelsize=9)
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(5.5/2.54, 5.5/2.54)
# plt.savefig('full_data.png', dpi = 600)
# =============================================================================

# visualisation - train/test split
blue = (0.1294, 0.4000, 0.6745)
red = (0.6980, 0.0941, 0.1686)
fig, ax = plt.subplots(figsize=(12, 7))
start = 0; stop = N
for i in range(K_train):
    ax.semilogx(times, X_train['target'].iloc[start:stop], color = blue, 
              linewidth = 1, alpha = 0.7)
    start += N; stop += N
train_targets =  mlines.Line2D([], [], color = blue, alpha = 0.7,
                              linewidth = 0.6, label='Train')

start = 0; stop = N
for i in range(K_test):
    ax.semilogx(times, X_test['target'].iloc[start:stop], color = red, 
              linewidth = 1, linestyle = '--', alpha = 0.7)
    start += N; stop += N
test_targets =  mlines.Line2D([], [], color = red, alpha = 0.7,
                              linewidth = 0.6, linestyle = '--', label='Test')

ax.semilogx(times, np.ones(len(times))*1E6, color = 'tab:green', alpha = 0.7, linewidth = 1.5)
f_pressure =  mlines.Line2D([], [], color = 'tab:green', alpha = 0.7,
                         linewidth = 1.5, label='Fracture pressure')

ax.set_xlim([min(times), 100])
ax.set_xlabel('Time (s)', fontsize = 12)
ax.set_ylim([1.5E4, 1.05E6]);
ax.set_ylabel('Pressure (Pa)', fontsize = 12)
ax.legend(handles=[f_pressure, train_targets, test_targets], fontsize = 8, loc = 'lower right')
plt.show()
# =============================================================================
# ax.axis('off')
# fig.tight_layout(pad=0)
# ax.figure.set_size_inches(5.5/2.54, 5.5/2.54)
# plt.savefig('train_test_data.png', dpi = 600)
# =============================================================================



