#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for testing encoder-decoder LSTM for the ML-based multiscale modelling data. 
"""
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from Utilities import D_creator
from Utilities import msa_ED_looper
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


#%% Import and process data for the LSTM implementation
base_data = pd.read_csv('../Data/diffusionData2D.csv')
data_obj = D_creator(base_data, 2, 0, 1E6)
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


# Scale inputs and targets
X_tr = X_train.reindex(['p^(n-2)', 'p_f^(n-2)', 'p^(n-1)', 'p_f^(n-1)', 'p^(n)', 'p_f^(n)'], axis=1).to_numpy()
X_te = X_test.reindex(['p^(n-2)', 'p_f^(n-2)', 'p^(n-1)', 'p_f^(n-1)', 'p^(n)', 'p_f^(n)'], axis=1).to_numpy()
sc_x = StandardScaler()
X_tr_scaled = sc_x.fit_transform(X_tr)
X_te_scaled = sc_x.transform(X_te)
sc_y = StandardScaler()
y_tr_scaled = sc_y.fit_transform(y_train.to_numpy().reshape((-1,1)))
y_tr_scaled = y_tr_scaled.reshape((K_train, N, 1)) # reshape
y_te_scaled = sc_y.transform(y_test.to_numpy().reshape((-1,1)))
y_te_scaled = y_te_scaled.reshape((K_test, N, 1)) # reshape


# Encoder, decoder arrays
X_tr_encoder = X_tr_scaled[np.arange(0, X_tr.shape[0], N), :].reshape((K_train, 3, 2))
X_te_encoder = X_te_scaled[np.arange(0, X_te.shape[0], N), :].reshape((K_test, 3, 2))
X_tr_decoder = X_tr_scaled[:, [2,5]]
X_tr_decoder[np.arange(0, X_tr.shape[0], N), :] = np.array([[0,0]])
X_tr_decoder = X_tr_decoder.reshape((K_train, N, 2))
X_te_decoder = X_te_scaled[:, [2,5]]
X_te_decoder[np.arange(0, X_te.shape[0], N), :] = np.array([[0,0]])
X_te_decoder = X_te_decoder.reshape((K_test, N, 2))


#%% Setup encoder-decoder LSTM
encoder_inputs = keras.layers.Input(shape = [None, 2])
decoder_inputs = keras.layers.Input(shape = [None, 2])

# encoder 
encoder = LSTM(20, return_state = True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder
decoder = LSTM(20, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(inputs = decoder_inputs, initial_state = encoder_states)
decoder_dense = Dense(1, activation='linear')
decoder_outputs = decoder_dense(decoder_outputs)

# define and train encoder-decoder model
ed_model = keras.Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs)
ed_model.compile(loss = 'mse', optimizer = Adam(lr = 0.01))
ed_model.fit([X_tr_encoder, X_tr_decoder], y_tr_scaled, epochs = 10, batch_size = 2)

# prediction on training sequence - treated as a singlestep-ahead prediction
p_ed_tr = sc_y.inverse_transform(ed_model.predict([X_tr_encoder, X_tr_decoder]).reshape((-1,1)))
RMSE_ed_tr = np.sqrt(mean_squared_error(y_train, p_ed_tr))
print(f'Training score across all timesteps: {RMSE_ed_tr}')


#%% Test multi-step ahead mode
# setup encoder model to output states given input sequence using trained encoder above
encoder_model = keras.Model(encoder_inputs, encoder_states) 

# create Tensorflow decoder objects for multistep-ahead prediction based on trained decoder above 
decoder_state_input_h = keras.layers.Input(shape=(20,))
decoder_state_input_c = keras.layers.Input(shape=(20,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h, state_c] 
decoder_outputs = decoder_dense(decoder_outputs)

# create decoder model from decoder objects
decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs,
                            [decoder_outputs] + decoder_states)

p_ed_msa = msa_ED_looper(X_te_encoder, np.zeros(y_te_scaled.shape), N, encoder_model, decoder_model)
p_ed_msa = sc_y.inverse_transform(p_ed_msa.reshape((-1,1)))
RMSE_ed_msa = np.sqrt(mean_squared_error(y_test, p_ed_msa))
print(f'encoder-decoder LSTM MSA RMSE: {RMSE_ed_msa}')


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
    ax.semilogx(times, p_ed_tr[start:stop], color = blue, alpha = 0.7, linewidth = 1, 
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
    ax.semilogx(times, p_ed_msa[start:stop], color = red, linewidth = 1, 
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



