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
from Utilities import msa_outer_loop
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


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
y_train = y_train.to_numpy().reshape((K_train, N, 1))
for p in p_i_test:
   X_test = X_test.append(data_set.loc[np.round(data_set['p_i'], 7) 
                                           == np.round(p, 7)], ignore_index = False)
y_test = target.loc[X_test.index]
y_test = y_test.to_numpy().reshape((K_test, N, 1))

# Encoder, decoder arrays
X_tr = X_train.reindex(['p^(n-2)', 'p^(n-1)', 'p^(n)', 'p_f^(n-2)', 'p_f^(n-1)', 'p_f^(n)'], axis=1).to_numpy()
X_te = X_test.reindex(['p^(n-2)', 'p^(n-1)', 'p^(n)', 'p_f^(n-2)', 'p_f^(n-1)', 'p_f^(n)'], axis=1).to_numpy()
X_tr_encoder = X_tr[np.arange(0, X_tr.shape[0], N), :].reshape((K_train, 3, 2))
X_te_encoder = X_te[np.arange(0, X_te.shape[0], N), :].reshape((K_test, 3, 2))
X_tr_decoder = X_tr[:, [2,5]]
X_tr_decoder[np.arange(0, X_tr.shape[0], N), :] = np.array([[0,0]])
X_tr_decoder = X_tr_decoder.reshape((K_train, N, 2))
X_te_decoder = X_te[:, [2,5]]
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
ed_model.fit([X_tr_encoder, X_tr_decoder], Y_train, epochs = 10)




