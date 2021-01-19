#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for the 'ML-enhanced multiscale modelling' project. Class
for preprocessing MATLAB data, and functions for performing the multi-step ahead  
prediction test.
"""
import numpy as np
import pandas as pd

class D_creator:
    """
    Class used to create data set, D, used for (autoregressive) training and 
    testing based on the data coming from the analytical solution generated in 
    MATLAB. 
    
    NOTE, there are some methods here not used in the final implementation,
    but were designed for experimentation. 
    
    Input:
        base_data - data coming from analytical solution
        d_y - number of internal sequence terms
        d_u - number of external sequence terms
        P_f - boundary pressure 
    Returns:
        data_set object containing (autoregressive) features and target for 
        learning and testing
    """
    def __init__(self, base_data, d_y, d_u, P_f):
        self.base_data = base_data
        self.data_set = base_data.copy()
        self.d_y = d_y
        self.d_u = d_u
        self.P_f = P_f
        self.N = len(np.unique(base_data.time.to_numpy())) # number of steps 
        self.update_D()
        
    def time_step(self):
        """
        Method to compute discrete time-step feature based on times in base_data.
        """
        times = np.unique(self.base_data.time.to_numpy())
        times = np.concatenate((np.array([0]), times))
        dt = np.diff(times)
        no_of_seq = int(self.base_data.shape[0]/self.N) # number of unique sequences 
        self.data_set['dt^(n+1)'] = np.tile(dt, no_of_seq)
        return 
    
    def pressure_seq(self):
        """
        Method to compute lagged pressure steps.
        """
        in_seq_m = self.base_data['p_m'].to_numpy().reshape((-1, self.N))
        out_seq_m = self.data_set['p_i'].to_numpy().reshape((-1, self.N))
        in_seq_f = np.ones(in_seq_m.shape)*1E6
        out_seq_f = self.data_set['p_i'].to_numpy().reshape((-1, self.N))
        
        for i in range(self.d_y+1):
            if i == 0:
                self.data_set['p^(n)'] = self.seq_shifter(in_seq_m, out_seq_m, i+1)
                self.data_set['p_f^(n)'] = in_seq_f.ravel()
            else:
                self.data_set[f'p^(n-{i})'] = self.seq_shifter(in_seq_m, out_seq_m, i+1)
                self.data_set[f'p_f^(n-{i})'] = self.seq_shifter(in_seq_f, out_seq_f, i)
        return
                
    def update_D(self):
        """
        Method to update our data set, D, with appropriate inputs and output 
        for the learning problem.
        """
        self.time_step()
        self.pressure_seq()
        self.data_set['target'] = (self.base_data['p_m']-self.data_set['p^(n)'])/self.data_set['dt^(n+1)']
        # internal vars.
        for i in range(self.d_u):
            if i == 0:
                self.data_set['diff_p^(n)'] = self.P_f - self.data_set['p^(n)']
            else:
                in_seq = self.data_set['diff_p^(n)'].to_numpy().reshape((-1, self.N))
                out_seq = np.zeros(self.data_set.shape[0]).reshape((-1, self.N))
                self.data_set[f'diff_p^(n-{i})'] = self.seq_shifter(in_seq, out_seq, i)
        # external vars.   
        in_seq = self.data_set['target'].to_numpy().reshape((-1, self.N))
        out_seq = np.zeros(self.data_set.shape[0]).reshape((-1, self.N))
        for i in range(self.d_y):
            if i == 0:
                self.data_set['dp^(n)'] = self.seq_shifter(in_seq, out_seq, i+1)
            else:
                self.data_set[f'dp^(n-{i})'] = self.seq_shifter(in_seq, out_seq, i+1)
        return
    
    @staticmethod
    def seq_shifter(in_seq, out_seq, shift_by):
        """
        Method to shift sequences according to the lagging parameter 'shift_by'
        for the autoregressive approach.
        Inputs:
            in_seq - 2D numpy array of sequences to be shifted
            out_seq - 2D numpy array that will have added shifted sequences
            shift_by - lagging parameter
        Returns:
            out_seq - 1D array of shifted sequences, ready to be added to pandas column
        """
        in_seq = in_seq[:, 0:(-shift_by)]
        out_seq = np.concatenate((out_seq[:, 0:shift_by], in_seq), axis=1)
        return out_seq.ravel()
    
    def D_4_MATLAB(self):
        """
        Method to prepare data_set for use in MATLAB.
        """
        MATLAB_D = self.data_set[self.data_set.columns[self.data_set.columns.isin(
                                ['p_i', 'p_f^(n-2)', 'p_f^(n-1)', 'p_f^(n)', 
                                 'p^(n-2)', 'p^(n-1)', 'p^(n)', 'target'])]]
        MATLAB_D = MATLAB_D.reindex(['p_i', 'p_f^(n-2)', 'p^(n-2)', 'p_f^(n-1)', 
                                     'p^(n-1)', 'p_f^(n)', 'p^(n)', 'target'], 
                                    axis = 1)
        return MATLAB_D
                    
        
#%% Multi-step ahead prediction functions        
def msa_inner_loop(model, x_init, N, poly = None, scaler = None):
    """
    Function to perform inner loop of the multi-step ahead prediction.
    Inputs:
        model  - trained model type e.g. neural net, polynomial regression etc.
        x_init - single intitial input
        N - number of steps
        poly - polynomial transformer object
        scaler - scaler object   
    Returns: 
        p - array of multi-step ahead predictions for a single starting point
    """ 
    x_i = x_init.copy()
    p = np.zeros(N)
    for j in range(N):
        input_ = x_i.reshape(1,-1)
        if poly != None and scaler != None:
            #import pdb; pdb.set_trace() 
            sc_input_ = scaler['sc_x'].transform(input_)
            sc_input_ = poly.transform(sc_input_)
            p[j] = scaler['sc_y'].inverse_transform(model.predict(sc_input_))  
        elif scaler != None:
            sc_input_ = scaler['sc_x'].transform(input_)
            p[j] = (scaler['sc_y'].inverse_transform(model.predict(sc_input_).reshape(1,-1))).ravel()
        else:
            try:
                input_ = poly.transform(input_)
                p[j] = model.predict(input_)
            except ValueError:
                break
        #import pdb; pdb.set_trace() 
        #x_i = np.concatenate((x_i[1:], np.array([1E6-p[j]]))) # diff pressures
        #x_i = np.concatenate((x_i[1:], np.array([p[j]]))) # only pressure
        x_i = np.concatenate((x_i[2:], np.array([1E6, p[j]]))) # with fracture pressure
    return p


def msa_outer_loop(model, X_init, no_steps, poly = None, scaler = None):
    """
    Function to run outer loop of the multi-step ahead prediction given.  
    Inputs:
        model  - trained model type e.g. neural net, polynomial regression etc.
        X_init - array of initial inputs
        no_steps - number of steps
        poly - polynomial transformer object
        scaler - scaler object       
    Returns: 
        predictions - array of multi-step ahead predictions
    """
    N = no_steps
    predictions = np.array([])
    n = 0 
    for x_init in X_init:
        predictions = np.append(predictions, msa_inner_loop(model, x_init, N, poly, scaler))
        print(f'{n}*--->{n+1}')
        n += 1
    return predictions 