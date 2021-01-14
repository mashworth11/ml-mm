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
base_data = pd.read_csv('../Data/diffusionData.csv')
