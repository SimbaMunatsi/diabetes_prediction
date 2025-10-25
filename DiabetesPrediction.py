# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 07:44:24 2025

@author: HP
"""

# Import Pandas, Sequential and Dense from Keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Read the csv file
diabetes = pd.read_csv('diabetes.csv')
diabetes.isnull().sum(axis=0)