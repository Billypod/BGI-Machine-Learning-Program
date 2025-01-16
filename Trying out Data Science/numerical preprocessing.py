#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:50:06 2024

@author: dex
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv("/Users/dex/Downloads/diabetes.csv")

print(diabetes.head())
X = diabetes.drop(columns = "Outcome", axis = 1)
Y = diabetes["Outcome"]
print(X)
print(Y)

'''
0 --> Reps Non-diabetic
1 --> Reps Diabetic

'''

scaler = StandardScaler()

Standardized = scaler.fit_transform(X)
print(Standardized)