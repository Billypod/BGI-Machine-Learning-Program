#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:01:45 2024

@author: dex
"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv("/Users/dex/Downloads/diabetes.csv")
print(diabetes_data.head())

print(diabetes_data["Outcome"].value_counts())


#Separating the data and labels

X = diabetes_data.drop(columns = "Outcome", axis = 1)
Y = diabetes_data["Outcome"]
print(X)
print(Y)