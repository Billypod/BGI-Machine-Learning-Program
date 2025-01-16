#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:22:05 2024

@author: dex
"""
#Standardization is the process of standardizing the data to a 
#common format and common range


import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Load dataset
dataset = sklearn.datasets.load_breast_cancer()
print(dataset)

#loading the data to pandas DATAFRAME

df = pd.DataFrame(dataset.data, columns = dataset.feature_names )
print(df.head())

print(df.shape)

X = df
Y = dataset.target

#Splitting the data into training and test dats

X_train, X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.2, random_state = 3 )
print(X.shape, X_train.shape, X_test.shape)

print(dataset.data.std())

scaler = StandardScaler()

scaler.fit(X_train)
X_train_standardized = scaler.transform(X_train)

print(X_train_standardized)

X_test_standardized = scaler.transform(X_test)

print(X_train_standardized.std())
print(X_test_standardized.std())






 