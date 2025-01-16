#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:21:11 2024

@author: dex
"""

import pandas as pd
import numpy as np

credit_data = pd.read_csv("/Users/dex/Downloads/credit_data.csv")
print(credit_data.head())

print(credit_data["Class"].value_counts())  #Highly imbalanced DataSet

#0 reps legit transactions
#1 reps fraudulent transactions

#Separating t e legit and fraudulent transactions

legit = credit_data[credit_data.Class == 0]
fraud = credit_data[credit_data.Class == 1]

print(legit.shape)
print(fraud.shape)


#UNDERSAMPLING
'''
Building a sample dataset containing similar distribution of legit 
& fraudulent transaction

No. of fraudulent --> 492

'''

legit_sample = legit.sample(n=492)

print(legit_sample.shape)

#CONCATINATE THE TWO DATAFRAMES

new_dataset = pd.concat([legit_sample, fraud], axis = 0)
print(new_dataset.shape)
print(new_dataset.head())
print(new_dataset["Class"].value_counts())

