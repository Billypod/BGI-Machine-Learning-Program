#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:30:27 2024

@author: dex
"""
#HANDLING MISSING VALUES
'''
Imputation
And Dropping
'''
    
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


placement_url = "/Users/dex/Desktop/Data Science Project/datascience/Placement_Dataset.csv"

data = pd.read_csv(placement_url)
'''
print(data.head())
print(data.shape)
print(data.isnull().sum())  #We find if there are empty columns

#We'll use imputation: replacing the null with statistical method
#Mean, Median or mode

#Salary column

fig, ax = plt.subplots(figsize=(15,15))
print(sns.displot(data.salary))

#Becuase of the right-skewedness of the data, we use "meDIan"
#to replACE 
'''
data["salary"].fillna(data["salary"].mean(), inplace= True)
print(data.isnull().sum())  #missing value has been sorted



