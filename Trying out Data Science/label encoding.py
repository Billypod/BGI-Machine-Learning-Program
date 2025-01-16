#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:23:09 2024

@author: dex
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

cancer_data = pd.read_csv("/Users/dex/Downloads/breast_cancer_data.csv")
print(cancer_data.head())
print(cancer_data.info())
#M represents advcanced while B represent curable


#Finding the count of different labels

print(cancer_data["diagnosis"].value_counts())

#Load the label encoder function

label_encode = LabelEncoder()

labels = label_encode.fit_transform(cancer_data.diagnosis)

#Appending the labels to the dataframe

cancer_data['target'] = labels

print(cancer_data.head())
# 0 represents Benign
# 1 represents Malignant cases

print(cancer_data["target"].value_counts())


#LABEL ENCODING FOR IRIS DATA
iris_data = pd.read_csv("/Users/dex/Downloads/iris_data.csv")

print(iris_data.head())

print(iris_data["Species"].value_counts())

label_encoder = LabelEncoder()

iris_label = label_encoder.fit_transform(iris_data.Species)

iris_data["target"] = iris_label

print(iris_data)







