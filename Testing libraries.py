'''D A Y   1   M A C H I N E  L E A R N I N G '''



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''

print("Hello World")




'DAY 1'
'Test loading a dataset from sklearn'

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)

print("Dataset Shape: ", df.shape)

print("First 5 Rows: ")
print(df.head())

sns.pairplot(df)
plt.plot()
'''
'DAY 2  D A T A   P R E P R O C E S S I N G'

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('titanic.csv')

print("Dataset Info: ")
df.info()

df["Age"].fillna(df["Age"].median(), inplace = True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace = True)

if "Cabin" in df.columns:
    df = df.drop(["Cabin"], axis =1)
    
df.drop_duplicates(inplace= True)

encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
encoder.fit_transform(df["Embarked"])

scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])


print("\nCleaned Dataset (First 5 Rows): ")
print(df.head())

D A Y  3   E X P L O R A T O R Y    D A T A   A N A L Y S I S'


# '''

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.impute import KNNImputer

# df = pd.read_csv("titanic.csv")

# print(df.describe())

# print("\nSurvival Count: ")
# print(df["Survived"].value_counts())




# plt.figure(figsize=(8,4))
# sns.histplot(df["Age"], bins = 30, kde = True)
# plt.title("Age Distribution")
# plt.show()

# plt.figure(figsize=(6,4))
# sns.countplot(data=df, x = "Sex", hue = "Survived")
# plt.title("Survival Count By Gender")
# plt.show()

# plt.figure(figsize=(8,4))
# sns.boxplot(data=df, x = "Pclass", y = "Fare")
# plt.title("Fare Distribution By Passenger Class")
# plt.show()

# 'Feature Engineering'

# imputer = KNNImputer(n_neighbors=5)

# df[["Age", "Fare"]] = imputer.fit_transform(df[["Age", "Fare"]])

# print(df[["Age", "Fare"]].isnull().sum())


# print(df.info())
# print(df.head())


# df = df.drop(["Cabin"], axis = 1)
# print(df.info())


"M O D E L  B U I L D I N G  (L O G I S T I C    R E G R E S S I O N)"
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate
X = df.drop("Survived", axis = 1)


df = df.drop(["Name", "Ticket"], axis = 1)

df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
print(df.head())
print(df.dtypes)

for column in df.columns:
    if df[column].dtypes == "object":
        print(f"Unique values in {column}: {df[column].unique()[:5]}")
        
X = df.drop("Survived", axis =1)
y = df["Survived"]


















