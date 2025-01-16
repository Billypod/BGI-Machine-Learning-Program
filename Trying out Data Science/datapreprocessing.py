#DATA PREPROCESSING
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["species"] = iris.target


df["species"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})

df.head()
print(df.head())
print(df.info())

#HANDLING MISSING DATA
    #Check for missing values
    #Handling missing values(if any)
    
missing_data = df.isnull().sum()        #Missing value per column

print("Missing data per column: \n", missing_data)

df_cleaned = df.dropna()


print(df.head())


#EXPLORATORY DATA ANALYSIS

print(df.describe())        #Display summary stats


#VISUALIZING FEATURE DISTRIBUTIONS
import matplotlib.pyplot as plt

df.hist(bins=20, figsize=(10,8), color ='skyblue')
plt.suptitle("Feature Distributions", fontsize = 15)
plt.show()

#PAIR PLOT
import seaborn as sns

sns.pairplot(df, hue="species", palette = "Set2")

plt.show()


#BOX PLOT

plt.figure(figsize=(12,3))

for i , column in enumerate(df.columns[:-1], 1):
    plt.subplot(2,2,i)
    sns.boxplot(data=df, x ="species", y =column)
    plt.title(f"Box plot of {column} by species")
    plt.tight_layout()
    plt.show()
    




















