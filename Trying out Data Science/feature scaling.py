#STANDARDIZATION: Transforms features to have a mean of 0 and a standard deviation of 1
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["species"] = iris.target


df["species"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})


#HANDLING MISSING DATA
    #Check for missing values
    #Handling missing values(if any)
    
missing_data = df.isnull().sum()        #Missing value per column


df_cleaned = df.dropna()



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = df.drop(columns = "species")
print("The standard columns are: \n",
      df_standardized)


df_standardized = scaler.fit_transform(df_standardized) #Appling standardization

df_standardized = pd.DataFrame(df_standardized, columns=df.drop(columns="species").columns)
print(df_standardized.head())                               
                             
#NOMALIZATION: Rescales the data to a fixed range, usually btw 0 and 1

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_normalized = df.drop(columns="species")

df_normalized = scaler.fit_transform(df_normalized)

df_normalized = pd.DataFrame(df_normalized, columns = df.drop(columns = "species").columns)

print(df_normalized.head())


df_encoded = pd.get_dummies(df,columns = ["species"])
'''
print(df_encoded)
'''




