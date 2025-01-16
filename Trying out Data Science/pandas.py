#PANDAS

import pandas as pd


#importing Boston House Price

from sklearn.datasets import load_boston

boston_dataset = load_boston()

type(boston_dataset)

print(boston_dataset)

#Pandas DataFrame

boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

print(boston_df)

print(boston_df.head())
print(boston_df.shape)




#LOAD CSV FILE INTO PANDAS DATAFRAME



diabetes_df = "/Users/dex/Desktop/Data Science Project/datascience/diabetes.csv"

data = pd.read_csv(diabetes_df)

print(type(data))
print(data.head())

#LOAD BOSTON FILE TO CSV FILE

boston_df.to_csv("Boston.csv")

#INSPECTING DATAFRAME

print(boston_df.shape)
print(boston_df.head())
print(boston_df.tail())

#INFORMATION ABOUT THE DATAFRAME(BOSTON)

print(boston_df.info())


#FINDING THE NUMBER OF MISSING VALUES

bd = boston_df.isnull().sum()
print(bd)

#DIABETES DATA
print(data.head())

#Where 1 represent diabetic and 0 represent non-diabetic

print(data.value_counts("Outcome"))


#Group the values based on the mean

print(data.groupby("Outcome").mean())

#STATISTICAL MEASURES

print(boston_df.count())
print(boston_df.mean())
print(boston_df.std())
print(boston_df.min())
print(boston_df.max())

print(boston_df.describe())


#MANIPULATING A DATASET
    #Adding a data column
    
boston_df["Price"] = boston_dataset.target
print(boston_df.head())


#REMOVING A ROW/COLUMN
    #ROW
print(boston_df.drop(index = 1, axis = 0))

    #COLUMN
print(boston_df.drop(columns= "ZN", axis = 1))
      

#LOCATING A ROW USING INDEX

print(boston_df.iloc[2])

#LOCATING A PARTICULAR COLUMN

print(boston_df.iloc[:,0])
print(boston_df.iloc[:,1])


#CORRELATION

print(boston_df.corr())
























