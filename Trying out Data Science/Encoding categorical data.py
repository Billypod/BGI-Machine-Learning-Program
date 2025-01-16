#ENCODING CATEGORICAL DATA
'''
In most ML algorithms, cat. data must be converted into a numerical format.
TWO WAYS:
    1. LABEL ENCODING: converts each category into a unique integer
    2. ONE-HOT ENCODING: creates binary columns for each category

'''

#LABEL ENCODING
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["species"] = iris.target


df["species"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})


#print(df["species"].unique())




encoder = LabelEncoder()

#Lets encode the species column

df["species_encoded"] = encoder.fit_transform(df["species"])

#print(df[["species", "species_encoded"]].drop_duplicates())

#FEATURE SCALING USING STANDARDIZATION OR NORMALIZATION

from sklearn.preprocessing import StandardScaler

numerical_features = df.drop(columns=["species", "species_encoded"]).columns

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df[numerical_features])

df_scaled = pd.DataFrame(df_scaled, columns=numerical_features)

df_final = pd.concat([df_scaled, df[["species", "species_encoded"]]], axis = 1)
#print(df_final.head())


#DIMENSIONALITY REDUCTION USING Principal Component Analysis

from sklearn.decomposition import PCA

pca = PCA(n_components= 0.95)

pca_components = pca.fit_transform(df_scaled)


df_pca = pd.DataFrame(pca_components, columns = [f'PC{i+1}' for i in range(pca_components.shape[1])])

df_pca = pd.concat(df[df_pca, df[["species", "species_encoded"]]], axis = 1)


print(df_pca.head())

print("Explained Variance ratio : ", pca.explained_variance_ratio_)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["species_encoded"], cmap="viridis", edgecolors="k", s=100)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - First Two Principal Component")

plt.colorbar(label = "Species")


plt.show()






