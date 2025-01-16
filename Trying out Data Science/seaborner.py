from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd



iris = load_iris()


df = pd.DataFrame(data= iris.data, columns=iris.feature_names)
df["target"] = iris.target


df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

print(df.head())

print(df.info())


#SCATTER PLOT
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target', 'species']

df_checked = df.drop(columns="target", axis=1)

print(sns.scatterplot(x = "sepal length", y = "petal width", hue = "species", data= df_checked))






