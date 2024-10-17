import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


'Test loading a dataset from sklearn'

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)

print("Dataset Shape: ", df.shape)

print("First 5 Rows: ")
print(df.head())

sns.pairplot(df)
plt.plot()