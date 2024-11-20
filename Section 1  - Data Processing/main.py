# libaraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# #printing the data
# print(X)
# print(Y)

# taking care of missing data
# we take care of this either by removing that data, 
# or ignoring that data, or by replacing that data with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X) # printing the updated data

# encoding the categorical data
# encoding the independent variable -> country
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X) # printing the updated data

# encoding the dependent variable -> purchased
le = LabelEncoder(X[:, -1])
Y = le.fit_transform(Y)
#print(Y) # printing the updated data



