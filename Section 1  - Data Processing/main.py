# libaraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#printing the data
# print(X)
# print(Y)

# taking care of missing data
# we take care of this either by removing that data, 
# or ignoring that data, or by replacing that data with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X) # printing the updated data

# encoding the categorical data
# encoding the independent variable -> country
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X) # printing the updated data

# encoding the dependent variable -> purchased
le = LabelEncoder(X[:, -1])
Y = le.fit_transform(Y)
#print(Y) # printing the updated data

# splitting the dataset into the training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)
# test_size is the percentage of the test set that is to be split from the dataset
# random_state is the seed value for the random number generator

# printing the split data
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# feature scaling always do after splitting the dataset
# feature scaling is done to make sure that all the features are in the same scale
# standardization and normalization are the two ways of feature scaling
# standardization = (x - mean(x)) / std(x)
# normalization = (x - min(x)) / (max(x) - min(x))

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])

# printing the scaled data
# print(X_train)
# print(X_test)
