# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:06:29 2022

@author: Kedar Pandya
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
#import the dataset
data =pd.read_csv('50_Startups.csv')

#%%
#check the null values
sns.heatmap(data.isnull(), yticklabels= False, cbar=False, cmap='YlGnBu') 

#%%
#seperating the values 
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
print('Dependent Variables=\n',X)
print('Independent Variable=\n',y)
#%%
#encdoing the dataset of X
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print('After encodind the dependent variables are:\n',X)
#%%
#splitting data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print('The training set for dependent variables =\n',X_train)
print('The test set for dependent variables =\n',X_test)
print('The training set for independent variables =\n',y_train)
print('The test set for dependent variables =\n',y_test)
#%%
#training the data
from sklearn.linear_model import LinearRegression
mlr= LinearRegression()
mlr.fit(X_train, y_train)
#%%
#predicting the test set values
y_pred = mlr.predict(X_test)
np.set_printoptions(precision=2)
print('Test set predicted results=\n',y_pred)
print('Test set real results=\n',y_test)
#to display the predictions with the real result side by side, we can also use the DataFrame 
#and create the DF with the two arrays
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1))

#%%
accuracy = round(mlr.score(X_train, y_train)*100,2)
print(accuracy)






















