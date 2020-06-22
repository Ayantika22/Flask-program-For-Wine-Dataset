# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:03:12 2020

@author: Ayantika
"""

# Importing necessary libraries
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
Wine = pd.read_csv("Wine.csv")
print(Wine.head())
#iris.drop("Id", axis=1, inplace = True)
y = Wine['Customer_Segment']
Wine.drop(columns='Customer_Segment',inplace=True)
X = Wine[['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids', 
          'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']]

# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb'))
