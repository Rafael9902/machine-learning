#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 08:36:35 2024

@author: Rafael Ortiz
"""

# Pre processing template

## Import libraries
import numpy as np #math library for number processing
from matplotlib import pyplot as plt # graphic representation, draws
import pandas as pd # load datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# import the dataset
dataset = pd.read_csv('../data/Data.csv')

# independent variables
x = dataset.iloc[:,:-1].values

# dependent variable
y = dataset.iloc[:,3].values


# nan processing
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

# categorical data
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:,0])

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)

y = label_encoder_x.fit_transform(y)


# Split dataset in training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Variable scaling
scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)

x_test = scaler_x.transform(x_test)