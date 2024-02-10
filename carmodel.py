# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:54:03 2024

@author: KRISHNANGSHU
"""

import pandas as pd
import numpy as np


import pickle
car_df=pd.read_csv("car.csv",encoding='ISO-8859-1')


X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
y=car_df['Car Purchase Amount']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
scaler.data_max_
scaler.data_min_
y=y.values.reshape(-1,1)
y_scaled=scaler.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(40, input_dim=5, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error')
epochs_hist=model.fit(X_train,y_train,epochs=100,batch_size=25,verbose=1,validation_split=0.2)
epochs_hist.history.keys()
# gender=int(input("Enter Gender/// Male(1),Fmale(0)"))
# age=int(input("Enter age"))
# salary=float(input("Enter Salary"))
# cred=float(input("Enter Credit Card Debt"))
# net=float(input("Enter Net Worth"))
# X_test=np.array([[gender,age,salary,cred,net]])
X_test=np.array([[1,45,1000,1500,50000]])
y_test=model.predict(X_test)
print("Expected Purchase Amoount",y_test)
pickle.dump(model,open('mod.pkl','wb'))
mod=pickle.load(open('mod.pkl','rb'))