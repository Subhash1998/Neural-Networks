#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:57:58 2019

@author: subhash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data pre-processing

#importing dataset
dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values

#Encoding geography and gender
from  sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


#Making an ANN

#importing keras library
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initialising ANN
classifier=Sequential()

#Input layer and first hiddden layer
classifier.add(Dense(activation="relu",input_dim=11,units=6,kernel_initializer="uniform"))
#dropout
classifier.add(Dropout(p=0.1))

#second hidden layer
classifier.add(Dense(activation="relu",units=6,kernel_initializer="uniform"))
#dropout
classifier.add(Dropout(p=0.1))

#output layer
classifier.add(Dense(activation="sigmoid",units=1,kernel_initializer="uniform"))

#compiling ANN
#adam :- stochastic gradient descent
#loss for categorical :- categorical_crossentropy
#loss for binary classifier :- binary_crossentropy 
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


#Fitting ANN to training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)


#Make the predicitions

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction > 0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Evaluating,Improving and tuning ANN

#Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(activation="relu",input_dim=11,units=6,kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu",units=6,kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid",units=1,kernel_initializer="uniform"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10 , n_jobs=1)

mean=accuracies.mean()
variance=accuracies.std()

#Improving ANN
#Dropout regularization for overfitting


#Tuning ANN

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(activation="relu",input_dim=11,units=6,kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu",units=6,kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid",units=1,kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size":[25,32],
              "nb_epoch":[100,500],
              "optimizer":['adam','rmsprop']
              }

grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_












