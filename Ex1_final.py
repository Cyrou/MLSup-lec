#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:34:02 2018

@author: philippenaroz
"""

import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
from numpy import linalg as LA
dataX=pd.read_csv('/Users/philippenaroz/Desktop/datasets/data_set_1_X.csv')
dataY=pd.read_csv('/Users/philippenaroz/Desktop/datasets/data_set_1_Y.csv')
dataY=np.ravel(dataY)

print('taille dataX', len(dataX))
print('taille dataY', len(dataY))

#question 1

r=[]
e2=[]

for i in range(1,11):
    r.append(i*100)
    e=[]
    for a in range(100):
          X_train, X_test, y_train, y_test = train_test_split(dataX,dataY, train_size=0.001*i, random_state=None)
          clf = Perceptron(tol=None, random_state=None)
          clf.fit(X_train,y_train)
          Yres=clf.predict(X_test)
          sumVect=abs(Yres-y_test)/2
          error=np.mean(sumVect)
          e.append(error)
    e2.append(np.mean(e))

print(e2)
plt.figure    
plt.plot(r,e2,'ro')
plt.title('error function of m ')
plt.xlabel('size of the trainset m ')
plt.ylabel('average error rate')
plt.show


#question 2


indice = 0
e_limit = 1

while e_limit > 0.01 :
    indice = indice + 5 #on va de 500 en 500
    e=[]
    for a in range(100):
          X_train, X_test, y_train, y_test = train_test_split(dataX,dataY, train_size=0.001*indice, random_state=None )
          clf = Perceptron(tol=None, random_state=None)
          clf.fit(X_train,y_train)
          Yres=clf.predict(X_test)
          sumVect=abs(Yres-y_test)/2
          error=np.mean(sumVect)
          e.append(error)
    e_limit = np.mean(e)

print('m_limit =', indice * 100)


    
#question 3
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 

def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 

r=np.array([])
exec_time=np.array([])

for i in range(1,900,100):
    r=np.append(r, [i*100])
    start_time=time.perf_counter()
    for a in range(10):
          X_train, X_test, y_train, y_test = train_test_split(dataX,dataY, train_size=0.001*i, random_state=None )
          clf = Perceptron(tol=None, random_state=None)
          clf.fit(X_train,y_train)
          #Yres=clf.predict(X_test)
    end_time=time.perf_counter()
    exec_time=np.append(exec_time, [(end_time - start_time)/100])
 


b = estimate_coef(r, exec_time)
print("Estimated coefficients:\nb_0 = {}  \
      \nb_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
plot_regression_line(r, exec_time, b) 


#question 4 A FAIRE
