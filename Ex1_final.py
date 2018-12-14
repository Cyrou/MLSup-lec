#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:34:02 2018

@author: philippenaroz
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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
          X_train, X_test, y_train, y_test = train_test_split(dataX,dataY, train_size=0.001*(i), random_state=None, shuffle = True)
          clf = Perceptron(tol=0.001, random_state=0)
          clf.fit(X_train,y_train)
          Yres=clf.predict(X_test)
          sumVect=abs(Yres-y_test)/2
          error=np.mean(sumVect)
          e.append(error)
    e2.append(np.mean(e))

plt.figure    
plt.plot(r,e2,'ro')
plt.title('error function of m ')
plt.xlabel('size of the trainset m ')
plt.ylabel('average error rate')
plt.show