# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:35:19 2020

@author: de''
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
def loaddata(file):
    mat = sio.loadmat(file)
    x = mat['m5']
    y = mat['water']
    return x,y
def Plotspectrum(spec):
    x = np.arange(400, 400 + 2 * spec.shape[1], 2)
    for i in range(spec.shape[0]):
        plt.plot(x, spec[i, :], linewidth=0.6)
 
    fonts = 8
    plt.xlim(350, 2550)
    # plt.ylim(0, 1)
    plt.xlabel('Wavelength (nm)', fontsize=fonts)
    plt.ylabel('absorbance (AU)', fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)
    plt.tight_layout(pad=0.3)
    plt.grid(True)
    return plt
def D1(sdata):
    temp1 = pd.DataFrame(sdata)
    temp2 = temp1.diff(axis=1)
    temp3 = temp2.values
    return np.delete(temp3, 0, axis=1)
 
def MSC(s):
    n = s.shape[0] 
    k = np.zeros(s.shape[0])
    b = np.zeros(s.shape[0])
    M = np.mean(s, axis=0)#duilieqiujunzhi

    for i in range(n):
        y = s[i, :]
        y = y.reshape(-1, 1)#zhuanchengyilie
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_
 
    spec_msc = np.zeros_like(s)
    for i in range(n):
        bb = np.repeat(b[i], s.shape[1])
        kk = np.repeat(k[i], s.shape[1])
        temp = (s[i, :] - bb)/kk
        spec_msc[i, :] = temp
    return spec_msc
file = '/Users/sunjinyu/Desktop/huaxuejiliangxue/corn.mat'
X,y=loaddata(file)
X=MSC(X)
Plotspectrum(X)
# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=1)

pca = PCA()
X_train_reduced = pca.fit_transform(scale(X_train))

# 10-fold CV, with shuffle

kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)

regr = LinearRegression()
mse = []

# Calculate MSE using CV for the 19 principle components, adding one component at the time.
for i in np.arange(1, 20):
    score = -1*model_selection.cross_val_score(regr, X_train_reduced[:,:i], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
'''
plt.plot(mse)
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('water')
plt.xlim(xmin=-1);   
'''
#the lowest cross-validation error occurs when  M=6 components are used
pca2 = PCA()
# Scale the data
X_reduced_train = pca2.fit_transform(scale(X_train))

X_reduced_test = pca2.transform(scale(X_test))[:,:6]

# Train regression model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:6], y_train)
X_reduced_train = pca2.fit_transform(scale(X_train))[:,:6]
pred = regr.predict(X_reduced_test)
# Prediction with test data

y_pred = regr.predict(X_reduced_train)
print('R_squared:','%.4f'% r2_score(y_test,pred))
print('Q_squared:','%.4f'% r2_score(y_train,y_pred))
print('RMSEC:','%.4f'% mean_squared_error(y_test,pred))
print('RMSEP:','%.4f'% mean_squared_error(y_train,y_pred))
