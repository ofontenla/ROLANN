# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:48:12 2020

@author: Oscar
"""

#########################################################################################            
#
# Example of use for a classification problem
#
#########################################################################################            

import numpy as np
from OL_reg import onelayer_reg, nnsimul

# Data set
DATASET = 'Prostate'; # Values: 'Prostate', 'AMLALL'
# Hyperparameter (regularization)
lamb = 1;
# Activation functions: 'logs' o 'rel' (Logistic or ReLu)
afun = 'logs'

if (afun == 'logs'):     # Logistic sctivation functions
    f      = 'logsig' 
    finv   = 'ilogsig'
    fderiv = 'dlogsig'
elif (afun == 'rel' ):  # ReLu sctivation functions
    f      = 'relu' 
    finv   = 'irelu'  
    fderiv = 'drelu'
    
# Load the Data set
if (DATASET == 'Prostate'):
    A = np.genfromtxt('./Data/' + DATASET + 'Train.csv', delimiter=',')
    B = np.genfromtxt('./Data/' + DATASET + 'Test.csv', delimiter=',')
elif (DATASET == 'AMLALL'):
    A = np.genfromtxt('./Data/' + DATASET + '_train.data', delimiter=',')
    B = np.genfromtxt('./Data/' + DATASET + '_test.data', delimiter=',')

C0 = 0.1  # Desired output for class 0
C1 = 0.9  # Desired output for class 1

X = A[:,:-1] # Exclude the last column
d = A[:,-1]  # Vector with the last column
d[d==0] = C0
d[d==1] = C1

Xtest = B[:,:-1]
dtest = B[:,-1]
dtest[dtest==0] = C0
dtest[dtest==1] = C1
dth = (C1 + C0)/2  # Threshold for the actual output
threshold = 0.5    # Threshold for output of the net
    
# Data normalization (z-score): mean 0 and std 1
#scaler = preprocessing.StandardScaler().fit(X);
#X = scaler.transform(X);
#Xtest = scaler.transform(Xtest);

X = X.transpose()
Xtest = Xtest.transpose()

# Number of data points (n)
n = np.size(X,1)
ntest = np.size(Xtest,1)

# Optimal weights: 
w = onelayer_reg(X,d,finv,fderiv,lamb)

# Results

y = nnsimul(w,X,f).transpose()
yTest = nnsimul(w,Xtest,f).transpose()

Aciertos = np.sum(np.logical_and(y>=threshold,d>=dth)) + np.sum(np.logical_and(y<threshold,d<dth))
AciertosTest = np.sum(np.logical_and(yTest>=threshold,dtest>=dth)) + np.sum(np.logical_and(yTest<threshold,dtest<dth))

print('Resultados para', DATASET, ' (lambda = ', lamb, ')')
print('------------------------------------')
print('Aciertos Train =', Aciertos, 'de', n)
print('Aciertos Test  =', AciertosTest, 'de', ntest)



