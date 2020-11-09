# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:59:20 2020

@author: Oscar
"""
import numpy as np
from neuralfun import logsig, ilogsig, dlogsig, relu, irelu, drelu

#########################################################################################            
# Syntax:
# ------
# w = onelayer_reg(W,x,f)
#
# Parameters of the function:
# --------------------------
# X : inputs of the network (size: m x n).
# d : desired outputs for the given inputs.
# finv : inverse of the activation function. 
# fderiv: derivative of the activation function.
# lam : regularization term (lambda)
#
# Returns:
# -------
# Optimal weights (w) of the network
    
def onelayer_reg(X,d,finv,fderiv,lam):
    
    # Number of data points (n)
    n = np.size(X,1);
    
    # The bias is included as the first input (first row)
    XX = np.insert(X, 0, np.ones(n), axis=0);
    
    # Inverse of the neural function
    f_d = eval(finv)(d);
    
    # Derivate of the neural function
    derf = eval(fderiv)(f_d);
    
    # Diagonal matrix
    F = np.diag(derf);
    
    # Singular Value Decomposition of H
    H = XX @ F;        
    U, S, V = np.linalg.svd(H, full_matrices=False);
    
    I = np.eye(np.size(S));
    S = np.diag(S);
    
    # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
    w = U @ (np.linalg.pinv(S*S+lam*I) @ (U.transpose() @ (XX @ (F @ (F @ f_d)))));

    return w;

#########################################################################################            
# Syntax:
# ------
# Output = nnsimul(W,x,f)
#
# Parameters of the function:
# --------------------------
# W : weights of the neural network (size: m+1 x 1). The 1st element is the bias.
# X : inputs of the network (matrix of size: m x n).
# f : neural function. 
# 
# Returns:
# -------
# Outputs of the network for all the input data.

def nnsimul(W,X,f):

    # Number of variables (m) and data points (n)
    m,n=X.shape;

    # Neural Network Simulation
    return eval(f)(W.transpose() @ np.insert(X, 0, np.ones(n), axis=0));
                        
#########################################################################################    