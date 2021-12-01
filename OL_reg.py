# -*- coding: utf-8 -*-
"""
Created on Mon Dic  1 17:40:00 2021

@author: Oscar
"""
import numpy as np
from neuralfun import logsig, ilogsig, dlogsig, relu, irelu, drelu, linear, ilinear, dlinear

#########################################################################################            
# Syntax:
# ------
# w, M, U, S = onelayer_reg(W,x,finv,fderiv,lam=0,Mk=None,Uk=None,Sk=None)
#
# Parameters of the function:
# --------------------------
# X : inputs of the network (size: m x n).
# d : desired outputs for the given inputs.
# finv : inverse of the activation function. 
# fderiv: derivative of the activation function.
# lam : regularization term (lambda)
# Mk (optional): Auxiliar matrix for incremental/distributed learning
# Uk (optional): Auxiliar matrix for incremental/distributed learning
# Sk (optional): Auxiliar matrix for incremental/distributed learning
#
# Returns:
# -------
# w: pptimal weights of the network
# M: Auxiliar matrix for incremental/distributed learning
# U: Auxiliar matrix for incremental/distributed learning
# S: Auxiliar matrix for incremental/distributed learning
    
def onelayer_reg(X,d,finv,fderiv,lam=0,Mk=None,Uk=None,Sk=None):
    
    # Number of data points (n)
    n = np.size(X,1);
    
    # The bias is included as the first input (first row)
    Xp = np.insert(X, 0, np.ones(n), axis=0);
    
    # Inverse of the neural function
    f_d = eval(finv)(d);
    
    # Derivate of the neural function
    derf = eval(fderiv)(f_d);
    
    # Diagonal matrix
    F = np.diag(derf);    

    # Matrix on which the Singular Value Decomposition will be calculated later    
    H = Xp @ F;        
    
    # If all the input matrices are empty
    if Mk is None and Uk is None and Sk is None:         
        # Singular Value Decomposition of H
        U, S, _ = np.linalg.svd(H, full_matrices=False);
        M = Xp @ (F @ (F @ f_d))
    elif Mk is not None and Uk is not None and Sk is not None:
        M = Mk + Xp @ (F @ (F @ f_d))
        Up, Sp, _ = np.linalg.svd(H, full_matrices=False);                
        U, S, _ = np.linalg.svd(np.concatenate((Uk @ Sk,Up @ np.diag(Sp)),axis=1), full_matrices=False);        
    else:
        print('Error: All the input matrices (Mk,Uk,Sk) must be all None or all not None');
        return

    I = np.eye(np.size(S));
    S = np.diag(S);
    
    # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
    w = U @ (np.linalg.pinv(S*S+lam*I) @ (U.transpose() @ M));

    return w, M, U, S;

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