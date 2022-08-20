#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""SimulationLBM"""

# Author: Hoseinipour Saeid <saeidhoseinipour@aut.ac.ir> 


import numpy as np
from scipy.stats import multinomial
from scipy.stats import truncnorm
from scipy.stats import bernoulli
from scipy.stats import poisson
from itertools import zip_longest

class rvs:
   

    def __init__(self, M = 100, N = 100, K = 2, R = 2, pi = None, rho = None):

        self.M = M
        self.N = N
        self.K = K
        self.R = R
        self.pi = pi
        self.rho = rho
        self.Data_matrix_G = None
        self.Data_matrix_B = None
        self.Data_matrix_P = None
        self.Z_G = None
        self.Z_B = None
        self.Z_P = None
        self.W_G = None
        self.W_B = None
        self.W_P = None
        self.true_row_labels_G = None
        self.true_row_labels_B = None
        self.true_row_labels_P = None
        self.true_column_labels_G = None
        self.true_column_labels_B = None
        self.true_column_labels_P = None

    def TGaussian(self, mu, sigma):
    
        Z = np.zeros((self.M, self.K),dtype=int)
        W = np.zeros((self.N, self.R),dtype=int)

        for i,j in zip_longest(range(self.M), range(self.N)):
            if i is not None:
                Z[i,:] = multinomial.rvs(1, self.pi, random_state=None)
            if j is not None:
                W[j,:] = multinomial.rvs(1, self.rho, random_state=None)
        Z_G = np.zeros_like(Z)
        Z_G[np.arange(len(Z)),np.sort(np.argmax(Z,axis=1))] = 1
        W_G = np.zeros_like(W)
        W_G[np.arange(len(W)),np.sort(np.argmax(W,axis=1))] = 1
        nw = W.sum(axis = 0)
        nz = Z.sum(axis = 0)
        n = np.array([nz]).T@np.array([nw])
        column_list = list(range(self.K))
        for h in range(self.K):            
            row_list = list(range(self.R))
            for k in range(self.R):
                row_list[k] = np.reshape(truncnorm.rvs(a = 0, b = (100 - mu[h][k])/sigma[h][k], size = n[h][k]),(nz[h],nw[k])) 
            column_list[h] = np.concatenate([row_list[k] for k in range(self.R)], axis=1)
        Data_matrix_G = np.concatenate([column_list[h] for h in range(self.K)], axis = 0)   
        Data_matrix_G = np.random.permutation(Data_matrix_G)
        Data_matrix_G = np.random.permutation(Data_matrix_G.T)
        true_row_labels_G = [x+1 for x in np.sort(np.argmax(Z_G, axis =1)).tolist()]
        true_column_labels_G = [x+1 for x in np.sort(np.argmax(W_G, axis =1)).tolist()]
        self.Data_matrix_G = Data_matrix_G.T
        self.Z_G = Z_G
        self.W_G = W_G
        self.true_row_labels_G = true_row_labels_G
        self.true_column_labels_G = true_column_labels_G

    def Binary(self, gamma):
    
        Z = np.zeros((self.M, self.K),dtype=int)
        W = np.zeros((self.N, self.R),dtype=int)
        for i,j in zip_longest(range(self.M), range(self.N)):
            if i is not None:
                Z[i,:] = multinomial.rvs(1, self.pi, random_state=None)
            if j is not None:
                W[j,:] = multinomial.rvs(1, self.rho, random_state=None) 
        nw = W.sum(axis = 0)
        Z_B = np.zeros_like(Z)
        Z_B[np.arange(len(Z)),np.sort(np.argmax(Z,axis=1))] = 1
        W_B = np.zeros_like(W)
        W_B[np.arange(len(W)),np.sort(np.argmax(W,axis=1))] = 1
        nz = Z.sum(axis = 0)
        n = np.array([nz]).T@np.array([nw])
        column_list = list(range(self.K))
        for h in range(self.K):
            row_list = list(range(self.R))
            for k in range(self.R):
                row_list[k] = np.reshape(bernoulli.rvs(p = gamma[h][k], size = n[h][k]),(nz[h],nw[k]))

            column_list[h] = np.concatenate([row_list[k] for k in range(self.R)], axis=1)
             #must combain all elements row_list how? and convert to a array2D numpy 
        Data_matrix_B = np.concatenate([column_list[h] for h in range(self.K)], axis = 0)
        Data_matrix_B = np.random.permutation(Data_matrix_B)
        Data_matrix_B = np.random.permutation(Data_matrix_B.T)
        true_row_labels_B = [x+1 for x in np.sort(np.argmax(Z_B, axis =1)).tolist()]
        true_column_labels_B = [x+1 for x in np.sort(np.argmax(W_B, axis =1)).tolist()]
        self.Data_matrix_B = Data_matrix_B.T
        self.Z_B = Z_B
        self.W_B = W_B
        self.true_row_labels_B = true_row_labels_B
        self.true_column_labels_B = true_column_labels_B

    def Poisson(self, alpha, mu, nu):
    
        Z = np.zeros((self.M, self.K),dtype=int)
        W = np.zeros((self.N, self.R),dtype=int)
        for i,j in zip_longest(range(self.M), range(self.N)):
            if i is not None:
                Z[i,:] = multinomial.rvs(1, self.pi, random_state=None)
            if j is not None:
                W[j,:] = multinomial.rvs(1, self.rho, random_state=None) 
        nw = W.sum(axis = 0)
        Z_P = np.zeros_like(Z)
        Z_P[np.arange(len(Z)),np.sort(np.argmax(Z,axis=1))] = 1
        W_P = np.zeros_like(W)
        W_P[np.arange(len(W)),np.sort(np.argmax(W,axis=1))] = 1
        nz = Z.sum(axis = 0)
        n = np.array([nz]).T@np.array([nw])
        Data_matrix_P = np.zeros((self.M,self.N))
        landa = np.zeros((self.M,self.N))
        for h in range(self.K):            
            for k in range(self.R):
                for i in range(self.M):
                    for j in range(self.N):
                        landa[i][j] = mu[h]*nu[k]*alpha[h][k]
                        Data_matrix_P[i][j] = poisson.rvs(landa[i][j], size = 1)
        Data_matrix_P = np.random.permutation(Data_matrix_P)
        Data_matrix_P = np.random.permutation(Data_matrix_P.T)
        true_row_labels_P = [x+1 for x in np.sort(np.argmax(Z_P, axis =1)).tolist()]
        true_column_labels_P = [x+1 for x in np.sort(np.argmax(W_P, axis =1)).tolist()]
        self.Data_matrix_P = Data_matrix_P.T
        self.Z_P = Z_P
        self.W_P = W_P
        self.true_row_labels_P = true_row_labels_P
        self.true_column_labels_P = true_column_labels_P


#In[]