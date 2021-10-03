#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ONMTF
"""

# Author: Hoseinipour Saeid <saeidhoseinipour@aut.ac.ir>       

# License: ??????????

import itertools
from math import *
from scipy.io import loadmat, savemat
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array
#from coclust.utils.initialization import (random_init, check_numbers,check_array)
# use sklearn instead FR 08-05-19
from initialization import random_init
from input_checking import check_positive
from numpy.random import rand
from numpy import nan_to_num
# from pylab import *




class ONMTF:

	def __init__(self,
				 n_row_clusters=2, n_col_clusters=2,  
				 F_init=None, S_init=None, G_init=None,
				 max_iter=100, n_init=1, tol=1e-9, 
				 random_state=None):
		
		self.n_row_clusters = n_row_clusters
		self.n_col_clusters=n_col_clusters
		self.F_init = F_init
		self.S_init = S_init
		self.G_init = G_init
		self.max_iter = max_iter
		self.n_init = n_init
		self.tol = tol
		self.random_state = check_random_state(random_state)
		self.row_labels_ = None
		self.column_labels_=None
		self.criterions = []
		self.criterion = -np.inf




	def fit(self, X, y=None):
		"""Perform clustering.
		Parameters
		----------
		X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
			Matrix to be analyzed
		"""

		check_array(X, accept_sparse=True, dtype="numeric", order=None,
					copy=False, force_all_finite=True, ensure_2d=True,
					allow_nd=False, ensure_min_samples=self.n_row_clusters,
					ensure_min_features=self.n_col_clusters, estimator=None)

		check_positive(X)

		criterion = self.criterion

		X = sp.csr_matrix(X)
		
		#X = X.astype(float)
		X=X.astype(int)

		random_state = check_random_state(self.random_state) 
		seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
		for seed in seeds:
			self._fit_single(X, seed, y)
			if np.isnan(self.criterion):
				raise ValueError("matrix may contain negative or unexpected NaN values")
			# remember attributes corresponding to the best criterion
			if (self.criterion > criterion): 
				criterion = self.criterion
				criterions = self.criterions
				row_labels_ = self.row_labels_
				column_labels_ = self.column_labels_
			  
		self.random_state = random_state
		
		# update attributes
		self.criterion = criterion
		self.criterions = criterions
		self.row_labels_ = row_labels_ 
		self.column_labels_ = column_labels_ 

		
	def _fit_single(self, X, random_state, y=None) :
		
		"""
		Orthogonal Subspace Non-negative Matrix Tri Factorization (OSNMTF) 

		Parameters
		----------
		X: array [m x n]
			Data matrix.
			
		rank1: int_number of row cluster
			Maximum rank of the row factor model.
		rank2: int_number of column cluster
			Maximum rank of the column factor model.
			
		
		   
		max_iter: int
			Maximum number of iterations.
			
		F_init: array [m x rank1]
			Fixed initial row basis matrix.
		S_init: array [rank1 x rank2]
			Fixed initial coefficient(summray) matrix.
		G_init: array [n x rank2]
			Fixed initial column basis matrix.

		Returns
		F: array [m x rank1]
			Coefficient matrix (row clustering).
		S: array [rank1 x rank2]
			Basis matrix (patterns).
		G:  array [n x rank2]
		   Coefficient matrix (column clustering).
		"""

		m, n = X.shape
		rank1 = self.n_row_clusters
		rank2 = self.n_col_clusters
		F = rand(m, rank1) if isinstance(self.F_init, type(None)) else self.F_init
		S = rand(rank1, rank2) if isinstance(self.S_init, type(None)) else self.S_init
		G = rand(n, rank2) if isinstance(self.G_init, type(None)) else self.G_init
		

		###################################loop: MUR------> Multiplactive Update Rules
		change = True
		c_init = float(-np.inf)
		c_list = []
		iteration = 0

		while change :
			change = False
				
			for itr in range(self.max_iter):
				if isinstance(self.F_init, type(None)):
					enum = X@G@S.T
					denom = F@S@G.T@X.T@F
					F = F * (enum/denom)

				if isinstance(self.G_init, type(None)):
					enum = X.T@F@S
					denom = G@S.T@F.T@X@G
					G = G * (enum / denom)
				
				if isinstance(self.S_init, type(None)):
					enum = F.T@X@G
					denom = F.T@F@S@G.T@G
					S = S * (enum / denom)

			#df DF and DG
			DF = np.nan_to_num(np.diag(F.sum(axis=0)**-.5))
			DG = np.nan_to_num(np.diag(G.sum(axis=0)**-.5)) #rank2*rank2

			#Normalization

			F = F@DF
			S = F.T@X@G
			G = G@DG  
			soft_matrix = F@S@G.T

			#criterion
	 
			c = np.trace((X - soft_matrix)@(X - soft_matrix).T)

			iteration += 1
			if (np.abs(c - c_init)  > self.tol and iteration < self.max_iter): 
				c_init=c
				change=True
				c_list.append(c)    

		

		self.max_iter = iteration
		self.criterion = c
		self.criterions = c_list
		self.row_labels_ = np.sort(np.argmax(F, axis =1)).tolist()
		self.column_labels_ = np.sort(np.argmax(G,axis=0)).tolist()

