#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
OSNMTF
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




class OSNMTF:

	def __init__(self,
		 n_row_clusters = 2 , n_col_clusters = 2 , 
		 mu = 0, landa = 0, 
		 F_init = None, S_init = None, G_init = None,
		 max_iter = 100, n_init = 1, tol = 1e-9, 
		 random_state = None):
		self.n_row_clusters = n_row_clusters
		self.n_col_clusters = n_col_clusters
		self.mu = mu
		self.landa = landa
		self.F_init = F_init
		self.S_init = S_init
		self.G_init = G_init
		self.max_iter = max_iter
		self.n_init = n_init
		self.tol = tol
		self.random_state = check_random_state(random_state)
		self.row_labels_ = None
		self.column_labels_= None
		self.criterions = []
		self.criterion = -np.inf

	def fit(self, X, y=None):

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


		m, n = X.shape
		rank1 = self.n_row_clusters
		rank2 = self.n_col_clusters
		F = rand(m, rank1) if isinstance(self.F_init, type(None)) else self.F_init
		S = rand(rank1 , rank2) if isinstance(self.S_init, type(None)) else self.S_init
		G = rand(n, rank2) if isinstance(self.G_init, type(None)) else self.G_init
		I1 = np.identity(rank1, dtype = None)
		I2 = np.identity(rank2, dtype = None)
		A1 = np.ones((rank1, rank1))
		A2 = np.ones((rank2, rank2))
		E = np.ones((m, n))

		###################################loop: MUR------> Multiplactive Update Rules
		change = True
		c_init = float(-np.inf)
		c_list = []
		iteration = 0

		while change :
			change = False

			for itr in range(self.max_iter):
				if isinstance(self.F_init, type(None)):
					print("***")
					enum = (X/(F.dot(S).dot(G.T))).dot(G).dot(S.T)+F.dot(I1/(F.T.dot(F)))*2*self.landa
					print("***",enum.shape)
					#denom = E.dot(G).dot(S.T)+F.dot(A1)*2*self.landa
					
					denom = E@G@S.T + F@A1*2*self.landa    
					F = np.nan_to_num(F * (enum/denom))

				if isinstance(self.G_init, type(None)):
					enum = (X/(F.dot(S).dot(G.T))).T.dot(F).dot(S)+G.dot(I2/(G.T.dot(G)))*2*self.mu
					denom = (E.T).dot(F).dot(S)+G.dot(A2)*2*self.mu
					G = np.nan_to_num(G * (enum / denom))
				
				if isinstance(self.S_init, type(None)):
					enum = (F.T).dot(X/(F.dot(S).dot(G.T))).dot(G)
					denom = (F.T).dot(E).dot(G)
					S = np.nan_to_num(S * (enum / denom))

			#df DF and DG
			DF = np.nan_to_num(np.diag(F.sum(axis=0)**-.5))
			DG = np.nan_to_num(np.diag(G.sum(axis=0)**-.5)) #rank2*rank2

			#Normalization

			F = F.dot(DF)
			S = DF.dot(S).dot(DG)
			G = DG.dot(G.T)   #rank2*n
			soft_matrix = F.dot(S).dot(G)

			#criterion
			c_1 = np.sum(np.where(X != 0, X * np.log(X / soft_matrix), 0))
			c_2 = np.sum(np.where(F.T@F != 0, F.T@F * np.log(F.T@F /I1), 0))
			c_3 = np.sum(np.where(G@G.T != 0, G@G.T * np.log(G@G.T /I2), 0))
			c = c_1 + self.mu * c_2 + self.landa * c_3


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


# In[ ]:




