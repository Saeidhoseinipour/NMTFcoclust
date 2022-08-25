#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
PNMTF
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
#from initialization import random_init
from ..initialization import random_init
from ..io.input_checking import check_positive
#from input_checking import check_positive
from numpy.random import rand
from numpy import nan_to_num
from numpy import linalg
from datetime import datetime

# from pylab import *



class PNMTF:

	def __init__(self,
		 n_row_clusters = 2 , n_col_clusters = 2 , tau = 0, 
		 eta = 0, gamma = 0,
		 F_init = None, S_init = None, G_init = None,
		 max_iter = 100, n_init = 1, tol = 1e-9, 
		 random_state = None):
		self.n_row_clusters = n_row_clusters
		self.n_col_clusters = n_col_clusters
		self.tau = tau
		self.eta = eta
		self.gamma = gamma
		self.F_init = F_init
		self.S_init = S_init
		self.G_init = G_init
		self.max_iter = max_iter
		self.n_init = n_init
		self.tol = tol
		self.random_state = check_random_state(random_state)
		self.F = None
		self.G = None
		self.S = None
		self.row_labels_ = None
		self.column_labels_= None
		self.soft_matrix = None
		self.reorganized_matrix = None
		self.runtime = None
		self.rowcluster_matrix = None
		self.orthogonality_F = None
		self.orthogonality_G = None
		self.criterions = []
		self.criterion = -np.inf

	def fit(self, X, y=None):

		check_array(X, accept_sparse=True, dtype="numeric", order=None,
				copy=False, force_all_finite=True, ensure_2d=True,
				allow_nd=False, ensure_min_samples=self.n_row_clusters,
				ensure_min_features=self.n_col_clusters, estimator=None)
		criterion = self.criterion
		criterions = self.criterions
		row_labels_ = self.row_labels_
		column_labels_ = self.column_labels_

		X = X.astype(float)

		random_state = check_random_state(self.random_state) 
		seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)

		for seed in seeds:
			self._fit_single(X, seed, y)
			if np.isnan(self.criterion):   # c --> self.criterion
				raise ValueError("matrix may contain negative or unexpected NaN values")
			# remember attributes corresponding to the best criterion
			if (self.criterion > criterion): 
				criterion = self.criterion
				criterions = self.criterions
				row_labels_ = self.row_labels_
				column_labels_ = self.column_labels_
				runtime = self.runtime

		self.random_state = random_state

		# update attributes
		self.runtime = runtime
		self.criterion = criterion
		self.criterions = criterions
		self.row_labels_ = row_labels_ 
		self.column_labels_ = column_labels_ 



	def _fit_single(self, X, random_state = None, y=None):


		n, m = X.shape
		g = self.n_row_clusters
		s = self.n_col_clusters
		F = rand(n, g) if isinstance(self.F_init, type(None)) else self.F_init
		S = rand(g , s) if isinstance(self.S_init, type(None)) else self.S_init
		G = rand(m, s) if isinstance(self.G_init, type(None)) else self.G_init
		I_g = np.identity(g, dtype = None)
		I_s = np.identity(s, dtype = None)
		P_g = np.ones((g,g))-I_g
		P_s = np.ones((s,s))-I_s

		################   OPNMTF_alpha   ###################loop: MUR------> Multiplactive Update Rules
		change = True
		c_init = float(-np.inf)
		c_list = []
		runtime = []
		D_alpha_F_list = []
		D_alpha_G_list = []
		iteration = 0
		start_time = datetime.now()


		while change :
			change = False

			for itr in range(self.max_iter):
				if isinstance(self.F_init, type(None)):
					enum = X@G@S.T
					denom = F@S@G.T@G@S.T + self.tau*F@P_g
					F = F * ((enum/denom)**0.5)

				if isinstance(self.G_init, type(None)):
					enum = X.T@F@S
					denom = G@S.T@F.T@F@S + self.eta*G@P_s
					G = G * ((enum / denom)**0.5)
				
				if isinstance(self.S_init, type(None)):
					enum = F.T@X@G
					denom = F.T@F@S@G.T@G + self.gamma*S
					S = S * ((enum / denom)**0.5)

			DF = np.diagflat(F.sum(axis = 0))
			DG = np.diagflat(G.sum(axis = 0))

			#Normalization

			F = F@np.diagflat(np.power(F.sum(axis = 0), -1))
			S = DF@S@DG
			G = (np.diagflat(np.power(G.sum(axis = 0), -1))@G.T).T   #rank2*n
			soft_matrix = np.nan_to_num(F@S@G.T)
			F_cluster = np.zeros_like(F)
			F_cluster[np.arange(len(F)),np.argmax(F,axis=1)] = 1
			G_cluster = np.zeros_like(G)
			G_cluster[np.arange(len(G)),np.argmax(G,axis=1)] = 1
			
			#criterion PNMTF algorithm

			c = 0.5*np.trace((X - soft_matrix)@(X - soft_matrix).T) + 0.5*self.tau* np.trace(F@P_g@F.T) + 0.5*self.eta* np.trace(G@P_s@G.T) + 0.5*self.gamma*np.trace(S.T@S)

			
			iteration += 1
			if (np.abs(c - c_init)  > self.tol and iteration < self.max_iter): 
				c_init = c
				change = True
				c_list.append(c)
								
		end_time = datetime.now()		
		runtime.append(format(end_time - start_time))		

		self.max_iter = iteration
		self.runtime = runtime
		self.criterion = c
		self.criterions = c_list
		self.F = F_cluster
		self.G = G_cluster
		self.soft_matrix = F@S@G.T
		self.hard_matrix = F_cluster@S@G_cluster.T
		self.rowcluster_matrix = F_cluster@F_cluster.T@X
		self.reorganized_matrix = F.T@X@G
		self.row_labels_ = [x+1 for x in np.argmax(F, axis =1).tolist()]
		self.column_labels_ = [x+1 for x in np.argmax(G, axis =1).tolist()]
		self.orthogonality_F = D_alpha_F_list
		self.orthogonality_G = D_alpha_G_list


# In[ ]:





