#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ONMTF_alpha
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
from ..initialization import random_init
from ..io.input_checking import check_positive
from numpy.random import rand
from numpy import nan_to_num
from numpy import linalg
from datetime import datetime

# from pylab import *




class ONMTF:

	def __init__(self,
				 n_row_clusters=2, n_col_clusters=2, alpha = 0.5+1e-1, delta = 0, beta = 0, 
				 F_init=None, S_init=None, G_init=None,
				 max_iter=100, n_init=1, tol=1e-9, 
				 random_state=None):
		
		self.n_row_clusters = n_row_clusters
		self.n_col_clusters=n_col_clusters
		self.alpha = alpha
		self.delta = delta
		self.beta = beta
		self.F_init = F_init
		self.S_init = S_init
		self.G_init = G_init
		self.max_iter = max_iter
		self.n_init = n_init
		self.tol = tol
		self.random_state = check_random_state(random_state)
		self.row_labels_ = None
		self.column_labels_=None
		self.soft_matrix = None
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

		#check_positive(X)

		criterion = self.criterion
		criterions = self.criterions
		row_labels_ = self.row_labels_
		column_labels_ = self.column_labels_

		#X = sp.csr_matrix(X)
		
		X = X.astype(float)
		#X=X.astype(int)

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

		return self

		
	def _fit_single(self, X, random_state, y=None) :
		
		n, m = X.shape
		g = self.n_row_clusters
		s = self.n_col_clusters
		F = rand(n, g) if isinstance(self.F_init, type(None)) else self.F_init
		S = rand(g, s) if isinstance(self.S_init, type(None)) else self.S_init
		G = rand(m, s) if isinstance(self.G_init, type(None)) else self.G_init
		E_nm = np.ones((n, m))
		I_g = np.identity(g, dtype = None)
		I_s = np.identity(s, dtype = None)
		P_g = np.ones((g,g))-I_g
		P_s = np.ones((s,s))-I_s
		F_zero = np.zeros_like(F)
		G_zero = np.zeros_like(G)
		S_zero = np.zeros_like(S)


		##############  ONMTF_alpha  ################loop: MUR------> Multiplactive Update Rules
		change = True
		c_init = float(-np.inf)
		c_list = []
		runtime = []
		Orthogonal_F_list = []
		Orthogonal_G_list = []
		iteration = 0
		start_time = datetime.now()

		while change :
			change = False
				
			for itr in range(self.max_iter):
				if isinstance(self.F_init, type(None)):
					enum = np.power(X/(F@S@G.T), self.alpha)@G@S.T
					denom = E_nm@G@S.T + 2*self.delta*self.alpha*F@P_g
					DDF = np.power(enum/denom, 1/self.alpha)
					F = np.nan_to_num(np.multiply(F, DDF))

				if isinstance(self.G_init, type(None)):
					enum = np.power((X/(F@S@G.T)).T, self.alpha)@F@S
					denom = E_nm.T@F@S + 2*self.beta*self.alpha*G@P_s
					DDG = np.power(enum / denom, 1/self.alpha)
					G = np.nan_to_num(np.multiply(G, DDG))
				
				if isinstance(self.S_init, type(None)):
					enum = F.T@np.power(X/(F@S@G.T), self.alpha)@G
					denom = F.T@E_nm@G
					DDS = np.power(enum/denom, 1/self.alpha)
					S = np.nan_to_num(np.multiply(S, DDS))

			# DF = diag(1.T F) (Convert to Probability)

			DF = np.nan_to_num(np.diag(F.sum(axis=0)**-1))
			DG = np.nan_to_num(np.diag(G.sum(axis=0)**-1))

			# Normalization (probabilistic interpretation)

			F = F@DF
			S = F.T@X@G
			G = G@DG 
			
			# Hard and Soft matrix

			F_cluster = np.zeros_like(F)
			F_cluster[np.arange(len(F)),np.sort(np.argmax(F,axis=1))] = 1
			G_cluster = np.zeros_like(G)
			G_cluster[np.arange(len(G)),np.sort(np.argmax(G,axis=1))] = 1


			# Orthogonality

			Orthogonal_F = linalg.norm(F.T@F - I_g, 'fro')
			Orthogonal_G = linalg.norm(G.T@G - I_s, 'fro')

			# Criterion xi_ONMTF_{alpha}

			z = np.nan_to_num(F@S@G.T/X+self.tol, posinf=0)
			f_z = np.nan_to_num(self.alpha+(1-self.alpha)*z-np.power(z,1-self.alpha))
			
			c = np.sum(np.multiply(X,f_z)) + self.delta* np.trace(F@P_g@F.T) + self.beta* np.trace(G@P_s@G.T)

			end_time = datetime.now()
	 
			iteration += 1
			if (np.abs(c - c_init)  > self.tol and iteration < self.max_iter): 
				c_init=c
				change=True
				c_list.append(c)
				Orthogonal_F_list.append(Orthogonal_F)
				Orthogonal_G_list.append(Orthogonal_G)    

		
		self.max_iter = iteration
		self.criterion = c
		self.criterions = c_list
		self.runtime = runtime
		self.row_labels_ = [x+1 for x in np.sort(np.argmax(F, axis =1)).tolist()]
		self.column_labels_ = [x+1 for x in np.sort(np.argmax(G,axis=1)).tolist()]
		self.hard_matrix = F_cluster@S@G_cluster.T
		self.soft_matrix = F@S@G.T
		self.rowcluster_matrix = F_cluster@F_cluster.T@X
		self.orthogonality_F = Orthogonal_F_list
		self.orthogonality_G = Orthogonal_G_list

