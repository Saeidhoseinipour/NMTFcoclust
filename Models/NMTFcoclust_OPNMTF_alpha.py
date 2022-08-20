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
#from initialization import random_init
from ..initialization import random_init
from ..io.input_checking import check_positive
#from input_checking import check_positive
from numpy.random import rand
from numpy import nan_to_num
from numpy import linalg
from datetime import datetime

# from pylab import *



class OPNMTF:

	def __init__(self,
		 n_row_clusters = 2 , n_col_clusters = 2 , 
		 mu = 0, landa = 0, alpha = 1+1e-1,
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
		self.alpha = alpha
		self.random_state = check_random_state(random_state)
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

		self.random_state = random_state

		# update attributes
		self.criterion = criterion
		self.criterions = criterions
		self.row_labels_ = row_labels_ 
		self.column_labels_ = column_labels_ 

		return self


	def _fit_single(self, X, random_state = None, y=None):


		n, m = X.shape
		g = self.n_row_clusters
		s = self.n_col_clusters
		F = rand(n, g) if isinstance(self.F_init, type(None)) else self.F_init
		S = rand(g , s) if isinstance(self.S_init, type(None)) else self.S_init
		G = rand(m, s) if isinstance(self.G_init, type(None)) else self.G_init
		I_g = np.identity(g, dtype = None)
		I_s = np.identity(s, dtype = None)
		E_gg = np.ones((self.n_row_clusters, self.n_row_clusters))
		E_ss = np.ones((self.n_col_clusters, self.n_col_clusters))
		E_nm = np.ones((n, m))

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
					
					enum = np.power(X/(F@S@G.T), self.alpha)@G@S.T + 2*self.landa*F@np.power(I_g/(F.T@F), self.alpha)
					denom = E_nm@G@S.T + F@E_gg*2*self.landa
					DDF = np.power(enum/denom, 1/self.alpha)
					F = np.nan_to_num(np.multiply(F, DDF))

				if isinstance(self.G_init, type(None)):
					enum = np.power((X/(F@S@G.T)).T, self.alpha)@F@S + G@np.power(I_s/(G.T@G), self.alpha)*2*self.mu
					denom = E_nm.T@F@S+G@E_ss*2*self.mu
					DDG = np.power(enum / denom, 1/self.alpha)
					G = np.nan_to_num(np.multiply(G, DDG))
				
				if isinstance(self.S_init, type(None)):
					enum = F.T@np.power(X/(F@S@G.T), self.alpha)@G
					denom = F.T@E_nm@G
					DDS = np.power(enum/denom, 1/self.alpha)
					S = np.nan_to_num(np.multiply(S, DDS))

			DF = np.diagflat(F.sum(axis = 0))
			DG = np.diagflat(G.sum(axis = 0))





			#Normalization

			F = F@np.diagflat(np.power(F.sum(axis = 0), -1))
			S = DF@S@DG
			G = (np.diagflat(np.power(G.sum(axis = 0), -1))@G.T).T   #rank2*n
			F_cluster = np.zeros_like(F)
			F_cluster[np.arange(len(F)),np.sort(np.argmax(F,axis=1))] = 1
			G_cluster = np.zeros_like(G)
			G_cluster[np.arange(len(G)),np.sort(np.argmax(G,axis=1))] = 1
			
			#criterion alpha-divargance with convex function    f(z) = 1/alpha(1-alpha)   alpha+ (1-alpha)z - z^{1-alpha}

			z = np.nan_to_num(F@S@G.T +self.tol/X +self.tol, posinf=0)
			z_F = np.nan_to_num(F.T@F +self.tol/I_g +self.tol, posinf=0)
			z_G = np.nan_to_num(G.T@G +self.tol/I_s +self.tol, posinf=0)
			
			f_z = np.nan_to_num(self.alpha+(1-self.alpha)*z-np.power(z,1-self.alpha), posinf=0)
			f_z_F = np.nan_to_num(self.alpha+(1-self.alpha)*z_F-np.power(z_F,1-self.alpha), posinf=0)
			f_z_G = np.nan_to_num(self.alpha+(1-self.alpha)*z_G-np.power(z_G,1-self.alpha), posinf=0)

			D_alpha = np.sum(np.multiply(X,f_z))
			D_alpha_F = np.sum(np.multiply(I_g,f_z_F))
			D_alpha_G = np.sum(np.multiply(I_s,f_z_G))
			c = D_alpha + self.landa * D_alpha_F + self.mu * D_alpha_G

			end_time = datetime.now()

			iteration += 1
			if (np.abs(c - c_init)  > self.tol and iteration < self.max_iter): 
				c_init = c
				change = True
				c_list.append(c)
				D_alpha_F_list.append(D_alpha_F)
				D_alpha_G_list.append(D_alpha_G)
				runtime.append(format(end_time - start_time))

		self.max_iter = iteration
		self.runtime = runtime
		self.criterion = c
		self.criterions = c_list
		self.soft_matrix = F@S@G.T
		self.hard_matrix = F_cluster@S@G_cluster.T
		self.rowcluster_matrix = F_cluster@F_cluster.T@X
		self.reorganized_matrix = F.T@X@G
		self.row_labels_ = [x+1 for x in np.sort(np.argmax(F, axis =1)).tolist()]
		self.column_labels_ = [x+1 for x in np.sort(np.argmax(G, axis =1)).tolist()]
		self.orthogonality_F = D_alpha_F_list
		self.orthogonality_G = D_alpha_G_list

# In[ ]:





