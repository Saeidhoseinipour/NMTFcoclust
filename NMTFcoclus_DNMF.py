
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




class DNMF:
	def __init__(self,
		 n_row_clusters=2, n_col_clusters=2,  
		 F_init=None, S_init=None, G_init=None,
		 max_iter=100, n_init=1, tol=1e-9, 
		 random_state=None):
		self.n_row_clusters = n_row_clusters
		self.n_col_clusters = n_col_clusters
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
		X = X.astype(int)

		random_state = check_random_state(self.random_state) 
		seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
		for seed in seeds:
			self._fit_single(X, seed, y)
			if np.isnan(self.criterion):
				raise ValueError("matrix may contain negative or unexpected NaN values")
			# remember attributes corresponding to the best criterion
			if self.criterion > criterion: 
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

	def _fit_single(self, X, random_state, y=None):
		m, n = X.shape
		rank1 = self.n_row_clusters
		rank2 = self.n_col_clusters
		F = rand(m, rank1) if isinstance(self.F_init, type(None)) else self.F_init
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
					X_G = X@G@G.T
					enum = 2*X@X_G.T@F
					denom = F@F.T@X_G@X_G.T@F + X_G@X_G.T@F@F.T@F
					F = F * (enum/denom)

				if isinstance(self.G_init, type(None)):
					X_F = F@F.T@X
					enum = 2*X_F.T@X@G
					denom = G@G.T@X_F.T@X_F@G + X_F.T@X_F@G@G.T@G
					G = G * (enum / denom)

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