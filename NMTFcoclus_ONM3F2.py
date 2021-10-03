#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def ONMTF(X, rank1, rank2, max_iter=100, F_init=None, G_init=None, S_init=None):
	

	m, n = X.shape
	F = rand(m, rank1) if isinstance(F_init, type(None)) else F_init
	S = rand(rank1, rank2) if isinstance(S_init, type(None)) else S_init
	G = rand(n, rank2) if isinstance(G_init, type(None)) else G_init


	for itr in range(max_iter):
	    

		
	#df DF and DG
	DF = np.diag(F.sum(axis=0))
	DG = np.diag(G.sum(axis=0)) #rank2*rank2

	#Normalization

	F = F.dot(np.linalg.inv(DF))
	S = DF.dot(S).dot(np.linalg.inv(DG))
	G = DG.dot(G.T)   #rank2*n
	soft_matrix = F.dot(S).dot(G)
	
	F_indicator = np.zeros_like(F)
	F_indicator[np.arange(len(F)), np.argpartition(F, -1, axis = 1)[:,-1]] = 1

	G_indicator = np.zeros_like(G)
	G_indicator[np.argpartition(G, -1, axis = 0)[-1,:], np.arange(len(G.T))] = 1
	
	hard_matrix = F_indicator.dot(S).dot(G_indicator)


	return F_indicator, G_indicator, S, soft_matrix, hard_matrix, F_indicator.T@F_indicator, G_indicator.T@G_indicator

