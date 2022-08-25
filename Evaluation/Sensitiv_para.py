


from NMTFcoclust.Models import NMTFcoclust_OPNMTF_alpha
from NMTFcoclust.Evaluation import EV
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def sensitive_orth_para(X, true_labels, n_row_clusters = None, n_col_clusters = None, landa = np.arange(0.01, 0.05,0.01) , mu = np.arange(1.01, 1.05,0.01) ):

	Acc = []
	NMI = []
	ARI = []
	AMIS = []
	runtime = []
	Orth_F = []
	Orth_G = []
	list_landa_mu_pairs = []
	for i in np.arange(len(landa)):
		for j in np.arange(len(mu)):
			for h in np.arange(10):

				OPNMTF_alpha = NMTFcoclust_OPNMTF_alpha.OPNMTF(n_row_clusters, n_col_clusters, landa = landa[i],  mu = mu[j],  alpha = 1.1, max_iter = 1)
				OPNMTF_alpha.fit(X)
				predicted_labels = np.sort(OPNMTF_alpha.row_labels_)
				pair = (landa[i] , mu[j])
				list_landa_mu_pairs.append(pair)
				from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
				from sklearn.metrics import accuracy_score as acc
				from sklearn.metrics import adjusted_rand_score as ars
				from sklearn.metrics import adjusted_mutual_info_score as amis

				NMI.append(nmi(true_labels, predicted_labels))
				Acc.append(acc(true_labels, predicted_labels))
				ARI.append(ars(true_labels, predicted_labels))
				AMIS.append(amis(true_labels, predicted_labels))
				Orth_F.append(OPNMTF_alpha.orthogonality_F)
				Orth_G.append(OPNMTF_alpha.orthogonality_G)
				runtime.append(OPNMTF_alpha.runtime)
	return 	[Acc, NMI, ARI, AMIS, Orth_F, Orth_G, runtime, list_landa_mu_pairs]

