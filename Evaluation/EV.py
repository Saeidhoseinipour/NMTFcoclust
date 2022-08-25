import argparse
import logging
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import adjusted_rand_score 
from sklearn.metrics import adjusted_mutual_info_score 
from sklearn.metrics import confusion_matrix

def Process_EV(args, model):

            nmi = normalized_mutual_info_score(args, np.sort(model.row_labels_))
            ari = adjusted_rand_score(args, np.sort(model.row_labels_))
            amis = adjusted_mutual_info_score(args, np.sort(model.row_labels_))
            acc = accuracy_score(args, np.sort(model.row_labels_))
            cm = confusion_matrix(args, np.sort(model.row_labels_))
            
            print("Accuracy              ======>"             + str(acc))
            print("NMI                   ======>"             + str(nmi))
            print("Adjusted Rand Index   ======>"             + str(ari))
            print("Adjusted Mutual Info  ======>"             + str(amis))
            print("Runtime               ======>"             + str(model.runtime))
            print("Confusion matrix      ======>")
            print(cm)