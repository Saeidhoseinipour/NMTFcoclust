# **NMTFcoclust**  
### **NMTFcocluster** (Non-negative Matrix Tri-Factorization for coclustering) is a package that implements decomposition on a data matrix $\mathbf{X}$ (document-word matrix and so on) with finding three  matrices $\mathbf{F}$ (roles membership rows), $\mathbf{G}$ (roles membership columns) and $\mathbf{S}$ (roles summary matrix) based optimazed $\alpha$-divergence.

 The low-rank approximation of $\mathbf{X}$ by
     $$\mathbf{X} \approx \mathbf{FSG}^{\top} $$
where $n$, $m$, $g \leqslant n$ and $s \leqslant m$ are the number of rows, columns, row clusters and column clusters, respectively.


![NMTF](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/nmtf7.png?raw=true)


### Brief description 
NMTF coclust implements three proposed algorithms and some NMTF according to objective functions below:
- $OPNMTF_{\alpha}$ 
```math
D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{\top})+
  \lambda \; D_{\alpha}(\mathbf{I}_{g}||\mathbf{F}^{\top}\mathbf{F})+
  \mu \; D_{\alpha}(\mathbf{I}_{s}||\mathbf{G}^{\top}\mathbf{G})
```
- $ONMTF_{\alpha}$
```math
   D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{\top})+
   \delta Tr(\mathbf{F}\Psi_{g}\mathbf{F}^{\top}) +	
   \beta Tr(\mathbf{G} \Psi_{s}\mathbf{G}^{\top})
```
- $NMTF_{\alpha}$
 $$D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{\top})$$ 
- $PNMTF$
```math
 0.5||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}+0.5 \tau \; Tr(\mathbf{F} \Psi_{g}\mathbf{F}^{\top})+0.5 \eta \; Tr(\mathbf{G} \Psi_{s}\mathbf{G}^{\top})+ 0.5 \gamma \; Tr(\mathbf{S}^{\top}\mathbf{S})
```
- $NBVD$
 $$||\mathbf{X}-\mathbf{FSG}^{\top}||^{2}$$
- $ONM3T$
```math
	||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}+ 
 Tr(\Lambda (\mathbf{F}^{\top}\mathbf{F}-\mathbf{I}_{s}))+ 
 Tr(\Gamma (\mathbf{G}^{\top}\mathbf{G}-\mathbf{I}_{g}))
```
- $ODNMTF$
```math
 ||\mathbf{X}-\mathbf{FF^{\top}XGG}^{\top}||^{2}+ Tr(\Lambda \mathbf{F}^{\top})+ Tr( \Gamma \mathbf{G}^{\top})
```
- $DNMTF$
```math
 ||\mathbf{X}-\mathbf{FF^{\top}XGG}^{\top}||^{2}
```

### Requirements
```python
numpy==1.18.3
pandas==1.0.3
scipy==1.4.1
matplotlib==3.0.3
scikit-learn==0.22.2.post1
coclust==0.2.1

```
### Installing NMTFcoclust

### License

### Examples

#### Datasets

| Datasets | Documents | Words | Sporsity | Number of clusters |
| -- | ----------- | -- | -- | -- |
| [CSTR](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/cstr.mat) | 475 | 1000 | 96% | 4 |
| [WebACE](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/WebACE..mat) |2340  |1000  | 91.83% |20  |
| [Classic3](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/classic3.mat) |3891  |4303  |98%  |3  |
| [Sports](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/sports..mat) |8580  |14870  | 99.99% |7  |
| [Reviews](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/reviews..mat) |4069  |18483  | 99.99% |5  |
| [RCV1_4Class](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_4Class.mat) |9625  |29992  | 99.75% |4  |
| [NG20](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/NG20..mat) |19949  | 43586 | 99.99% |20  |


- [20Newsgroups](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/20Newsgroups.mat)
- [TDT2](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/TDT2..mat)
- [Reuters21578](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/Reuters21578..mat)
- [RCV1_ori](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_ori..mat)
```python
import pandas as pd 
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix 

#from coclust.io.data_loading import load_doc_term_data

                                                                   # Read Data Sets ------->  Classic3

file_name=r"D:\My paper\Application\NMTFcoclust\Dataset\Classic3\classic3.mat"
mydata = loadmat(file_name)
#print(mydata['A'])
#matlab_dict=pd.read_csv(mydata)
#matlab_dict=numpy.loadtxt(file_name)
#type(mydata) ---> dic		

#mydata=mydata.copy()


                                                                    # Data matrix 
X_Classic3 = mydata['A'].toarray()
print(X_Classic3)
X_Classic3_sum_1 = X_Classic3/X_Classic3.sum()

E_m1 = np.ones((4303, 1))
X_Classic3_normalization = X_Classic3@np.diagflat(np.power(X_Classic3.T@X_Classic3@E_m1, -0.5))
#print(X_Classic3_normalization)
#print(X_Classic3.sum(axis=0).shape)
freq = X_Classic3.sum(axis = 0)
#print(list(freq))
                                                                    
true_labels = mydata['labels'].flatten().tolist()                   # True labels list [0,0,0,..,1,1,1,..,2,2,2]  n_row_cluster = 3
true_labels = [x+1 for x in true_labels]                            # True labels list [1,1,1,..,2,2,2,..,3,3,3]  n_row_cluster = 3
print(confusion_matrix(true_labels, true_labels))

#print(true_labels)                                               
#print(mydata['ts'].shape)
#print(mydata['ms'].shape)
#print(X_Classic3,X_Classic3_normalization.sum(),X_Classic3_sum_1.sum())


```

## Model
```python
from NMTFcoclust.Models import NMTFcoclust_ONMTF_alpha
from NMTFcoclust.Models import NMTFcoclust_NMTF_alpha
from NMTFcoclust.Models import NMTFcoclust_PNMTF
from NMTFcoclust.Models import NMTFcoclust_ONM3F
from NMTFcoclust.Models import NMTFcoclust_ONMTF
from NMTFcoclust.Models import NMTFcoclust_NBVD
from NMTFcoclust.Models import NMTFcoclust_ODNMF
from NMTFcoclust.Models import NMTFcoclust_DNMF
```
```python
from NMTFcoclust.Models.NMTFcoclust_OPNMTF_alpha_2 import OPNMTF
from NMTFcoclust.Evaluation.EV import Process_EV

OPNMTF_alpha = OPNMTF(n_row_clusters = 3, n_col_clusters = 3, landa = 0.3,  mu = 0.3,  alpha = 0.4, max_iter=1)
OPNMTF_alpha.fit(X_Classic3_sum_1)
Process_Ev = Process_EV( true_labels ,X_Classic3_sum_1, OPNMTF_alpha) 



Accuracy (Acc):0.9100488306347982
Normalized Mutual Info (NMI):0.7703948803438703
Adjusted Rand Index (ARI):0.7641161476685447
Adjusted Mutual Info (AMI):0.7702867787943636
Intra-cluster Average Similarity (IAS):0.027380015679156534
Inter-cluster Centroids Average Similarity (ICAS):0.335635399782488
Runtime:4.049925799999983
Confusion matrix   (CM):
[[1033    0    0]
 [ 276 1184    0]
 [   0   74 1324]]
Total Time:  26.558243700000276
```

![WC](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/WC_1_5_bold_31_32_11_22_33_v2.png?raw=true)


### Cite
Please cite the following paper in your publication if you are using [NMTFcoclust]() in your research:

```bibtex
 @article{NMTFcoclust, 
    title={Orthogonal Parametric Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Co-clustering}, 
    DOI={10.1007/s11063-021-10528-4}, 
    journal={Information processing and management}, 
    author={Saeid Hoseinipour, Mina Aminghafari, Adel Mohammadpour}, 
    year={2023}
} 
```
### References

