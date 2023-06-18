# **NMTFcoclust**  
### **NMTFcocluster** (Non-negative Matrix Tri-Factorization for Co-clustering) is a package that implements decomposition on a data matrix $\mathbf{X}$ (document-word matrix and so on) with finding three  matrices $\mathbf{F}$ (roles membership rows), $\mathbf{G}$ (roles membership columns), and $\mathbf{S}$ (roles summary matrix) based on optimized $\alpha$-divergence.

 The low-rank approximation of $\mathbf{X}$ by
     $$\mathbf{X} \approx \mathbf{FSG}^{\top}$$.

![NMTF](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/nmtf7.png?raw=true)


### Brief description of models
NMTFcoclust implements the proposed algorithm (**OPNMTF**) and some NMTF according to the objective functions below:
- [**OPNMTF**](https://www.sciencedirect.com/science/article/abs/pii/S095741742301182X) 
```math
D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{\top})+
  \lambda \; D_{\alpha}(\mathbf{I}_{g}||\mathbf{F}^{\top}\mathbf{F})+
  \mu \; D_{\alpha}(\mathbf{I}_{s}||\mathbf{G}^{\top}\mathbf{G})
```
- [PNMTF](https://www.sciencedirect.com/science/article/abs/pii/S0957417417300283)
```math
 0.5||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}+0.5 \tau \; Tr(\mathbf{F} \Psi_{g}\mathbf{F}^{\top})+0.5 \eta \; Tr(\mathbf{G} \Psi_{s}\mathbf{G}^{\top})+ 0.5 \gamma \; Tr(\mathbf{S}^{\top}\mathbf{S})
```
- [ONMTF](https://www.sciencedirect.com/science/article/abs/pii/S0306457310000038)
```math
	0.5 ||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}
```
- [NBVD](https://dl.acm.org/doi/abs/10.1145/1081870.1081949)
 $$||\mathbf{X}-\mathbf{FSG}^{\top}||^{2}$$
- [ONM3T](https://dl.acm.org/doi/abs/10.1145/1150402.1150420)
```math
	||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}+ 
 Tr(\Lambda (\mathbf{F}^{\top}\mathbf{F}-\mathbf{I}_{s}))+ 
 Tr(\Gamma (\mathbf{G}^{\top}\mathbf{G}-\mathbf{I}_{g}))
```
- [ODNMTF](https://link.springer.com/chapter/10.1007/978-3-642-24958-7_82)
```math
 ||\mathbf{X}-\mathbf{FF^{\top}XGG}^{\top}||^{2}+ Tr(\Lambda \mathbf{F}^{\top})+ Tr( \Gamma \mathbf{G}^{\top})
```
- [DNMTF](https://link.springer.com/chapter/10.1007/978-3-642-24958-7_82)
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


### [Datasets](https://github.com/Saeidhoseinipour/NMTFcoclust/tree/master/Datasets)

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


                                                                   # Read Data Sets ------->  Classic3

file_name=r"NMTFcoclust\Dataset\Classic3\classic3.mat"
mydata = loadmat(file_name)

                                                                    # Data matrix 
X_Classic3 = mydata['A'].toarray()
X_Classic3_sum_1 = X_Classic3/X_Classic3.sum()
                                                                   
true_labels = mydata['labels'].flatten().tolist()                   # True labels list [0,0,0,..,1,1,1,..,2,2,2]  n_row_cluster = 3
true_labels = [x+1 for x in true_labels]                            # True labels list [1,1,1,..,2,2,2,..,3,3,3]  n_row_cluster = 3
print(confusion_matrix(true_labels, true_labels))


```

### Model

```python
from NMTFcoclust.Models.NMTFcoclust_OPNMTF_alpha_2 import OPNMTF
from NMTFcoclust.Evaluation.EV import Process_EV

OPNMTF_alpha = OPNMTF(n_row_clusters = 3, n_col_clusters = 3, landa = 0.3,  mu = 0.3,  alpha = 0.4, max_iter=1)
OPNMTF_alpha.fit(X_Classic3_sum_1)
Process_Ev = Process_EV( true_labels ,X_Classic3_sum_1, OPNMTF_alpha) 



Accuracy (Acc):0.9100488306347982
Normalized Mutual Info (NMI):0.7703948803438703
Adjusted Rand Index (ARI):0.7641161476685447

Confusion matrix   (CM):
[[1033    0    0]
 [ 276 1184    0]
 [   0   74 1324]]
Total Time:  26.558243700000276
```

![DC](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/WC_classic3.png?raw=true)

### Cite
Please cite the following paper in your publication if you are using [NMTFcoclust]() in your research:

```bibtex
 @article{NMTFcoclust, 
    title={Orthogonal Parametric Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Co-clustering}, 
    DOI={10.1016/j.eswa.2023.120680}, 
    journal={Expert Systems with Applications}, 
    author={Saeid Hoseinipour, Mina Aminghafari, Adel Mohammadpour}, 
    year={2023}
} 
```

-[For 50 day article is free in Expert Systems With Applications](https://pdf.sciencedirectassets.com/271506/1-s2.0-S0957417423X00177/1-s2.0-S095741742301182X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBoaCXVzLWVhc3QtMSJHMEUCIE%2FMTa4OPpt6ORN8h%2BXGLiAw1Rov15FpDwCRQgVRDfr5AiEA28bU4Wyx0Qp8Qimo5z%2FV1tGV2YmnoFhkLhYLVtQ46yEqsgUIchAFGgwwNTkwMDM1NDY4NjUiDA7cGsIC%2F11WW3ELvyqPBUrj6hiol%2B%2FibqdN41K%2BguRJuQL4cI7kJ96Ut07e89JmlovaaRTXbXDVkmslf8D11CkWF3zgMK8jizMa3u%2Fm0gVIv39atPra9hqPHE%2F0qJWKdmHzxOQRkL%2F5wTEJzUyuC2jTTlyXODhTLb03IKH5pKusyB1mU6IW16mz2Y8RrqoesXtK1Ku46U85AjRMtjreqJIRJTP9iRsmwwFfapcvRXx3SRTbp6MICsaD9cXw3SLtDvDiRcpxo8A5Sadqz1CVBsmT%2BZcN4WGgJHWPCGE2tE8ulTHyPlykSc1mVIWgrgLMihkoc3XWLUXKseRZPLRQi6bBHukIAt10cxkpaCsyyCREXF1Olgnnw3QXUPL%2FY72T3N%2BK8TmvtPvTvECROgJSABp9Tpy65o%2ByQKRuLT4GHUQ4r5hLIPNs%2F4QboZHzjLhKITmDEUEehVvFWNRiK0AhkXowwCIPybVfDdwL1r%2FDmQP%2BC06v4g1m%2BajBpW8jcAsQyIw2tMtBWPWJOha9j8So1WA4cBc1%2Fhh89PJriceYzr9VajX1kD5s4lIzly1a3nCtMsAF%2Bwz7qJsgsQfm0dJKRjMkEksan9bSjGadLluf%2F45l8GbuSgPf0GH3n%2FbfuJH%2F7fwRuUM6PHqF7Lk0sWK36%2FhnQrjy560tfMeVzrFcNtTQ6mOeKVmpdDwCtE3xM1H%2BrTT5HyXi1Yfl4JedxrL9ys6551QUErDEgNk5aG4%2FIy8aicbiNfoDjPqxMDyGp8xcIYhsj0QLHk%2FFq6pNEaqvi2wUOXMfwnvUHpteRlTk60pD5hdQRXiTGGLCbOWh0cQWEZhSASGizArlCKHR%2FpvP5CcfR%2FwIcooTZtA2lC6yOrLRUEDhPj94UUtov8BprmcwgJm7pAY6sQFoKHL%2Bg%2FMRjD97PpeapzDdiCFCTHPDBbSF5ZeOE5TvvnZB00uIqFx8iEvtWxKLTFY9JwrWKGuCqB2sgUQhtIpNIKIXubCkDK7W%2BwunbUqzIt08FazYWJpl7I1ALU%2BML7xqTUCg9DxyuKiP2nY%2Ff8Ml%2BzYCU%2F6VbZQyoyyCTnCUSacTlWPA8uah2GhTauQlB%2FP9yXgKMNioF0l%2FqSDtKyiE3UPDFJyL0QeVdbdAcyGzJo4%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230618T101926Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3URYNLBQ%2F20230618%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0f238cf956d9bc91b08aa960b9a1c4e4cc17658d2bd55d4b8db356497d1e1727&hash=463f074364c5510892fef9323fdf4efb8fe3ae64183a8cf6383b0d195a663646&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S095741742301182X&tid=spdf-2762f286-7c38-4a9f-866d-ee7a83431d01&sid=fdc209922bf74845887919b12059e80fe22fgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0b0a520b5d5002075206&rr=7d92cb3d5f4cb966&cc=nl)
### Supplementary material
OPNMTF implements on synthetic datasets such as Bernoulli, Poisson, and Truncated Gaussian.
[Available from GitHub](https://github.com/Saeidhoseinipour/NMTFcoclust/tree/master/Supplementary%20material)
[Available from ESWA](https://ars.els-cdn.com/content/image/1-s2.0-S095741742301182X-mmc1.pdf)
### References

[1] [Wang et al, Penalized nonnegative matrix tri-factorization for co-clustering (2017), Expert Systems with Applications.](https://www.sciencedirect.com/science/article/abs/pii/S0957417417300283)

[2] [Yoo et al, Orthogonal nonnegative matrix tri-factorization for co-clustering: Multiplicative updates on Stiefel manifolds (2010), Information Processing and Management.](https://www.sciencedirect.com/science/article/abs/pii/S0306457310000038)
	
[3] [Ding et al, Orthogonal nonnegative matrix tri-factorizations for clustering (2008), Proceedings of the 12th ACM SIGKDD International Conference on Knowledge 	Discovery and Data Mining.](https://dl.acm.org/doi/abs/10.1145/1150402.1150420)

[4] [Long et al, Co-clustering by block value decomposition (2005), Proceedings of the Eleventh ACM SIGKDD International Conference on Knowledge Discovery in Data 	Mining.](https://dl.acm.org/doi/abs/10.1145/1081870.1081949)

[5] [Labiod et al, Co-clustering under nonnegative matrix tri-factorization (2011), International Conference on Neural Information Processing.](https://link.springer.com/chapter/10.1007/978-3-642-24958-7_82)

[6] [Li et al, Nonnegative Matrix Factorization on Orthogonal Subspace (2010), Pattern Recognition Letters.](sciencedirect.com/science/article/abs/pii/S0167865509003651)

[7] [Li et al, Nonnegative Matrix Factorizations for Clustering: A Survey (2019), Data Clustering.](https://www.taylorfrancis.com/chapters/edit/10.1201/9781315373515-7/nonnegative-matrix-factorizations-clustering-survey-tao-li-cha-charis-ding)

[8] [Cichocki et al, Non-negative matrix factorization with $\alpha$-divergence (2008), Pattern Recognition Letters.](https://www.sciencedirect.com/science/article/abs/pii/S0167865508000767)
